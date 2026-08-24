"""Exact per-request checkpoints for stochastic lm-eval generation.

Unlike lm-eval's ordinary response cache, this wrapper is safe for sampling:
it records responses only as a completed prefix of the model's batch-1 request
order and atomically stores the Python, NumPy, Torch, and CUDA RNG states after
every request.  Resume restores the state after that prefix before generating
the remaining requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from lm_eval.api.model import LM


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _stable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if key != "past_key_values"
        }
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _request_hash(req: Iterable[Any]) -> str:
    context, gen_kwargs = tuple(req)
    payload = json.dumps([context, _stable(gen_kwargs)], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class _CheckpointHook:
    def __init__(self, owner: "ExactSamplingCheckpointLM") -> None:
        self.owner = owner

    def add_partial(self, attr: str, req: Iterable[Any], res: Any) -> None:
        if attr == "generate_until":
            self.owner._record(req, res)


class ExactSamplingCheckpointLM(LM):
    """Proxy an lm-eval LM while checkpointing stochastic batch-1 requests."""

    def __init__(self, lm, checkpoint: Path, signature: dict[str, Any]) -> None:
        super().__init__()
        batch_size = getattr(lm, "batch_size", None)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("exact sampling checkpoints require a fixed positive batch size")
        self.lm = lm
        self._rank = lm.rank
        self._world_size = lm.world_size
        self._device = lm.device
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.signature = _stable(signature)
        self.state = {
            "version": 1,
            "signature": self.signature,
            "sequence": [],
            "responses": {},
            "rng_after": None,
        }
        if checkpoint.is_file():
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if loaded.get("signature") != self.signature:
                raise RuntimeError(f"sampling checkpoint signature mismatch: {checkpoint}")
            self.state = loaded
        self._expected_remaining: list[str] = []
        self._record_base = 0
        self.lm.set_cache_hook(_CheckpointHook(self))

    def __getattr__(self, name: str):
        return getattr(self.lm, name)

    def loglikelihood(self, requests):
        return self.lm.loglikelihood(requests)

    def loglikelihood_rolling(self, requests):
        return self.lm.loglikelihood_rolling(requests)

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.lm.apply_chat_template(chat_history, add_generation_prompt)

    @property
    def tokenizer_name(self) -> str:
        return self.lm.tokenizer_name

    def chat_template(self, chat_template: bool | str = False):
        return self.lm.chat_template(chat_template)

    def _ordered(self, requests) -> list:
        # Reuse lm-eval's own grouping/sorting implementation so cache object
        # identity and group insertion order match HFLM exactly.
        from lm_eval.models.utils import Collator

        collator = Collator(
            requests,
            sort_fn=lambda request: (
                -len(self.lm.tok_encode(request.args[0])),
                request.args[0],
            ),
            group_by="gen_kwargs",
            group_fn=lambda request: request.args[1],
        )
        return [request for chunk in collator.get_batched(n=self.batch_size) for request in chunk]

    def generate_until(self, requests):
        ordered = self._ordered(requests)
        ordered_hashes = [_request_hash(request.args) for request in ordered]
        if len(set(ordered_hashes)) != len(ordered_hashes):
            raise RuntimeError("exact sampling checkpoint does not support duplicate requests")
        completed = self.state["sequence"]
        if ordered_hashes[: len(completed)] != completed:
            raise RuntimeError("sampling checkpoint is not a prefix of the current request order")
        if len(completed) % self.batch_size and len(completed) != len(ordered_hashes):
            raise RuntimeError("sampling checkpoint prefix does not end at a batch boundary")

        responses = self.state["responses"]
        remaining = [request for request in requests if _request_hash(request.args) not in responses]
        if completed:
            _restore_rng(self.state["rng_after"])
        self._expected_remaining = ordered_hashes[len(completed):]
        self._record_base = len(completed)
        new_results = self.lm.generate_until(remaining) if remaining else []
        new_iter = iter(new_results)
        merged = [
            responses[_request_hash(request.args)]
            if _request_hash(request.args) in responses
            else next(new_iter)
            for request in requests
        ]
        return merged

    def _record(self, req: Iterable[Any], response: Any) -> None:
        request_hash = _request_hash(req)
        index = len(self.state["sequence"]) - self._record_base
        if index >= len(self._expected_remaining) or self._expected_remaining[index] != request_hash:
            raise RuntimeError("lm-eval generated requests in an unexpected order")
        self.state["sequence"].append(request_hash)
        self.state["responses"][request_hash] = response
        self.state["rng_after"] = _rng_state()
        completed_this_run = len(self.state["sequence"]) - self._record_base
        at_batch_boundary = completed_this_run % self.batch_size == 0
        at_final_request = len(self.state["sequence"]) == self._record_base + len(self._expected_remaining)
        if at_batch_boundary or at_final_request:
            self._save()

    def _save(self) -> None:
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint.with_suffix(self.checkpoint.suffix + ".tmp")
        torch.save(self.state, temporary)
        os.replace(temporary, self.checkpoint)

    def discard(self) -> None:
        self.checkpoint.unlink(missing_ok=True)


__all__ = ["ExactSamplingCheckpointLM"]
