"""Hugging Face Cache adapter for the PyTorch DartKV reference path."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ..cache import DartKVCache, DartKVCacheConfig

try:
    from transformers import Cache
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    raise ImportError(
        "DartHFCache requires transformers; install the DartKV model dependencies first."
    ) from exc


class DartHFCache(Cache):
    """A standard Transformers cache facade backed by compressed Dart layers.

    ``update`` returns materialized tensors because standard eager/SDPA model
    attention expects dense K/V tensors. It mirrors Kitty's ``PostQuant``
    lifecycle: the current attention call receives the cache state before a
    newly eligible page is quantized, while the backing store is packed for the
    next call. This adapter is therefore a correctness and lifecycle milestone,
    not yet a fused-memory attention implementation.
    """

    is_compileable = False

    def __init__(self, config: Optional[DartKVCacheConfig] = None) -> None:
        super().__init__()
        self.dart_config = config or DartKVCacheConfig()
        self._layers: Dict[int, DartKVCache] = {}
        self._last_update_prefill: Dict[int, bool] = {}
        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused_update = bool(cache_kwargs and cache_kwargs.get("dart_fused_attention"))
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        layer = self._layers.get(layer_idx)
        if layer is None:
            layer = DartKVCache(self.dart_config)
            self._layers[layer_idx] = layer
            self._last_update_prefill[layer_idx] = True
            keys_to_return = key_states.detach().clone()
            values_to_return = value_states.detach().clone()
        else:
            self._last_update_prefill[layer_idx] = False
            if fused_update:
                # The fused Qwen path consumes the packed cache directly; do
                # not materialize the complete historical K/V tensors just to
                # throw them away before launching page attention.
                keys_to_return = key_states.detach().clone()
                values_to_return = value_states.detach().clone()
            else:
                prior_keys, prior_values = layer.get()
                keys_to_return = torch.cat((prior_keys, key_states), dim=-2)
                values_to_return = torch.cat((prior_values, value_states), dim=-2)
        layer.append(key_states, value_states)
        return keys_to_return, values_to_return

    def layer_for_attention(self, layer_idx: int) -> DartKVCache:
        """Return a layer's packed cache for fused attention integration."""

        try:
            return self._layers[layer_idx]
        except KeyError as exc:
            raise RuntimeError(f"layer {layer_idx} has not been initialized") from exc

    def last_update_was_prefill(self, layer_idx: int) -> bool:
        """Whether the most recent update for ``layer_idx`` created the layer."""

        return self._last_update_prefill.get(layer_idx, False)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        layer = self._layers.get(layer_idx)
        return layer.get_seq_length() if layer is not None else 0

    def get_max_cache_shape(self) -> None:
        return None

    def page_table(
        self,
        layer_idx: int = 0,
        *,
        device: torch.device | str | None = None,
        rebuild: bool = False,
    ):
        """Expose a layer's cached Dart page table to fused attention callers."""

        layer = self._layers.get(layer_idx)
        return layer.page_table(device=device, rebuild=rebuild) if layer is not None else None

    def page_runs(
        self,
        layer_idx: int = 0,
        *,
        device: torch.device | str | None = None,
        rebuild: bool = False,
    ):
        """Expose a layer's cached uniform page runs."""

        layer = self._layers.get(layer_idx)
        return layer.page_runs(device=device, rebuild=rebuild) if layer is not None else None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> "DartHFCache":
        """Rebuild selected beams without retaining a dense fallback copy."""

        for layer_idx, layer in list(self._layers.items()):
            keys, values = layer.get()
            indices = beam_idx.to(keys.device)
            rebuilt = DartKVCache(self.dart_config)
            rebuilt.update(keys.index_select(0, indices), values.index_select(0, indices))
            self._layers[layer_idx] = rebuilt
        self._last_update_prefill.clear()
        return self

    @property
    def key_cache(self) -> list[torch.Tensor]:
        return [self._layers[index].key_states for index in sorted(self._layers)]

    @property
    def value_cache(self) -> list[torch.Tensor]:
        return [self._layers[index].value_states for index in sorted(self._layers)]

    @property
    def storage_bytes(self) -> int:
        return sum(layer.storage_bytes for layer in self._layers.values())

    @property
    def dense_bytes(self) -> int:
        return sum(layer.dense_bytes for layer in self._layers.values())

    @property
    def compression_ratio(self) -> float:
        return self.dense_bytes / self.storage_bytes if self.storage_bytes else 1.0


__all__ = ["DartHFCache"]
