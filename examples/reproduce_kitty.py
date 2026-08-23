"""Reproduce Kitty-style KV-cache accuracy runs with the local lm-eval harness.

The paper's accuracy simulation compares FP16, KIVI-2, KIVI*-2, Kitty, and
Kitty-Pro under the same prompt/task seeds.  This runner keeps that matrix in a
single explicit command while using the local Dart cache implementation.  It
is intentionally resumable at the process level: run one variant at a time so
model memory is released between configurations.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
from pathlib import Path
from typing import Any

import torch


VARIANTS = {
    "fp16": {
        "label": "K16V16",
        "cache": "dense",
    },
    "kivi-2": {
        "label": "KIVI-K2V2",
        "cache": "dart",
        "sink_tokens": 0,
        "promote_ratio": 0.0,
    },
    "kivi-2-sink": {
        "label": "KIVI-K2V2*",
        "cache": "dart",
        "sink_tokens": 32,
        "promote_ratio": 0.0,
    },
    "kitty": {
        "label": "Kitty",
        "cache": "dart",
        "sink_tokens": 32,
        "promote_ratio": 0.125,
    },
    "kitty-pro": {
        "label": "Kitty-Pro",
        "cache": "dart",
        "sink_tokens": 32,
        "promote_ratio": 0.25,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--task", required=True, help="lm-eval task, e.g. gsm8k_cot_llama or aime24")
    parser.add_argument("--variant", choices=tuple(VARIANTS) + ("all",), default="kitty-pro")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--buffer-length", type=int, default=128)
    parser.add_argument("--sink-length", type=int, default=32)
    parser.add_argument("--kbits", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--vbits", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--promote-bit", type=int, choices=(4,), default=4)
    parser.add_argument("--channel-selection", choices=("magnitude", "variance"), default="magnitude")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Limit task samples for a smoke run")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--output", default="results/kitty_repro")
    parser.add_argument("--confirm-run-unsafe-code", action="store_true")
    return parser


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items() if key != "gen_kwargs"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _cache_for_variant(args: argparse.Namespace, variant: str):
    config = VARIANTS[variant]
    if config["cache"] == "dense":
        from transformers import DynamicCache

        return DynamicCache(), None
    from dart import DartKVCacheConfig
    from dart.integrations import DartHFCache

    cache_config = DartKVCacheConfig(
        key_bits=args.kbits,
        value_bits=args.vbits,
        key_group_size=args.group_size,
        value_group_size=args.group_size,
        sink_tokens=int(config["sink_tokens"]),
        page_size=args.buffer_length,
        local_tokens=args.buffer_length,
        hold_partial_pages=True,
        promote_bits=args.promote_bit,
        promote_ratio=float(config["promote_ratio"]),
        channel_selection=args.channel_selection,
        metadata_dtype=torch.float16,
    )
    return DartHFCache(cache_config), cache_config


def _task_fewshot(task: str) -> int:
    task_lower = task.lower()
    for key, value in (("mmlu", 4), ("gsm8k", 8), ("gpqa", 5), ("math", 4)):
        if key in task_lower:
            return value
    return 0


def run_variant(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    from lm_eval.models.huggingface import HFLM
    from lm_eval.evaluator import simple_evaluate
    from transformers import AutoModelForCausalLM

    model_path = Path(args.model).expanduser()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"no local config.json found under {model_path}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_dtype(args.dtype),
        local_files_only=True,
        attn_implementation=args.attn_implementation,
    ).to(device).eval()
    cache, cache_config = _cache_for_variant(args, variant)
    lm = HFLM(pretrained=model, batch_size=args.batch_size)
    gen_kwargs: dict[str, Any] = {"past_key_values": cache, "max_gen_toks": args.max_new_tokens}
    task_lower = args.task.lower()
    result = None
    for repeat in range(args.repeats):
        repeat_seed = args.seed + repeat
        result = simple_evaluate(
            model=lm,
            tasks=[args.task],
            num_fewshot=_task_fewshot(args.task),
            batch_size=args.batch_size,
            limit=args.limit,
            log_samples=True,
            apply_chat_template="aime" in task_lower,
            fewshot_as_multiturn=False,
            gen_kwargs=gen_kwargs,
            random_seed=repeat_seed,
            numpy_random_seed=1234 + repeat,
            torch_random_seed=1234 + repeat,
            fewshot_random_seed=1234 + repeat,
            confirm_run_unsafe_code=args.confirm_run_unsafe_code,
        )
        if result is None:
            raise RuntimeError("lm-eval returned no result on the local rank")
        output_dir = Path(args.output) / Path(args.model).name / args.task / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        repeat_path = output_dir / f"repeat_{repeat}.json"
        repeat_path.write_text(json.dumps({
            "variant": variant,
            "label": VARIANTS[variant]["label"],
            "repeat": repeat,
            "seed": repeat_seed,
            "model": str(model_path),
            "task": args.task,
            "config": vars(args),
            "cache_config": _jsonable(cache_config.__dict__) if cache_config is not None else None,
            "results": _jsonable(result.get("results", {})),
            "samples": _jsonable(result.get("samples", {})),
        }, ensure_ascii=False, indent=2) + "\n")
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    output = {
        "variant": variant,
        "label": VARIANTS[variant]["label"],
        "model": str(model_path),
        "task": args.task,
        "repeats": args.repeats,
        "limit": args.limit,
        "dtype": args.dtype,
        "device": str(device),
        "seed": args.seed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "peak_memory_bytes": peak_memory,
        "results": _jsonable(result.get("results", {})),
        "cache_config": _jsonable(cache_config.__dict__) if cache_config is not None else None,
    }
    del lm, model, cache
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    summary_path = Path(args.output) / Path(args.model).name / args.task / f"{variant}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    output["result_path"] = str(summary_path)
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.repeats <= 0 or args.max_new_tokens <= 0:
        raise SystemExit("batch-size, repeats, and max-new-tokens must be positive")
    variants = tuple(VARIANTS) if args.variant == "all" else (args.variant,)
    outputs = [run_variant(args, variant) for variant in variants]
    print(json.dumps(outputs[0] if len(outputs) == 1 else outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
