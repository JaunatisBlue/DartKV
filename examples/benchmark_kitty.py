"""Benchmark Kitty-style dense/local-buffer generation on a local model.

This is the Dart-side counterpart of ``reference/code/Kitty/latency_benchmarking``.
It fixes the prompt, sequence length, batch size, warmup, repeat count, device,
and seed in the result JSON, then reports generation throughput and KV storage.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

from dart import DartKVCacheConfig
from dart.integrations import DartHFCache


DEFAULT_PROMPT = (
    "Given the following problem, think step by step and give a final answer to the problem. "
    "Problem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars "
    "are in the parking lot? The final answer is"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", choices=("dense", "dart", "kitty-reference"), default="dart")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--bits", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--promote-ratio", type=float, default=0.25)
    parser.add_argument("--sink-tokens", type=int, default=32)
    parser.add_argument("--buffer-length", type=int, default=128)
    parser.add_argument("--key-group-size", type=int, default=128)
    parser.add_argument("--value-group-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--output", default="results/kitty_latency")
    return parser


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dense_storage(cache) -> int:
    tensors = []
    for key_cache, value_cache in zip(cache.key_cache, cache.value_cache, strict=True):
        tensors.extend((key_cache, value_cache))
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _new_cache(args: argparse.Namespace):
    if args.cache == "dense":
        from transformers import DynamicCache

        return DynamicCache()
    if args.cache == "kitty-reference":
        reference_src = Path(__file__).resolve().parents[1] / "reference" / "code" / "Kitty" / "src"
        sys.path.insert(0, str(reference_src))
        from kitty_sim import KittyKVCache, KittyKVCacheConfig

        return KittyKVCache(KittyKVCacheConfig(
            sink_length=args.sink_tokens,
            buffer_length=args.buffer_length,
            group_size=args.key_group_size,
            kbits=args.bits,
            vbits=args.bits,
            promote_ratio=args.promote_ratio,
            promote_bit=4,
            channel_selection=1,
        ))
    return DartHFCache(DartKVCacheConfig(
        bits=args.bits,
        key_bits=args.bits,
        value_bits=args.bits,
        key_group_size=args.key_group_size,
        value_group_size=args.value_group_size,
        sink_tokens=args.sink_tokens,
        page_size=args.buffer_length,
        local_tokens=1,
        value_local_tokens=args.buffer_length,
        hold_partial_pages=True,
        promote_bits=4,
        promote_ratio=args.promote_ratio,
        metadata_dtype=torch.float16,
    ))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.batch_size, args.max_seq_len, args.warmup_runs, args.repeat_runs) <= 0:
        raise SystemExit("batch-size, max-seq-len, warmup-runs, and repeat-runs must be positive")
    model_path = Path(args.model).expanduser()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"no local config.json found under {model_path}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_dtype(args.dtype),
        local_files_only=True,
        attn_implementation=args.attn_implementation,
    ).to(device).eval()
    texts = [args.prompt] * args.batch_size
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    if input_ids.shape[-1] >= args.max_seq_len:
        raise SystemExit("max-seq-len must exceed the tokenized prompt length")

    def generate_once():
        cache = _new_cache(args)
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.max_seq_len,
                max_new_tokens=None,
                return_dict_in_generate=False,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                eos_token_id=None,
                use_cache=True,
                past_key_values=cache,
            )
        _sync(device)
        if isinstance(cache, DartHFCache):
            storage_bytes = cache.storage_bytes
            dense_bytes = cache.dense_bytes
            ratio = cache.compression_ratio
        else:
            storage_bytes = _dense_storage(cache)
            dense_bytes = storage_bytes
            ratio = 1.0
        del cache
        return output, storage_bytes, dense_bytes, ratio

    for _ in range(args.warmup_runs):
        generate_once()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    records = []
    for _ in range(args.repeat_runs):
        start = time.perf_counter()
        output, storage_bytes, dense_bytes, ratio = generate_once()
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings.append(elapsed_ms)
        records.append({
            "elapsed_ms": elapsed_ms,
            "storage_bytes": storage_bytes,
            "dense_bytes": dense_bytes,
            "compression_ratio": ratio,
            "output_tokens": int(output.shape[-1]),
            "first_output_token_ids": output[0, : min(32, output.shape[-1])].tolist(),
        })
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    avg_ms = sum(timings) / len(timings)
    generated_tokens = args.batch_size * (args.max_seq_len - input_ids.shape[-1])
    result = {
        "model": str(model_path),
        "cache": args.cache,
        "batch_size": args.batch_size,
        "prompt_tokens": int(input_ids.shape[-1]),
        "max_seq_len": args.max_seq_len,
        "generated_tokens": generated_tokens,
        "warmup_runs": args.warmup_runs,
        "repeat_runs": args.repeat_runs,
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "average_elapsed_ms": avg_ms,
        "tokens_per_second": generated_tokens / (avg_ms / 1000),
        "peak_memory_bytes": peak_memory,
        "records": records,
        "dart_config": vars(args) if args.cache == "dart" else None,
        "reference_simulation": args.cache == "kitty-reference",
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_path.name}_{args.cache}_b{args.batch_size}_l{args.max_seq_len}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "result_path": str(output_path)}, indent=2))
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
