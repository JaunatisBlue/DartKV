"""Run a deterministic local Qwen3 dense or Dart cache baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from dart.cache import DartKVCacheConfig
from dart.integrations import DartHFCache


DEFAULT_MODEL = "/opt/model/Qwen/Qwen-8B"
DEFAULT_PROMPT = "Explain why keeping a key/value cache reduces autoregressive decoding cost."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local Hugging Face model directory")
    parser.add_argument("--cache", choices=("dense", "dart"), default="dense")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--bits", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--key-bits", type=int, choices=(2, 4, 8), default=None)
    parser.add_argument("--value-bits", type=int, choices=(2, 4, 8), default=None)
    parser.add_argument("--key-group-size", type=int, default=128)
    parser.add_argument("--value-group-size", type=int, default=64)
    parser.add_argument("--promote-ratio", type=float, default=0.25)
    parser.add_argument("--promote-bits", type=int, choices=(4,), default=4)
    parser.add_argument("--channel-selection", choices=("magnitude", "variance"), default="magnitude")
    parser.add_argument("--sink-tokens", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--hold-partial-pages", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--output", default="results/qwen3")
    return parser


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dense_storage(cache) -> int:
    key_cache = getattr(cache, "key_cache", [])
    value_cache = getattr(cache, "value_cache", [])
    return sum(t.numel() * t.element_size() for t in (*key_cache, *value_cache) if isinstance(t, torch.Tensor) and t.numel())


def _make_cache(args: argparse.Namespace):
    if args.cache == "dense":
        from transformers import DynamicCache

        return DynamicCache()
    return DartHFCache(
        DartKVCacheConfig(
            bits=args.bits,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            key_group_size=args.key_group_size,
            value_group_size=args.value_group_size,
            sink_tokens=args.sink_tokens,
            page_size=args.page_size,
            hold_partial_pages=args.hold_partial_pages,
            promote_bits=args.promote_bits,
            promote_ratio=args.promote_ratio,
            channel_selection=args.channel_selection,
            metadata_dtype=torch.float16,
        )
    )


def run(args: argparse.Namespace) -> dict:
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    model_path = Path(args.model).expanduser()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"no local config.json found under {model_path}")
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
    ).to(device).eval()
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    cache = _make_cache(args)

    with torch.inference_mode():
        _sync(device)
        prefill_start = time.perf_counter()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        _sync(device)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        generated = [next_token]

        decode_start = time.perf_counter()
        for _ in range(args.max_new_tokens - 1):
            position = torch.arange(cache.get_seq_length(), cache.get_seq_length() + 1, device=device)
            outputs = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                cache_position=position,
                return_dict=True,
            )
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
            generated.append(next_token)
        _sync(device)
        decode_ms = (time.perf_counter() - decode_start) * 1000

    generated_ids = torch.cat(generated, dim=-1)
    text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    generated_only = text
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device)
    else:
        peak_memory = None
    if args.cache == "dart":
        storage_bytes = cache.storage_bytes
        dense_bytes = cache.dense_bytes
        compression_ratio = cache.compression_ratio
    else:
        storage_bytes = _dense_storage(cache)
        dense_bytes = storage_bytes
        compression_ratio = 1.0
    return {
        "model": str(model_path),
        "transformers": transformers_version,
        "cache": args.cache,
        "prompt": args.prompt,
        "generated_text": text,
        "generated_tokens": generated_only,
        "input_tokens": int(input_ids.shape[-1]),
        "new_tokens": int(generated_ids.shape[-1]),
        "dtype": args.dtype,
        "device": str(device),
        "attention_backend": args.attn_implementation,
        "seed": args.seed,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "decode_ms_per_token": decode_ms / max(1, args.max_new_tokens - 1),
        "peak_memory_bytes": peak_memory,
        "cache_storage_bytes": storage_bytes,
        "dense_cache_bytes": dense_bytes,
        "cache_compression_ratio": compression_ratio,
        "dart_config": dict(vars(args)) if args.cache == "dart" else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache == "dart":
        variant = f"dart_{'mixed' if args.promote_ratio > 0 else 'uniform'}_p{args.promote_ratio:g}"
    else:
        variant = "dense"
    filename = f"{Path(args.model).name}_{variant}_{args.seed}.json"
    output_path = output_dir / filename
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({**result, "result_path": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
