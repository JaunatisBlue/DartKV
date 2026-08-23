"""Command-line smoke test for the DartKV reference implementation."""

from __future__ import annotations

import argparse
import json

import torch

from .cache import DartKVCache, DartKVCacheConfig


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small DartKV quantization smoke test")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--bits", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=2)
    parser.add_argument("--device", default="auto", help="torch device, or auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.batch, args.heads, args.tokens, args.head_dim) <= 0:
        raise SystemExit("dimensions must be positive")
    device = _device(args.device)
    torch.manual_seed(args.seed)
    keys = torch.randn(args.batch, args.heads, args.tokens, args.head_dim, device=device, dtype=torch.float16)
    values = torch.randn_like(keys)
    cache = DartKVCache(DartKVCacheConfig(
        bits=args.bits,
        group_size=args.group_size,
        sink_tokens=args.sink_tokens,
    ))
    restored_keys, restored_values = cache.update(keys, values)
    result = {
        "device": str(device),
        "shape": list(keys.shape),
        "bits": args.bits,
        "group_size": args.group_size,
        "sink_tokens": min(args.sink_tokens, args.tokens),
        "max_abs_error_key": float((keys - restored_keys).abs().max().cpu()),
        "max_abs_error_value": float((values - restored_values).abs().max().cpu()),
        "dense_bytes": cache.dense_bytes,
        "storage_bytes": cache.storage_bytes,
        "compression_ratio": cache.compression_ratio,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
