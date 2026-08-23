"""Reproducible synthetic attention baseline for the first DartKV milestone."""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from dart import DartKVCache, DartKVCacheConfig


def attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    logits = torch.matmul(query.float(), keys.float().transpose(-1, -2)) / math.sqrt(keys.shape[-1])
    weights = torch.softmax(logits, dim=-1)
    return torch.matmul(weights, values.float())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(fn, device: torch.device, repeats: int) -> float:
    for _ in range(2):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000 / repeats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--bits", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--sink-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260823)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dimensions = (args.batch, args.heads, args.tokens, args.head_dim, args.repeats)
    if min(dimensions) <= 0:
        raise SystemExit("dimensions and repeats must be positive")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    keys = torch.randn(args.batch, args.heads, args.tokens, args.head_dim, device=device, dtype=torch.float16)
    values = torch.randn_like(keys)
    query = torch.randn(args.batch, args.heads, 1, args.head_dim, device=device, dtype=torch.float16)

    cache = DartKVCache(DartKVCacheConfig(bits=args.bits, group_size=args.group_size, sink_tokens=args.sink_tokens))
    start = time.perf_counter()
    cache.update(keys, values)
    _sync(device)
    quantize_ms = (time.perf_counter() - start) * 1000
    restored_keys, restored_values = cache.get()
    dense_output = attention(query, keys, values)
    quantized_output = attention(query, restored_keys, restored_values)
    error = dense_output - quantized_output

    dense_ms = _measure(lambda: attention(query, keys, values), device, args.repeats)
    quantized_ms = _measure(lambda: attention(query, restored_keys, restored_values), device, args.repeats)
    result = {
        "seed": args.seed,
        "device": str(device),
        "shape": [args.batch, args.heads, args.tokens, args.head_dim],
        "bits": args.bits,
        "group_size": args.group_size,
        "sink_tokens": min(args.sink_tokens, args.tokens),
        "max_abs_attention_error": float(error.abs().max().cpu()),
        "rmse_attention_error": float(error.square().mean().sqrt().cpu()),
        "dense_attention_ms": dense_ms,
        "materialized_quantized_attention_ms": quantized_ms,
        "quantize_and_store_ms": quantize_ms,
        "dense_bytes": cache.dense_bytes,
        "storage_bytes": cache.storage_bytes,
        "compression_ratio": cache.compression_ratio,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
