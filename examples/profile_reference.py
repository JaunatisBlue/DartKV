"""Profile dense versus page-streamed Dart attention on synthetic tensors."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from dart import (
    DartKVCache,
    DartKVCacheConfig,
    dense_attention,
    fused_dart_attention,
    streamed_dart_attention,
    triton_available,
    triton_dequantize,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(fn, device: torch.device, repeats: int) -> tuple[float, int | None]:
    for _ in range(2):
        fn()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync(device)
    elapsed = (time.perf_counter() - start) * 1000 / repeats
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return elapsed, peak


def _dequantize_segments(cache: DartKVCache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Dequantize pages without concatenating them into one dense cache."""

    segments = []
    for key_segment, value_segment in cache.iter_segments():
        keys = key_segment if isinstance(key_segment, torch.Tensor) else triton_dequantize(key_segment)
        values = value_segment if isinstance(value_segment, torch.Tensor) else triton_dequantize(value_segment)
        segments.append((keys, values))
    return segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--query-tokens", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--key-group-size", type=int, default=128)
    parser.add_argument("--value-group-size", type=int, default=64)
    parser.add_argument("--sink-tokens", type=int, default=32)
    parser.add_argument("--promote-ratio", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--trace", action="store_true", help="write a PyTorch profiler Chrome trace")
    parser.add_argument("--output", default="results/profile")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dimensions = (args.batch, args.kv_heads, args.query_heads, args.tokens, args.query_tokens, args.head_dim, args.page_size, args.repeats)
    if min(dimensions) <= 0:
        raise SystemExit("all dimensions, page-size, and repeats must be positive")
    if args.query_heads % args.kv_heads:
        raise SystemExit("query-heads must be divisible by kv-heads")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    keys = torch.randn(args.batch, args.kv_heads, args.tokens, args.head_dim, device=device, dtype=torch.float16)
    values = torch.randn_like(keys)
    query = torch.randn(args.batch, args.query_heads, args.query_tokens, args.head_dim, device=device, dtype=torch.float16)
    cache = DartKVCache(DartKVCacheConfig(
        page_size=args.page_size,
        key_group_size=args.key_group_size,
        value_group_size=args.value_group_size,
        sink_tokens=args.sink_tokens,
        promote_ratio=args.promote_ratio,
        hold_partial_pages=True,
        metadata_dtype=torch.float16,
    ))

    _sync(device)
    quantize_start = time.perf_counter()
    cache.update(keys, values)
    _sync(device)
    quantize_ms = (time.perf_counter() - quantize_start) * 1000
    page_table_start = time.perf_counter()
    page_table = cache.page_table(device=device).validate()
    _sync(device)
    page_table_ms = (time.perf_counter() - page_table_start) * 1000
    # Keep the dense tensors for the numerical oracle, then measure the two
    # materialization stages separately. ``segment_dequantize`` includes the
    # packed unpack operation; ``materialize`` additionally concatenates pages.
    dense_keys, dense_values = cache.get()
    expanded_keys = dense_keys.repeat_interleave(args.query_heads // args.kv_heads, dim=1)
    expanded_values = dense_values.repeat_interleave(args.query_heads // args.kv_heads, dim=1)
    expected = dense_attention(query, expanded_keys, expanded_values)
    original_expected = dense_attention(
        query,
        keys.repeat_interleave(args.query_heads // args.kv_heads, dim=1),
        values.repeat_interleave(args.query_heads // args.kv_heads, dim=1),
    )
    streamed = streamed_dart_attention(query, cache)
    fused = fused_dart_attention(query, cache, page_table=page_table)
    stream_error = streamed.float() - expected.float()
    fused_error = fused.float() - expected.float()
    quant_error = expected.float() - original_expected.float()
    segment_dequantize_ms, segment_dequantize_peak = _measure(
        lambda: _dequantize_segments(cache), device, args.repeats
    )
    materialize_ms, materialize_peak = _measure(lambda: cache.get(), device, args.repeats)
    dense_ms, dense_peak = _measure(lambda: dense_attention(query, expanded_keys, expanded_values), device, args.repeats)
    streamed_ms, streamed_peak = _measure(lambda: streamed_dart_attention(query, cache), device, args.repeats)
    fused_ms, fused_peak = _measure(
        lambda: fused_dart_attention(query, cache, page_table=page_table), device, args.repeats
    )

    trace_path = None
    fused_trace_path = None
    if args.trace:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / f"dart_attention_{args.tokens}_{args.promote_ratio:g}.json"
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            streamed_dart_attention(query, cache)
        profiler.export_chrome_trace(str(trace_path))
        fused_trace_path = output_dir / f"dart_fused_attention_{args.tokens}_{args.promote_ratio:g}.json"
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            fused_dart_attention(query, cache, page_table=page_table)
        profiler.export_chrome_trace(str(fused_trace_path))

    result = {
        "device": str(device),
        "shape": [args.batch, args.kv_heads, args.tokens, args.head_dim],
        "query_shape": [args.batch, args.query_heads, args.query_tokens, args.head_dim],
        "page_size": args.page_size,
        "key_group_size": args.key_group_size,
        "value_group_size": args.value_group_size,
        "sink_tokens": min(args.sink_tokens, args.tokens),
        "promote_ratio": args.promote_ratio,
        "dequant_backend": "triton" if device.type == "cuda" and triton_available() else "pytorch",
        "seed": args.seed,
        "quantize_and_store_ms": quantize_ms,
        "page_table_build_ms": page_table_ms,
        "segment_dequantize_ms": segment_dequantize_ms,
        "materialize_ms": materialize_ms,
        "dense_attention_ms": dense_ms,
        "streamed_attention_ms": streamed_ms,
        "fused_page_attention_ms": fused_ms,
        "fused_page_attention_backend": "triton" if device.type == "cuda" and triton_available() else "pytorch-fallback",
        "segment_dequantize_peak_bytes": segment_dequantize_peak,
        "materialize_peak_bytes": materialize_peak,
        "dense_peak_bytes": dense_peak,
        "streamed_peak_bytes": streamed_peak,
        "fused_page_attention_peak_bytes": fused_peak,
        "max_abs_stream_error": float(stream_error.abs().max().cpu()),
        "rmse_stream_error": float(stream_error.square().mean().sqrt().cpu()),
        "max_abs_fused_error": float(fused_error.abs().max().cpu()),
        "rmse_fused_error": float(fused_error.square().mean().sqrt().cpu()),
        "max_abs_quantization_error": float(quant_error.abs().max().cpu()),
        "rmse_quantization_error": float(quant_error.square().mean().sqrt().cpu()),
        "cache_storage_bytes": cache.storage_bytes,
        "dense_cache_bytes": cache.dense_bytes,
        "cache_compression_ratio": cache.compression_ratio,
        "trace_path": str(trace_path) if trace_path else None,
        "fused_trace_path": str(fused_trace_path) if fused_trace_path else None,
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"reference_{args.tokens}_{args.promote_ratio:g}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "result_path": str(result_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
