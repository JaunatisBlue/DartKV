import pytest
import torch

from dart import (
    DartKVCache,
    DartKVCacheConfig,
    dense_attention,
    fused_dart_attention,
    streamed_dart_attention,
    triton_available,
)


def test_fused_attention_cpu_falls_back_to_streaming_reference():
    torch.manual_seed(71)
    keys = torch.randn(1, 1, 9, 8)
    values = torch.randn_like(keys)
    query = torch.randn(1, 1, 1, 8)
    cache = DartKVCache(DartKVCacheConfig(page_size=4, hold_partial_pages=True, metadata_dtype=torch.float32))
    cache.update(keys, values)
    actual = fused_dart_attention(query, cache)
    expected = streamed_dart_attention(query, cache)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available(),
    reason="CUDA or Triton is not available",
)
def test_fused_attention_handles_sink_pages_pending_and_gqa_without_get():
    torch.manual_seed(72)
    keys = torch.randn(1, 2, 21, 16, device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    query = torch.randn(1, 4, 1, 16, device="cuda", dtype=torch.float16)
    cache = DartKVCache(DartKVCacheConfig(
        page_size=8,
        sink_tokens=3,
        hold_partial_pages=True,
        key_group_size=8,
        value_group_size=8,
        promote_ratio=0.25,
        metadata_dtype=torch.float16,
    ))
    cache.update(keys, values)
    dense_keys, dense_values = cache.get()
    expected = dense_attention(
        query,
        dense_keys.repeat_interleave(2, dim=1),
        dense_values.repeat_interleave(2, dim=1),
    )
    cache.get = lambda: (_ for _ in ()).throw(AssertionError("fused path materialized the full cache"))
    actual = fused_dart_attention(query, cache, fallback=False)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available(),
    reason="CUDA or Triton is not available",
)
def test_fused_attention_page_run_matches_per_page_kernel():
    torch.manual_seed(73)
    keys = torch.randn(1, 2, 35, 32, device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    query = torch.randn(1, 8, 1, 32, device="cuda", dtype=torch.float16)
    cache = DartKVCache(DartKVCacheConfig(
        page_size=8,
        sink_tokens=3,
        hold_partial_pages=True,
        key_group_size=8,
        value_group_size=8,
        metadata_dtype=torch.float16,
    ))
    cache.update(keys, values)
    table = cache.page_table(device=query.device).validate()
    runs = table.uniform_quantized_runs(cache)
    assert runs and runs[0].page_count >= 3
    page_run = fused_dart_attention(query, cache, page_table=table, page_runs=runs, fallback=False)
    per_page = fused_dart_attention(query, cache, page_table=table, page_runs=(), fallback=False)
    torch.testing.assert_close(page_run, per_page, rtol=2e-3, atol=2e-3)
