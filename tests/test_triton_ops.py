import pytest
import torch

from dart import DartKVCache, DartKVCacheConfig, triton_available, triton_dequantize


def _cache(device: str, promote_ratio: float) -> DartKVCache:
    torch.manual_seed(61)
    keys = torch.randn(1, 2, 17, 12, device=device, dtype=torch.float16)
    values = torch.randn_like(keys)
    cache = DartKVCache(DartKVCacheConfig(
        page_size=8,
        key_group_size=8,
        value_group_size=6,
        promote_ratio=promote_ratio,
        metadata_dtype=torch.float16,
    ))
    cache.update(keys, values)
    return cache


def test_triton_dequantize_cpu_falls_back_to_reference():
    assert triton_available()
    for promote_ratio in (0.0, 0.25):
        cache = _cache("cpu", promote_ratio)
        for key_segment, value_segment in cache.iter_segments():
            torch.testing.assert_close(triton_dequantize(key_segment), key_segment.dequantize())
            torch.testing.assert_close(triton_dequantize(value_segment), value_segment.dequantize())


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available(),
    reason="CUDA or Triton is not available",
)
def test_triton_dequantize_matches_pytorch_for_uniform_and_mixed_pages():
    for promote_ratio in (0.0, 0.25):
        cache = _cache("cuda", promote_ratio)
        for key_segment, value_segment in cache.iter_segments():
            key_actual = triton_dequantize(key_segment)
            value_actual = triton_dequantize(value_segment)
            # Triton computes the affine expression in fp32 before storing to
            # fp16; the reference may round the multiply in metadata dtype.
            torch.testing.assert_close(key_actual, key_segment.dequantize(), rtol=2e-3, atol=4e-3)
            torch.testing.assert_close(value_actual, value_segment.dequantize(), rtol=2e-3, atol=4e-3)
