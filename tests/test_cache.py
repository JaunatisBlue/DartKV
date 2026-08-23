import pytest
import torch

from dart import DartKVCache, DartKVCacheConfig


def test_cache_preserves_sink_and_supports_incremental_updates():
    torch.manual_seed(2)
    keys = torch.randn(1, 2, 5, 7, dtype=torch.float32)
    values = torch.randn_like(keys)
    cache = DartKVCache(DartKVCacheConfig(bits=4, group_size=4, sink_tokens=2, metadata_dtype=torch.float32))

    first_keys, first_values = cache.update(keys[..., :1, :], values[..., :1, :])
    assert cache.get_seq_length() == 1
    torch.testing.assert_close(first_keys, keys[..., :1, :])
    torch.testing.assert_close(first_values, values[..., :1, :])

    restored_keys, restored_values = cache.update(keys[..., 1:, :], values[..., 1:, :])
    assert cache.get_seq_length() == 5
    assert restored_keys.shape == keys.shape
    assert restored_values.shape == values.shape
    torch.testing.assert_close(restored_keys[..., :2, :], keys[..., :2, :])
    torch.testing.assert_close(restored_values[..., :2, :], values[..., :2, :])
    assert cache.storage_bytes < cache.dense_bytes
    assert cache.compression_ratio > 1.0


def test_cache_rejects_incompatible_updates_and_can_clear():
    cache = DartKVCache()
    key = torch.zeros(1, 1, 1, 4)
    cache.update(key, key)
    with pytest.raises(ValueError, match="batch/head/head_dim"):
        incompatible = torch.zeros(1, 2, 1, 4)
        cache.update(incompatible, incompatible)
    with pytest.raises(ValueError, match="dtype"):
        cache.update(key.half(), key.half())
    cache.clear()
    assert len(cache) == 0
    with pytest.raises(RuntimeError, match="empty"):
        cache.get()


def test_cache_rejects_mismatched_key_value_dtype():
    cache = DartKVCache()
    keys = torch.zeros(1, 1, 1, 4, dtype=torch.float32)
    values = torch.zeros_like(keys, dtype=torch.float16)
    with pytest.raises(ValueError, match="same dtype and device"):
        cache.update(keys, values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cache_runs_on_gpu():
    key = torch.randn(1, 1, 3, 16, device="cuda", dtype=torch.float16)
    cache = DartKVCache(DartKVCacheConfig(bits=2, group_size=8, metadata_dtype=torch.float16))
    restored, _ = cache.update(key, key)
    assert restored.device.type == "cuda"
    cache.to("cpu")
    restored_cpu, _ = cache.get()
    assert restored_cpu.device.type == "cpu"
