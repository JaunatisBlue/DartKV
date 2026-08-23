import pytest
import torch

from dart import DartKVCache, DartKVCacheConfig
from dart.integrations import DartHFCache


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


def test_cache_separates_key_value_axes_and_flushes_pages():
    torch.manual_seed(5)
    keys = torch.randn(1, 2, 9, 16)
    values = torch.randn_like(keys)
    cache = DartKVCache(DartKVCacheConfig(
        page_size=4,
        hold_partial_pages=True,
        sink_tokens=1,
        key_group_size=4,
        value_group_size=8,
        promote_ratio=0.25,
        metadata_dtype=torch.float32,
    ))
    restored_keys, restored_values = cache.update(keys, values)
    assert cache.get_seq_length() == 9
    assert cache._pending_key is None
    assert restored_keys.shape == keys.shape
    assert restored_values.shape == values.shape
    cache.update(keys[..., :3, :], values[..., :3, :])
    assert cache._pending_key is not None
    assert cache.get_seq_length() == 12
    assert cache.storage_bytes < cache.dense_bytes


def test_huggingface_cache_adapter_reorders_without_losing_length():
    torch.manual_seed(6)
    config = DartKVCacheConfig(page_size=2, key_group_size=2, value_group_size=4, sink_tokens=0, metadata_dtype=torch.float32)
    cache = DartHFCache(config)
    keys = torch.randn(2, 1, 3, 8)
    values = torch.randn_like(keys)
    cache.update(keys, values, layer_idx=0)
    cache.reorder_cache(torch.tensor([1, 1]))
    assert cache.get_seq_length() == 3
    next_keys = keys[1:2].expand(2, -1, -1, -1)
    next_values = values[1:2].expand(2, -1, -1, -1)
    restored, _ = cache.update(next_keys[:, :, :1, :], next_values[:, :, :1, :], layer_idx=0)
    assert restored.shape == (2, 1, 4, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cache_runs_on_gpu():
    key = torch.randn(1, 1, 3, 16, device="cuda", dtype=torch.float16)
    cache = DartKVCache(DartKVCacheConfig(bits=2, group_size=8, metadata_dtype=torch.float16))
    restored, _ = cache.update(key, key)
    assert restored.device.type == "cuda"
    cache.to("cpu")
    restored_cpu, _ = cache.get()
    assert restored_cpu.device.type == "cpu"
