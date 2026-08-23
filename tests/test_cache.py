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


def test_page_metadata_freezes_sequence_order_and_storage_strides():
    torch.manual_seed(51)
    keys = torch.randn(1, 2, 11, 12)
    values = torch.randn_like(keys)
    cache = DartKVCache(DartKVCacheConfig(
        page_size=4,
        hold_partial_pages=True,
        sink_tokens=2,
        key_group_size=4,
        value_group_size=6,
        promote_ratio=0.25,
        metadata_dtype=torch.float32,
    ))
    cache.update(keys, values)
    pages = cache.page_metadata()
    assert [(page.sequence_start, page.token_count) for page in pages] == [(0, 2), (2, 4), (6, 4), (10, 1)]
    assert pages[0].key.kind == pages[0].value.kind == "dense"
    assert pages[1].key.kind == "mixed_key"
    assert pages[1].value.kind == "quantized"
    assert pages[1].key.fields[0][0] == "low_values"
    assert pages[1].key.fields[0][2][-1] == 1
    assert pages[1].value.fields[0][2][-1] == 1
    assert pages[-1].token_count == 1
    assert sum(page.token_count for page in pages) == cache.seen_tokens


def test_page_table_tracks_kitty_style_ids_modes_and_reorder():
    torch.manual_seed(52)
    keys = torch.randn(2, 2, 11, 12)
    values = torch.randn_like(keys)
    cache = DartKVCache(DartKVCacheConfig(
        page_size=4,
        hold_partial_pages=True,
        sink_tokens=2,
        key_group_size=4,
        value_group_size=6,
        promote_ratio=0.25,
        metadata_dtype=torch.float32,
    ))
    cache.update(keys, values)
    table = cache.page_table().validate()
    assert table.page_ids.shape == (2, 4)
    assert table.page_ids.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert table.sequence_ranges().tolist() == [[0, 2], [2, 6], [6, 10], [10, 11]]
    assert table.key_modes.tolist() == [0, 2, 2, 0]
    assert table.value_modes.tolist() == [0, 1, 1, 0]
    assert table.reorder(torch.tensor([1, 1])).page_ids.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert table.to("cpu").device.type == "cpu"


def test_empty_page_table_is_valid_and_append_refreshes_metadata():
    cache = DartKVCache(DartKVCacheConfig(page_size=4))
    empty = cache.page_table().validate()
    assert empty.page_ids.shape == (0, 0)
    key = torch.randn(1, 1, 5, 8)
    cache.update(key, key)
    table = cache.page_table().validate()
    assert table.page_count == 2
    assert table.seen_tokens == 5


def test_page_table_stacks_consecutive_uniform_pages_page_major():
    torch.manual_seed(53)
    keys = torch.randn(1, 2, 13, 16)
    values = torch.randn_like(keys)
    cache = DartKVCache(DartKVCacheConfig(
        page_size=4,
        hold_partial_pages=True,
        sink_tokens=1,
        key_group_size=4,
        value_group_size=8,
        metadata_dtype=torch.float32,
    ))
    cache.update(keys, values)
    table = cache.page_table().validate()
    runs = table.uniform_quantized_runs(cache)
    assert len(runs) == 1
    run = runs[0].validate()
    assert run.first_page_index == 1
    assert run.page_indices.tolist() == [1, 2, 3]
    assert run.sequence_starts.tolist() == [1, 5, 9]
    assert run.token_counts.tolist() == [4, 4, 4]
    assert run.key_values.shape[:4] == (3, 1, 2, 16)
    assert run.value_values.shape[:4] == (3, 1, 2, 4)
    assert run.key_scale.shape[0] == run.value_scale.shape[0] == 3


def test_page_runs_stop_at_mixed_and_pending_tail_pages():
    torch.manual_seed(54)
    keys = torch.randn(1, 1, 10, 8)
    cache = DartKVCache(DartKVCacheConfig(
        page_size=4,
        hold_partial_pages=True,
        promote_ratio=0.25,
        key_group_size=4,
        value_group_size=4,
        metadata_dtype=torch.float32,
    ))
    cache.update(keys, keys)
    table = cache.page_table().validate()
    assert table.key_modes.tolist() == [2, 2, 0]
    assert table.value_modes.tolist() == [1, 1, 0]
    assert table.uniform_quantized_runs(cache) == ()


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
    table = cache.page_table().validate()
    assert table.device.type == "cpu"
    assert table.page_count == 1
