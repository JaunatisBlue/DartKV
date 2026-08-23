import torch

from dart import DartKVCache, DartKVCacheConfig, dense_attention, streamed_dart_attention


def test_streamed_attention_matches_dense_materialized_cache_without_get():
    torch.manual_seed(7)
    keys = torch.randn(1, 2, 19, 8)
    values = torch.randn_like(keys)
    query = torch.randn(1, 4, 3, 8)
    cache = DartKVCache(DartKVCacheConfig(
        page_size=5,
        key_group_size=5,
        value_group_size=4,
        promote_ratio=0.25,
        sink_tokens=1,
        metadata_dtype=torch.float32,
    ))
    cache.update(keys, values)
    dense_keys, dense_values = cache.get()
    expanded_keys = dense_keys.repeat_interleave(2, dim=1)
    expanded_values = dense_values.repeat_interleave(2, dim=1)
    expected = dense_attention(query, expanded_keys, expanded_values)

    # The streaming path must use segment metadata rather than calling get(),
    # which would materialize the complete cache before attention.
    cache.get = lambda: (_ for _ in ()).throw(AssertionError("streaming path materialized the full cache"))
    actual = streamed_dart_attention(query, cache)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_streamed_attention_supports_pending_page():
    torch.manual_seed(8)
    keys = torch.randn(1, 1, 7, 8)
    values = torch.randn_like(keys)
    query = torch.randn(1, 1, 1, 8)
    cache = DartKVCache(DartKVCacheConfig(
        page_size=4,
        hold_partial_pages=True,
        key_group_size=4,
        value_group_size=4,
        metadata_dtype=torch.float32,
    ))
    cache.update(keys, values)
    dense_keys, dense_values = cache.get()
    expected = dense_attention(query, dense_keys, dense_values)
    actual = streamed_dart_attention(query, cache)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
