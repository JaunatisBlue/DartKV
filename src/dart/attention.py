"""Reference attention paths for dense and page-streamed Dart caches."""

from __future__ import annotations

import math
from typing import Optional

import torch

from .cache import DartKVCache


def dense_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute eager attention for ``[B, H, Q/T, D]`` tensors."""

    if query.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        raise ValueError("query, keys, and values must have shape [B, H, T, D]")
    if keys.shape != values.shape or query.shape[0] != keys.shape[0] or query.shape[-1] != keys.shape[-1]:
        raise ValueError("query/key/value batch, head_dim, and key/value shapes must be compatible")
    factor = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    logits = torch.matmul(query.float(), keys.float().transpose(-1, -2)) * factor
    return torch.matmul(torch.softmax(logits, dim=-1), values.float()).to(query.dtype)


def streamed_dart_attention(
    query: torch.Tensor,
    cache: DartKVCache,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Attend to packed Dart pages with an online softmax accumulator.

    Pages are dequantized and discarded one at a time. This is intentionally a
    PyTorch reference implementation: it demonstrates the memory lifecycle
    required by a fused kernel while keeping the result easy to compare with
    ``dense_attention``. Query heads may use GQA and are mapped to KV heads by
    contiguous head groups.
    """

    if query.ndim != 4:
        raise ValueError("query must have shape [B, Hq, Q, D]")
    if cache.get_seq_length() == 0:
        raise ValueError("cache is empty")
    cache_shape = cache.shape
    if cache_shape is None:
        raise ValueError("cache is empty")
    batch, kv_heads, _, head_dim = cache_shape
    if query.shape[0] != batch or query.shape[-1] != head_dim:
        raise ValueError("query batch and head_dim must match the cache")
    query_heads = query.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by cache KV heads")
    groups = query_heads // kv_heads
    factor = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    query_float = query.float()
    output = torch.zeros_like(query_float)

    # Online softmax: maintain m=max(logits), l=sum(exp(logits-m)), and the
    # weighted value accumulator for every query head and query position.
    for query_head in range(query_heads):
        kv_head = query_head // groups
        q = query_float[:, query_head]
        running_max = torch.full((batch, query.shape[-2]), -torch.inf, dtype=torch.float32, device=query.device)
        running_sum = torch.zeros_like(running_max)
        running_value = torch.zeros((batch, query.shape[-2], head_dim), dtype=torch.float32, device=query.device)
        for key_segment, value_segment in cache.iter_segments():
            keys = key_segment if isinstance(key_segment, torch.Tensor) else key_segment.dequantize()
            values = value_segment if isinstance(value_segment, torch.Tensor) else value_segment.dequantize()
            keys = keys[:, kv_head].float()
            values = values[:, kv_head].float()
            logits = torch.matmul(q, keys.transpose(-1, -2)) * factor
            page_max = logits.amax(dim=-1)
            new_max = torch.maximum(running_max, page_max)
            old_weight = torch.exp(running_max - new_max)
            page_weight = torch.exp(logits - new_max.unsqueeze(-1))
            running_value = running_value * old_weight.unsqueeze(-1) + torch.matmul(page_weight, values)
            running_sum = running_sum * old_weight + page_weight.sum(dim=-1)
            running_max = new_max
        output[:, query_head] = running_value / running_sum.clamp_min(1e-20).unsqueeze(-1)
    return output.to(query.dtype)


__all__ = ["dense_attention", "streamed_dart_attention"]
