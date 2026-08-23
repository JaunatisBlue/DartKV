"""A minimal device-agnostic compressed KV cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from .quantization import QuantizedTensor, quantize


Segment = Union[torch.Tensor, QuantizedTensor]


@dataclass(frozen=True)
class DartKVCacheConfig:
    """Configuration for the reference cache.

    ``sink_tokens`` keeps the first tokens in the original dtype, following
    the sink/local-buffer idea used by several KV-cache papers. Remaining
    segments use uniform group-wise affine quantization.
    """

    bits: int = 2
    group_size: int = 64
    sink_tokens: int = 0
    metadata_dtype: torch.dtype = torch.float16

    def __post_init__(self) -> None:
        if self.bits not in (2, 4, 8):
            raise ValueError("bits must be one of 2, 4, or 8")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.sink_tokens < 0:
            raise ValueError("sink_tokens must be non-negative")
        if self.metadata_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError("metadata_dtype must be a floating-point torch dtype")


class DartKVCache:
    """Append-only KV cache with a readable PyTorch quantized backing store.

    Inputs and returned tensors use the Hugging Face-compatible layout
    ``[batch, kv_heads, sequence, head_dim]``. The cache owns detached copies
    of appended tensors and therefore is intended for inference, not training.
    """

    def __init__(self, config: Optional[DartKVCacheConfig] = None) -> None:
        self.config = config or DartKVCacheConfig()
        self._key_segments: List[Segment] = []
        self._value_segments: List[Segment] = []
        self._shape: Optional[Tuple[int, int, int]] = None
        self._dtype: Optional[torch.dtype] = None
        self._device: Optional[torch.device] = None
        self._seen_tokens = 0

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    def get_seq_length(self) -> int:
        return self._seen_tokens

    def __len__(self) -> int:
        return self._seen_tokens

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append a chunk and return the materialized cache contents."""

        self._validate_inputs(key_states, value_states)
        key_states = key_states.detach().contiguous()
        value_states = value_states.detach().contiguous()
        chunk_length = key_states.shape[-2]
        sink_remaining = max(0, self.config.sink_tokens - self._seen_tokens)
        sink_length = min(sink_remaining, chunk_length)

        if sink_length:
            self._key_segments.append(key_states[..., :sink_length, :].clone())
            self._value_segments.append(value_states[..., :sink_length, :].clone())
        if sink_length < chunk_length:
            key_quantized = quantize(
                key_states[..., sink_length:, :],
                bits=self.config.bits,
                group_size=self.config.group_size,
                metadata_dtype=self.config.metadata_dtype,
            )
            value_quantized = quantize(
                value_states[..., sink_length:, :],
                bits=self.config.bits,
                group_size=self.config.group_size,
                metadata_dtype=self.config.metadata_dtype,
            )
            self._key_segments.append(key_quantized)
            self._value_segments.append(value_quantized)
        self._seen_tokens += chunk_length
        return self.get()

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._key_segments:
            raise RuntimeError("DartKVCache is empty; call update() before get()")
        keys = torch.cat([_materialize(segment) for segment in self._key_segments], dim=-2)
        values = torch.cat([_materialize(segment) for segment in self._value_segments], dim=-2)
        return keys, values

    @property
    def key_states(self) -> torch.Tensor:
        return self.get()[0]

    @property
    def value_states(self) -> torch.Tensor:
        return self.get()[1]

    @property
    def storage_bytes(self) -> int:
        return sum(_segment_nbytes(segment) for segment in (*self._key_segments, *self._value_segments))

    @property
    def dense_bytes(self) -> int:
        if self._shape is None or self._dtype is None:
            return 0
        batch, heads, head_dim = self._shape
        return 2 * batch * heads * self._seen_tokens * head_dim * torch.tensor([], dtype=self._dtype).element_size()

    @property
    def compression_ratio(self) -> float:
        return self.dense_bytes / self.storage_bytes if self.storage_bytes else 1.0

    def to(self, device: torch.device | str) -> "DartKVCache":
        """Move all backing tensors to ``device`` and return ``self``."""

        self._key_segments = [_segment_to(segment, device) for segment in self._key_segments]
        self._value_segments = [_segment_to(segment, device) for segment in self._value_segments]
        self._device = torch.device(device)
        return self

    def clear(self) -> None:
        self._key_segments.clear()
        self._value_segments.clear()
        self._shape = None
        self._dtype = None
        self._device = None
        self._seen_tokens = 0

    def _validate_inputs(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError("key_states and value_states must have shape [B, H, T, D]")
        if key_states.shape != value_states.shape:
            raise ValueError(f"key/value shapes must match, got {key_states.shape} and {value_states.shape}")
        if key_states.dtype != value_states.dtype or key_states.device != value_states.device:
            raise ValueError("key_states and value_states must use the same dtype and device")
        if key_states.shape[-2] == 0 or key_states.shape[-1] == 0:
            raise ValueError("key/value sequence and head dimensions must be non-empty")
        if not torch.is_floating_point(key_states) or not torch.is_floating_point(value_states):
            raise TypeError("key_states and value_states must be floating-point tensors")
        shape = (key_states.shape[0], key_states.shape[1], key_states.shape[3])
        if self._shape is None:
            self._shape = shape
            self._dtype = key_states.dtype
            self._device = key_states.device
        elif shape != self._shape or key_states.dtype != self._dtype or key_states.device != self._device:
            raise ValueError("all updates must keep batch/head/head_dim, dtype, and device unchanged")


def _materialize(segment: Segment) -> torch.Tensor:
    return segment if isinstance(segment, torch.Tensor) else segment.dequantize()


def _segment_nbytes(segment: Segment) -> int:
    return segment.numel() * segment.element_size() if isinstance(segment, torch.Tensor) else segment.nbytes


def _segment_to(segment: Segment, device: torch.device | str) -> Segment:
    return segment.to(device) if isinstance(segment, QuantizedTensor) else segment.to(device)


__all__ = ["DartKVCache", "DartKVCacheConfig"]
