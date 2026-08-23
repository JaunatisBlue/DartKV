"""A minimal device-agnostic compressed KV cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from .mixed import MixedQuantizedKey, quantize_key_mixed
from .quantization import QuantizedTensor, quantize, quantize_axis


Segment = Union[torch.Tensor, QuantizedTensor, MixedQuantizedKey]


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
    key_bits: Optional[int] = None
    value_bits: Optional[int] = None
    key_group_size: Optional[int] = None
    value_group_size: Optional[int] = None
    page_size: int = 128
    hold_partial_pages: bool = False
    promote_bits: int = 4
    promote_ratio: float = 0.0
    channel_selection: str = "magnitude"

    def __post_init__(self) -> None:
        if self.bits not in (2, 4, 8):
            raise ValueError("bits must be one of 2, 4, or 8")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.sink_tokens < 0:
            raise ValueError("sink_tokens must be non-negative")
        for name, value in (("key_bits", self.key_bits), ("value_bits", self.value_bits)):
            if value is not None and value not in (2, 4, 8):
                raise ValueError(f"{name} must be one of 2, 4, or 8")
        for name, value in (("key_group_size", self.key_group_size), ("value_group_size", self.value_group_size)):
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if self.promote_bits != 4:
            raise ValueError("the reference mixed representation currently supports promote_bits=4")
        if not 0.0 <= self.promote_ratio <= 1.0:
            raise ValueError("promote_ratio must be in [0, 1]")
        if self.channel_selection not in {"magnitude", "variance"}:
            raise ValueError("channel_selection must be 'magnitude' or 'variance'")
        if self.metadata_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError("metadata_dtype must be a floating-point torch dtype")

    @property
    def resolved_key_bits(self) -> int:
        return self.key_bits if self.key_bits is not None else self.bits

    @property
    def resolved_value_bits(self) -> int:
        return self.value_bits if self.value_bits is not None else self.bits

    @property
    def resolved_key_group_size(self) -> int:
        return self.key_group_size if self.key_group_size is not None else self.group_size

    @property
    def resolved_value_group_size(self) -> int:
        return self.value_group_size if self.value_group_size is not None else self.group_size


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
        self._pending_key: Optional[torch.Tensor] = None
        self._pending_value: Optional[torch.Tensor] = None
        self._shape: Optional[Tuple[int, int, int]] = None
        self._dtype: Optional[torch.dtype] = None
        self._device: Optional[torch.device] = None
        self._seen_tokens = 0

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    def get_seq_length(self) -> int:
        return self._seen_tokens

    @property
    def shape(self) -> Optional[Tuple[int, int, int, int]]:
        """Logical cache shape ``[batch, kv_heads, sequence, head_dim]``."""

        if self._shape is None:
            return None
        batch, heads, head_dim = self._shape
        return batch, heads, self._seen_tokens, head_dim

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
            self._append_remainder(key_states[..., sink_length:, :], value_states[..., sink_length:, :])
        self._seen_tokens += chunk_length
        return self.get()

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._seen_tokens == 0:
            raise RuntimeError("DartKVCache is empty; call update() before get()")
        key_segments = [*_materialize_segments(self._key_segments), self._pending_key] if self._pending_key is not None else _materialize_segments(self._key_segments)
        value_segments = [*_materialize_segments(self._value_segments), self._pending_value] if self._pending_value is not None else _materialize_segments(self._value_segments)
        keys = torch.cat(key_segments, dim=-2)
        values = torch.cat(value_segments, dim=-2)
        return keys, values

    @property
    def key_states(self) -> torch.Tensor:
        return self.get()[0]

    @property
    def value_states(self) -> torch.Tensor:
        return self.get()[1]

    @property
    def storage_bytes(self) -> int:
        pending = 0
        if self._pending_key is not None:
            pending += self._pending_key.numel() * self._pending_key.element_size()
        if self._pending_value is not None:
            pending += self._pending_value.numel() * self._pending_value.element_size()
        return sum(_segment_nbytes(segment) for segment in (*self._key_segments, *self._value_segments)) + pending

    @property
    def dense_bytes(self) -> int:
        if self._shape is None or self._dtype is None:
            return 0
        batch, heads, head_dim = self._shape
        return 2 * batch * heads * self._seen_tokens * head_dim * torch.tensor([], dtype=self._dtype).element_size()

    @property
    def compression_ratio(self) -> float:
        return self.dense_bytes / self.storage_bytes if self.storage_bytes else 1.0

    def iter_segments(self) -> list[Tuple[Segment, Segment]]:
        """Return key/value pages in sequence order for streaming attention.

        The returned objects are still packed segments (except an optional
        pending partial page), so callers can dequantize one page at a time
        without materializing the complete cache.
        """

        pairs = list(zip(self._key_segments, self._value_segments))
        if self._pending_key is not None:
            pairs.append((self._pending_key, self._pending_value))
        return pairs

    def to(self, device: torch.device | str) -> "DartKVCache":
        """Move all backing tensors to ``device`` and return ``self``."""

        self._key_segments = [_segment_to(segment, device) for segment in self._key_segments]
        self._value_segments = [_segment_to(segment, device) for segment in self._value_segments]
        if self._pending_key is not None:
            self._pending_key = self._pending_key.to(device)
        if self._pending_value is not None:
            self._pending_value = self._pending_value.to(device)
        self._device = torch.device(device)
        return self

    def clear(self) -> None:
        self._key_segments.clear()
        self._value_segments.clear()
        self._pending_key = None
        self._pending_value = None
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

    def _append_remainder(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if self.config.hold_partial_pages:
            if self._pending_key is not None:
                key_states = torch.cat((self._pending_key, key_states), dim=-2)
                value_states = torch.cat((self._pending_value, value_states), dim=-2)
            full_tokens = (key_states.shape[-2] // self.config.page_size) * self.config.page_size
            if full_tokens:
                self._quantize_pages(key_states[..., :full_tokens, :], value_states[..., :full_tokens, :])
            self._pending_key = key_states[..., full_tokens:, :].clone() if full_tokens < key_states.shape[-2] else None
            self._pending_value = value_states[..., full_tokens:, :].clone() if full_tokens < value_states.shape[-2] else None
            return
        for start in range(0, key_states.shape[-2], self.config.page_size):
            stop = min(start + self.config.page_size, key_states.shape[-2])
            self._quantize_pages(key_states[..., start:stop, :], value_states[..., start:stop, :])

    def _quantize_pages(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        key_kwargs = {
            "group_size": self.config.resolved_key_group_size,
            "metadata_dtype": self.config.metadata_dtype,
        }
        if self.config.promote_ratio > 0.0:
            key_segment: Segment = quantize_key_mixed(
                key_states,
                **key_kwargs,
                promote_ratio=self.config.promote_ratio,
                promote_bits=self.config.promote_bits,
                strategy=self.config.channel_selection,
            )
        else:
            key_segment = quantize_axis(
                key_states,
                axis=-2,
                bits=self.config.resolved_key_bits,
                **key_kwargs,
            )
        value_segment = quantize(
            value_states,
            bits=self.config.resolved_value_bits,
            group_size=self.config.resolved_value_group_size,
            metadata_dtype=self.config.metadata_dtype,
        )
        self._key_segments.append(key_segment)
        self._value_segments.append(value_segment)


def _materialize(segment: Segment) -> torch.Tensor:
    return segment if isinstance(segment, torch.Tensor) else segment.dequantize()


def _materialize_segments(segments: List[Segment]) -> List[torch.Tensor]:
    return [_materialize(segment) for segment in segments]


def _segment_nbytes(segment: Segment) -> int:
    return segment.numel() * segment.element_size() if isinstance(segment, torch.Tensor) else segment.nbytes


def _segment_to(segment: Segment, device: torch.device | str) -> Segment:
    return segment.to(device) if isinstance(segment, QuantizedTensor) else segment.to(device)


__all__ = ["DartKVCache", "DartKVCacheConfig"]
