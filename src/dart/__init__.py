"""DartKV: a small, readable PyTorch KV-cache reference implementation."""

from .cache import DartKVCache, DartKVCacheConfig
from .attention import dense_attention, streamed_dart_attention
from .mixed import MixedQuantizedKey, quantize_key_mixed, select_key_channels
from .quantization import QuantizedTensor, dequantize, quantize, quantize_axis

__all__ = [
    "DartKVCache",
    "DartKVCacheConfig",
    "dense_attention",
    "QuantizedTensor",
    "MixedQuantizedKey",
    "dequantize",
    "quantize",
    "quantize_axis",
    "quantize_key_mixed",
    "select_key_channels",
    "streamed_dart_attention",
]

__version__ = "0.1.0"
