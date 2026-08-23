"""DartKV: a small, readable PyTorch KV-cache reference implementation."""

from .cache import DartKVCache, DartKVCacheConfig
from .mixed import MixedQuantizedKey, quantize_key_mixed, select_key_channels
from .quantization import QuantizedTensor, dequantize, quantize, quantize_axis

__all__ = [
    "DartKVCache",
    "DartKVCacheConfig",
    "QuantizedTensor",
    "MixedQuantizedKey",
    "dequantize",
    "quantize",
    "quantize_axis",
    "quantize_key_mixed",
    "select_key_channels",
]

__version__ = "0.1.0"
