"""Direct Kitty operator facade used by the Qwen3-8B system path.

Kitty's MIT-licensed Triton operators remain the authoritative implementation
for the system reproduction.  This module gives DartKV one stable import point
for the exact page allocation, quantize-pack, QK and SV operators instead of
reimplementing them with a semantically similar generic kernel.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_kitty_source() -> None:
    try:
        import kitty  # noqa: F401

        return
    except ImportError:
        source = Path(__file__).resolve().parents[2] / "reference" / "code" / "Kitty" / "src"
        sys.path.insert(0, str(source))


_ensure_kitty_source()

from kitty.kvcache.kernels.kitty_attention import (  # noqa: E402
    kitty_attention_forward,
    qk_kernel,
    sv_kernel,
)
from kitty.kvcache.kernels.kitty_quant_pack import (  # noqa: E402
    quantize_pack_k,
    quantize_pack_v,
)
from kitty.kvcache.kitty import KittyCache, get_kvcache_kitty  # noqa: E402
from kitty.kvcache.utils_kv_per_layer import KVCache_Layer  # noqa: E402
from kitty.models.qwen3 import Qwen3ForCausalLM_Kitty  # noqa: E402


KITTY_OPERATOR_SOURCE = "reference/code/Kitty/src/kitty/kvcache"
KITTY_OPERATOR_IMPLEMENTATION = "Kitty MIT Triton operators"

__all__ = [
    "KITTY_OPERATOR_SOURCE",
    "KITTY_OPERATOR_IMPLEMENTATION",
    "KVCache_Layer",
    "KittyCache",
    "Qwen3ForCausalLM_Kitty",
    "get_kvcache_kitty",
    "kitty_attention_forward",
    "qk_kernel",
    "sv_kernel",
    "quantize_pack_k",
    "quantize_pack_v",
]
