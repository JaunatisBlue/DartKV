"""Audit that DartKV's native Kitty facade uses the authoritative operators."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
KITTY_ROOT = REPO_ROOT / "reference" / "code" / "Kitty" / "src"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments" / "kitty_operator_audit.json",
    )
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_path(value) -> Path:
    inspectable = getattr(value, "fn", value)
    source_file = inspect.getsourcefile(inspectable)
    if source_file is None:
        module = inspect.getmodule(value)
        source_file = getattr(module, "__file__", None)
    if source_file is None:
        raise RuntimeError(f"cannot resolve source for {value!r}")
    return Path(source_file).resolve()


def audit() -> dict:
    from dart import kitty_kernels as facade
    from kitty.kvcache.kernels import kitty_attention, kitty_quant_pack
    from kitty.kvcache import kitty as kitty_cache
    from kitty.kvcache import utils_kv_per_layer
    from kitty.models.qwen3 import Qwen3ForCausalLM_Kitty

    authoritative = {
        "qk_kernel": kitty_attention.qk_kernel,
        "sv_kernel": kitty_attention.sv_kernel,
        "kitty_attention_forward": kitty_attention.kitty_attention_forward,
        "quantize_pack_k": kitty_quant_pack.quantize_pack_k,
        "quantize_pack_v": kitty_quant_pack.quantize_pack_v,
        "KittyCache": kitty_cache.KittyCache,
        "get_kvcache_kitty": kitty_cache.get_kvcache_kitty,
        "KVCache_Layer": utils_kv_per_layer.KVCache_Layer,
        "Qwen3ForCausalLM_Kitty": Qwen3ForCausalLM_Kitty,
    }
    symbols = {}
    for name, reference in authoritative.items():
        exposed = getattr(facade, name)
        source = _source_path(reference)
        symbols[name] = {
            "same_object": exposed is reference,
            "source": str(source.relative_to(REPO_ROOT)),
            "source_under_kitty": source.is_relative_to(KITTY_ROOT),
        }

    source_files = {
        "kitty_attention": KITTY_ROOT / "kitty" / "kvcache" / "kernels" / "kitty_attention.py",
        "kitty_quant_pack": KITTY_ROOT / "kitty" / "kvcache" / "kernels" / "kitty_quant_pack.py",
        "kitty_cache": KITTY_ROOT / "kitty" / "kvcache" / "kitty.py",
        "kitty_layer": KITTY_ROOT / "kitty" / "kvcache" / "utils_kv_per_layer.py",
        "qwen3_model": KITTY_ROOT / "kitty" / "models" / "qwen3" / "modeling_qwen3.py",
    }
    cache_source = source_files["kitty_cache"].read_text()
    model_source = source_files["qwen3_model"].read_text()
    invariants = {
        "page_size_128": "self.page_size = 128" in cache_source,
        "sink_length_32": "self.sink_length = 32" in cache_source,
        "low_bit_2": "self.low_bit = 2" in cache_source,
        "high_bit_4": "self.high_bit = 4" in cache_source,
        "boost_ratio_quarter": "self.d_boosted = self.head_dim // 4" in cache_source,
        "qwen_decode_calls_kitty_attention": "kitty_attention_forward(" in model_source,
        "qwen_prefill_quantizes_after_attention": "past_key_value.quantize_prefill(self.layer_idx)" in model_source,
        "qwen_decode_quantizes_after_attention": "past_key_value.quantize_decode(self.layer_idx)" in model_source,
    }
    passed = (
        all(item["same_object"] and item["source_under_kitty"] for item in symbols.values())
        and all(invariants.values())
    )
    return {
        "scope": "Qwen3-8B Kitty native system operators",
        "facade": "dart.kitty_kernels",
        "passed": passed,
        "symbols": symbols,
        "invariants": invariants,
        "source_sha256": {
            name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)}
            for name, path in source_files.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
