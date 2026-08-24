import argparse
import json
import sys
from pathlib import Path

import pytest

from examples.check_kitty_reproduction import DEFAULT_MANIFEST, audit
from examples.check_kitty_operator_parity import audit as audit_operator_parity
from examples.reproduce_kitty import (
    _resolved_variant,
    _stop_words,
    _summarize_repeats,
)
from examples.benchmark_kitty import (
    _kitty_prompt,
    _limit_prompt_inputs,
    _new_cache,
    _recursive_tensor_storage,
    build_parser as build_benchmark_parser,
)
from examples.plot_kitty_figure5 import load_points
from dart.mixed import quantize_key_mixed
from dart.mixed import select_key_channels
from dart.quantization import quantize
from dart import DartKVCacheConfig
from dart.integrations import DartHFCache


def test_manifest_averages_match_reported_tables():
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    for table_name in ("table3", "table4"):
        for variants in manifest[table_name].values():
            for targets in variants.values():
                task_means = [value[0] for task, value in targets.items() if task != "average"]
                assert sum(task_means) / len(task_means) == pytest.approx(targets["average"], abs=0.01)


def test_protocol_resolves_paper_and_artifact_kitty_pro_ratios():
    common = {"promote_ratio": None, "channel_selection": None}
    paper = _resolved_variant(argparse.Namespace(protocol="paper", **common), "kitty-pro")
    artifact = _resolved_variant(argparse.Namespace(protocol="artifact", **common), "kitty-pro")
    assert paper["promote_ratio"] == 0.25
    assert artifact["promote_ratio"] == 0.2


def test_stop_words_match_reference_runner_rules():
    qwen = _stop_words(Path("/models/Qwen3-8B"), "gsm8k_cot_llama")
    llama = _stop_words(Path("/models/Llama-3.1-8B"), "humaneval_instruct")
    assert qwen == ["<|endoftext|>", "<|im_end|>", "Given the following problem"]
    assert llama[-1] == "\n```"
    assert "<|eot_id|>" in llama


def test_latency_prompt_is_the_kitty_reference_gsm8k_prompt():
    task_name, prompt = _kitty_prompt(1)
    assert task_name == "gsm8k"
    assert "Janet" in prompt
    assert "The final answer is [answer]" in prompt


def test_latency_benchmark_defaults_to_official_flash_attention_prefill():
    args = build_benchmark_parser().parse_args(["--model", "/models/Qwen3-8B"])
    assert args.attn_implementation == "flash_attention_2"
    assert args.protocol == "paper"
    assert args.paper_prompt_tokens == 100


def test_paper_latency_prompt_keeps_exact_budget_and_chat_suffix():
    import torch
    from transformers.tokenization_utils_base import BatchEncoding

    inputs = BatchEncoding({
        "input_ids": torch.arange(120).reshape(1, 120),
        "attention_mask": torch.ones(1, 120, dtype=torch.long),
    })
    limited = _limit_prompt_inputs(inputs, 100)
    assert limited.input_ids.shape == (1, 100)
    assert limited.input_ids[0, -8:].tolist() == list(range(112, 120))


def test_figure5_hf_baseline_caches_are_available_on_cpu():
    from transformers import Qwen3Config

    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    common = {
        "batch_size": 2,
        "max_seq_len": 16,
        "device": "cpu",
        "dtype": "fp16",
        "quantized_bits": 4,
        "quantized_group_size": 8,
        "quantized_residual_length": 8,
    }
    static = _new_cache(argparse.Namespace(cache="static", **common), config)
    quanto = _new_cache(argparse.Namespace(cache="quanto", **common), config)
    hqq = _new_cache(argparse.Namespace(cache="hqq", **common), config)
    assert type(static).__name__ == "StaticCache"
    assert type(quanto).__name__ == "QuantoQuantizedCache"
    assert type(hqq).__name__ == "HQQQuantizedCache"
    assert _recursive_tensor_storage(static) == 4096


def test_figure5_plot_loader_keeps_both_memory_metrics(tmp_path):
    result = {
        "cache": "static",
        "protocol": "paper",
        "max_seq_len": 8192,
        "batch_size": 32,
        "generated_tokens_per_second": 600.0,
        "sequence_tokens_per_second": 610.0,
        "peak_memory_bytes": 50,
        "peak_memory_reserved_bytes": 60,
    }
    (tmp_path / "point.json").write_text(json.dumps(result))
    points = load_points(tmp_path, "paper", 8192, {"static"})
    assert points[0]["peak_memory_allocated_bytes"] == 50
    assert points[0]["peak_memory_reserved_bytes"] == 60


def test_random_channel_selection_matches_kitty_mask_cpu():
    import torch

    reference_src = Path(__file__).parents[1] / "reference" / "code" / "Kitty" / "src"
    sys.path.insert(0, str(reference_src))
    from kitty_sim.utils_quant import build_promote_mask

    torch.manual_seed(1234)
    states = torch.randn(2, 3, 128, 8, dtype=torch.float16)
    torch.manual_seed(2026)
    reference = build_promote_mask(states.transpose(2, 3), 0.25, 0)
    torch.manual_seed(2026)
    dart, indices = select_key_channels(states, 0.25, strategy="random")
    assert torch.equal(dart, reference)
    assert torch.equal(indices[0], indices[1])


def test_kitty_post_quant_lifecycle_matches_dart_cpu():
    import torch

    reference_src = Path(__file__).parents[1] / "reference" / "code" / "Kitty" / "src"
    sys.path.insert(0, str(reference_src))
    from kitty_sim import KittyKVCache, KittyKVCacheConfig

    reference = KittyKVCache(KittyKVCacheConfig(
        sink_length=2,
        buffer_length=8,
        group_size=8,
        kbits=2,
        vbits=2,
        promote_ratio=0.25,
        promote_bit=4,
        channel_selection=1,
    ))
    from dart import DartKVCacheConfig
    from dart.integrations import DartHFCache

    dart = DartHFCache(DartKVCacheConfig(
        key_bits=2,
        value_bits=2,
        key_group_size=8,
        value_group_size=8,
        sink_tokens=2,
        page_size=8,
        local_tokens=1,
        value_local_tokens=8,
        hold_partial_pages=True,
        promote_bits=4,
        promote_ratio=0.25,
        metadata_dtype=torch.float16,
    ))
    torch.manual_seed(2026)
    for tokens in [13] + [1] * 10:
        keys = torch.randn(1, 2, tokens, 8, dtype=torch.float16)
        values = torch.randn_like(keys)
        ref_keys, ref_values = reference.update(keys, values, 0)
        dart_keys, dart_values = dart.update(keys, values, 0, 0)
        assert torch.equal(dart_keys, ref_keys)
        assert torch.equal(dart_values, ref_values)


def test_repeat_summary_reports_maximum_deviation():
    summary = _summarize_repeats([
        {"task": {"metric": 0.8, "metric_stderr": 0.1}},
        {"task": {"metric": 1.0, "metric_stderr": 0.0}},
    ])
    assert summary["task"]["metric"]["mean"] == pytest.approx(0.9)
    assert summary["task"]["metric"]["max_deviation"] == pytest.approx(0.1)
    assert "metric_stderr" not in summary["task"]


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA is required")
def test_packed_fp16_quantization_matches_kitty_fake_quantization():
    import torch

    reference_src = Path(__file__).parents[1] / "reference" / "code" / "Kitty" / "src"
    sys.path.insert(0, str(reference_src))
    from kitty_sim.utils_quant import build_promote_mask, fake_quant_groupwise_lastdim

    torch.manual_seed(1234)
    states = torch.randn(1, 8, 128, 128, device="cuda", dtype=torch.float16)
    transposed = states.transpose(2, 3).contiguous()
    promote_mask = build_promote_mask(transposed, 0.25, 1)
    reference_keys = fake_quant_groupwise_lastdim(
        transposed, 128, 2, promote_mask, 4
    ).transpose(2, 3).contiguous()
    dart_keys = quantize_key_mixed(
        states,
        group_size=128,
        promote_ratio=0.25,
        promote_bits=4,
        metadata_dtype=torch.float16,
    ).dequantize()
    reference_values = fake_quant_groupwise_lastdim(states, 128, 2)
    dart_values = quantize(
        states, bits=2, group_size=128, metadata_dtype=torch.float16
    ).dequantize()
    assert torch.equal(dart_keys, reference_keys)
    assert torch.equal(dart_values, reference_values)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("prefill_tokens", [13, 18])
def test_dart_hf_lifecycle_matches_kitty_post_quant_simulation(prefill_tokens):
    import torch

    reference_src = Path(__file__).parents[1] / "reference" / "code" / "Kitty" / "src"
    sys.path.insert(0, str(reference_src))
    from kitty_sim import KittyKVCache, KittyKVCacheConfig

    reference = KittyKVCache(KittyKVCacheConfig(
        sink_length=2,
        buffer_length=8,
        group_size=8,
        kbits=2,
        vbits=2,
        promote_ratio=0.25,
        promote_bit=4,
        channel_selection=1,
    ))
    dart = DartHFCache(DartKVCacheConfig(
        key_bits=2,
        value_bits=2,
        key_group_size=8,
        value_group_size=8,
        sink_tokens=2,
        page_size=8,
        local_tokens=1,
        value_local_tokens=8,
        hold_partial_pages=True,
        promote_bits=4,
        promote_ratio=0.25,
        metadata_dtype=torch.float16,
    ))
    torch.manual_seed(20260823)
    for tokens in [prefill_tokens] + [1] * 10:
        keys = torch.randn(1, 2, tokens, 8, device="cuda", dtype=torch.float16)
        values = torch.randn_like(keys)
        reference_keys, reference_values = reference.update(keys, values, 0)
        dart_keys, dart_values = dart.update(keys, values, 0)
        assert torch.equal(dart_keys, reference_keys)
        assert torch.equal(dart_values, reference_values)


def test_audit_reads_summary_and_marks_remaining_cells_missing(tmp_path):
    summary_path = (
        tmp_path
        / "paper"
        / "kitty-reference"
        / "Qwen-8B"
        / "gsm8k_cot_llama"
        / "fp16_summary.json"
    )
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps({
        "repeat_statistics": {
            "gsm8k_cot_llama": {
                "exact_match,strict-match": {"mean": 0.9479}
            }
        }
    }))
    report = audit(argparse.Namespace(
        manifest=DEFAULT_MANIFEST,
        results=tmp_path,
        table="table3",
        model_key="Qwen3-8B",
        model_dir="Qwen-8B",
        protocol="paper",
        backend="kitty-reference",
        absolute_tolerance=None,
    ))
    assert report["counts"]["matched"] == 1
    assert report["counts"]["missing"] == 19
    assert not report["reproduced"]


def test_tiny_qwen3_dart_engine_generation_cpu():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from dart import DartKVCacheConfig
    from dart.integrations import DartHFCache, attach_dart_fused_attention

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        layer_types=["full_attention", "full_attention"],
    )
    model = attach_dart_fused_attention(Qwen3ForCausalLM(config)).eval()
    cache = DartHFCache(DartKVCacheConfig(
        key_bits=2,
        value_bits=2,
        key_group_size=8,
        value_group_size=8,
        sink_tokens=2,
        page_size=8,
        local_tokens=1,
        value_local_tokens=8,
        hold_partial_pages=True,
        promote_bits=4,
        promote_ratio=0.25,
        metadata_dtype=torch.float32,
    ))
    with torch.inference_mode():
        output = model.generate(
            input_ids=torch.randint(0, 128, (1, 13)),
            max_new_tokens=20,
            do_sample=False,
            past_key_values=cache,
            disable_compile=True,
        )
    assert output.shape == (1, 33)
    assert cache.get_seq_length() == 32


def test_direct_kitty_operator_facade_exposes_authoritative_kernels():
    from dart import kitty_kernels

    assert kitty_kernels.KITTY_OPERATOR_SOURCE.endswith("kitty/kvcache")
    assert kitty_kernels.KITTY_OPERATOR_IMPLEMENTATION.startswith("Kitty")
    for name in ("qk_kernel", "sv_kernel", "quantize_pack_k", "quantize_pack_v", "get_kvcache_kitty"):
        assert callable(getattr(kitty_kernels, name))


def test_native_kitty_operator_audit_passes():
    report = audit_operator_parity()
    assert report["passed"]
    assert all(item["same_object"] for item in report["symbols"].values())
    assert all(report["invariants"].values())
