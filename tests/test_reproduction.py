import argparse
import json
import sys
from pathlib import Path

import pytest

from examples.check_kitty_reproduction import DEFAULT_MANIFEST, audit
from examples.reproduce_kitty import (
    _resolved_variant,
    _stop_words,
    _summarize_repeats,
)
from examples.benchmark_kitty import _kitty_prompt, build_parser as build_benchmark_parser
from dart.mixed import quantize_key_mixed
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
