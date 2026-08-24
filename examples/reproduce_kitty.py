"""Reproduce Kitty-style KV-cache accuracy runs with the local lm-eval harness.

The paper's accuracy simulation compares FP16, KIVI-2, KIVI*-2, Kitty, and
Kitty-Pro under the same prompt/task seeds.  This runner keeps that matrix in a
single explicit command while using the local Dart cache implementation.  It
is intentionally resumable at the process level: run one variant at a time so
model memory is released between configurations.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch


VARIANTS = {
    "fp16": {
        "label": "K16V16",
        "cache": "dense",
    },
    "kivi-2": {
        "label": "KIVI-K2V2",
        "cache": "dart",
        "sink_tokens": 0,
        "promote_ratio": 0.0,
        "channel_selection": "random",
    },
    "kivi-2-sink": {
        "label": "KIVI-K2V2*",
        "cache": "dart",
        "sink_tokens": 32,
        "promote_ratio": 0.0,
        "channel_selection": "random",
    },
    "kitty": {
        "label": "Kitty",
        "cache": "dart",
        "sink_tokens": 32,
        "promote_ratio": 0.125,
        "channel_selection": "magnitude",
    },
    "kitty-pro": {
        "label": "Kitty-Pro",
        "cache": "dart",
        "sink_tokens": 32,
        "promote_ratio": 0.25,
        "channel_selection": "magnitude",
    },
}

PROTOCOLS = {
    "paper": {
        "description": "Section 5.1 sampling protocol and Table 3 promotion ratios",
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "kitty_pro_ratio": 0.25,
    },
    "artifact": {
        "description": "checked-in accuracy_eval.sh plus task-YAML generation settings",
        "do_sample": None,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "kitty_pro_ratio": 0.2,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--task", required=True, help="lm-eval task, e.g. gsm8k_cot_llama or aime24")
    parser.add_argument("--variant", choices=tuple(VARIANTS) + ("all",), default="kitty-pro")
    parser.add_argument("--backend", choices=("kitty-reference", "dart"), default="kitty-reference")
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), default="paper")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--buffer-length", type=int, default=128)
    parser.add_argument("--sink-length", type=int, default=32)
    parser.add_argument("--kbits", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--vbits", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--promote-bit", type=int, choices=(4,), default=4)
    parser.add_argument("--promote-ratio", type=float, default=None, help="Override the selected variant ratio")
    parser.add_argument(
        "--channel-selection",
        choices=("magnitude", "random", "variance"),
        default=None,
        help="Override the selected variant heuristic",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Limit task samples for a smoke run")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--numpy-seed", type=int, default=1234)
    parser.add_argument("--torch-seed", type=int, default=1234)
    parser.add_argument("--fewshot-seed", type=int, default=1234)
    parser.add_argument("--output", default="results/kitty_repro")
    parser.add_argument(
        "--gpqa-data",
        type=Path,
        default=Path(os.environ.get(
            "DARTKV_GPQA_DATA",
            "/home/yx/.cache/dartkv/gpqa/dataset/gpqa_diamond.csv",
        )),
        help="Official local GPQA-Diamond CSV prepared by examples/prepare_gpqa.py",
    )
    parser.add_argument("--confirm-run-unsafe-code", action="store_true")
    parser.add_argument(
        "--request-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Atomically checkpoint stochastic response batches and RNG states",
    )
    return parser


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items() if key != "gen_kwargs"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resolved_variant(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    config = dict(VARIANTS[variant])
    if variant == "kitty-pro":
        config["promote_ratio"] = PROTOCOLS[args.protocol]["kitty_pro_ratio"]
    if args.promote_ratio is not None:
        config["promote_ratio"] = args.promote_ratio
    if args.channel_selection is not None:
        config["channel_selection"] = args.channel_selection
    return config


def _run_name(args: argparse.Namespace, variant: str, config: dict[str, Any]) -> str:
    if args.promote_ratio is None and args.channel_selection is None:
        return variant
    ratio = str(config["promote_ratio"]).replace(".", "_")
    return f"{variant}__{config['channel_selection']}__pr{ratio}"


def _cache_for_variant(args: argparse.Namespace, variant: str):
    config = _resolved_variant(args, variant)
    if config["cache"] == "dense":
        from transformers import DynamicCache

        return DynamicCache(), None, config
    if args.backend == "kitty-reference":
        if config["channel_selection"] == "variance":
            raise ValueError("the Kitty reference backend does not implement variance selection")
        reference_src = Path(__file__).resolve().parents[1] / "reference" / "code" / "Kitty" / "src"
        if str(reference_src) not in sys.path:
            sys.path.insert(0, str(reference_src))
        from kitty_sim import KittyKVCache, KittyKVCacheConfig

        cache_config = KittyKVCacheConfig(
            sink_length=int(config["sink_tokens"]),
            buffer_length=args.buffer_length,
            group_size=args.group_size,
            kbits=args.kbits,
            vbits=args.vbits,
            promote_ratio=float(config["promote_ratio"]),
            promote_bit=args.promote_bit,
            channel_selection=1 if config["channel_selection"] == "magnitude" else 0,
        )
        return KittyKVCache(cache_config), cache_config, config
    dart_selection = config["channel_selection"]
    from dart import DartKVCacheConfig
    from dart.integrations import DartHFCache

    cache_config = DartKVCacheConfig(
        key_bits=args.kbits,
        value_bits=args.vbits,
        key_group_size=args.group_size,
        value_group_size=args.group_size,
        sink_tokens=int(config["sink_tokens"]),
        page_size=args.buffer_length,
        # Kitty simulation flushes a full key Q-buffer only after the next
        # decode token arrives, so the current PostQuant snapshot stays dense.
        local_tokens=1,
        value_local_tokens=args.buffer_length,
        hold_partial_pages=True,
        promote_bits=args.promote_bit,
        promote_ratio=float(config["promote_ratio"]),
        channel_selection=dart_selection,
        metadata_dtype=torch.float16,
    )
    return DartHFCache(cache_config), cache_config, config


def _task_fewshot(task: str) -> int:
    task_lower = task.lower()
    for key, value in (("mmlu", 4), ("gsm8k", 8), ("gpqa", 5), ("math", 4)):
        if key in task_lower:
            return value
    return 0


def _stop_words(model_path: Path, task: str) -> list[str]:
    """Mirror the checked-in Kitty runner's model/task-specific terminators."""

    model_name = str(model_path).lower()
    if "qwen" in model_name:
        stop_words = ["<|endoftext|>", "<|im_end|>"]
    elif "llama" in model_name:
        stop_words = ["<|end_of_text|>", "<|eot_id|>", "<|end_header_id|>"]
    else:
        stop_words = [
            "<|end_of_text|>",
            "<|eot_id|>",
            "<|end_header_id|>",
            "<|endoftext|>",
            "<|im_end|>",
        ]
    task_lower = task.lower()
    for key, values in (
        ("gsm8k", ["Given the following problem"]),
        ("math", ["Problem:"]),
        ("gpqa", ["Question:"]),
        ("humaneval", ["\n```"]),
        ("aime", ["Given the following problem"]),
    ):
        if key in task_lower:
            stop_words.extend(values)
            break
    return stop_words


def _generation_kwargs(args: argparse.Namespace, cache: Any, model_path: Path) -> dict[str, Any]:
    protocol = PROTOCOLS[args.protocol]
    gen_kwargs: dict[str, Any] = {
        "past_key_values": cache,
        # Standard lm-eval 0.4.12 translates this to transformers.max_new_tokens.
        # The Kitty branch passes max_new_tokens plus max_length=None directly,
        # which raises in the upstream harness/Transformers versions used here.
        "max_gen_toks": args.max_new_tokens,
        "until": _stop_words(model_path, args.task),
    }
    if protocol["do_sample"] is not None:
        gen_kwargs.update({
            "do_sample": protocol["do_sample"],
            "temperature": protocol["temperature"],
            "top_p": protocol["top_p"],
            "top_k": protocol["top_k"],
        })
    return gen_kwargs


def _experiment_signature(
    args: argparse.Namespace,
    variant: str,
    resolved_variant: dict[str, Any],
) -> dict[str, Any]:
    """Fields that must agree before an existing repeat can be resumed."""

    signature = {
        "model": str(Path(args.model).expanduser()),
        "task": args.task,
        "variant": variant,
        "resolved_variant": resolved_variant,
        "backend": args.backend,
        "protocol": args.protocol,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "group_size": args.group_size,
        "buffer_length": args.buffer_length,
        "kbits": args.kbits,
        "vbits": args.vbits,
        "promote_bit": args.promote_bit,
        "seed_bases": {
            "random": args.random_seed,
            "numpy": args.numpy_seed,
            "torch": args.torch_seed,
            "fewshot": args.fewshot_seed,
        },
    }
    if args.task == "gpqa_diamond_cot_n_shot":
        signature["gpqa_data"] = str(args.gpqa_data)
        signature["gpqa_data_sha256"] = (
            hashlib.sha256(args.gpqa_data.read_bytes()).hexdigest()
            if args.gpqa_data.is_file()
            else None
        )
    return signature


def _task_spec(args: argparse.Namespace):
    """Use the official local CSV when the gated GPQA Hub dataset is unavailable."""

    if args.task != "gpqa_diamond_cot_n_shot":
        return [args.task], None
    if not args.gpqa_data.is_file():
        raise FileNotFoundError(
            f"missing GPQA-Diamond data: {args.gpqa_data}; "
            "run `python examples/prepare_gpqa.py`"
        )
    import lm_eval
    from lm_eval.tasks._yaml_loader import load_yaml

    yaml_path = (
        Path(lm_eval.__file__).resolve().parent
        / "tasks" / "gpqa" / "cot_n_shot" / "gpqa_diamond_cot_n_shot.yaml"
    )
    config = load_yaml(yaml_path, resolve_func=True, recursive=True)
    config["dataset_path"] = "csv"
    config.pop("dataset_name", None)
    config["dataset_kwargs"] = {"data_files": {"train": str(args.gpqa_data)}}
    metadata = {
        "path": str(args.gpqa_data),
        "sha256": hashlib.sha256(args.gpqa_data.read_bytes()).hexdigest(),
    }
    return [config], metadata


def _result_base(args: argparse.Namespace, model_path: Path) -> Path:
    base = Path(args.output) / args.protocol / args.backend / model_path.name
    if args.limit is not None:
        base /= f"smoke_limit_{args.limit}"
    return base


def _summarize_repeats(repeat_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate numeric lm-eval metrics while preserving task/metric names."""

    summary: dict[str, Any] = {}
    if not repeat_results:
        return summary
    task_names = sorted({task for result in repeat_results for task in result})
    for task in task_names:
        metric_names = sorted({
            metric
            for result in repeat_results
            for metric in result.get(task, {})
            if not metric.endswith("_stderr")
        })
        task_summary: dict[str, Any] = {}
        for metric in metric_names:
            values = [
                float(result[task][metric])
                for result in repeat_results
                if isinstance(result.get(task, {}).get(metric), (int, float))
            ]
            if values:
                mean = sum(values) / len(values)
                task_summary[metric] = {
                    "values": values,
                    "mean": mean,
                    "min": min(values),
                    "max": max(values),
                    "max_deviation": max(abs(value - mean) for value in values),
                }
        summary[task] = task_summary
    return summary


def run_variant(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    from lm_eval.models.huggingface import HFLM
    from lm_eval.evaluator import simple_evaluate
    from transformers import AutoModelForCausalLM

    model_path = Path(args.model).expanduser()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"no local config.json found under {model_path}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(args.torch_seed)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": _dtype(args.dtype),
        "local_files_only": True,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(device).eval()
    cache, cache_config, resolved_variant = _cache_for_variant(args, variant)
    run_name = _run_name(args, variant, resolved_variant)
    experiment_signature = _experiment_signature(args, variant, resolved_variant)
    lm = HFLM(pretrained=model, batch_size=args.batch_size)
    gen_kwargs = _generation_kwargs(args, cache, model_path)
    task_lower = args.task.lower()
    if "humaneval" in task_lower and args.confirm_run_unsafe_code:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    task_spec, local_task_data = _task_spec(args)
    result = None
    repeat_results: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        repeat_path = (
            _result_base(args, model_path)
            / args.task
            / run_name
            / f"repeat_{repeat}.json"
        )
        if repeat_path.is_file():
            completed = json.loads(repeat_path.read_text())
            if completed.get("experiment_signature") != experiment_signature:
                raise RuntimeError(
                    f"checkpoint signature mismatch: {repeat_path}; "
                    "use a different --output for smoke/full or changed configurations"
                )
            repeat_results.append(completed["results"])
            result = {"results": completed["results"], "samples": completed.get("samples", {})}
            print(f"checkpoint: {repeat_path}")
            continue
        random_seed = args.random_seed + repeat
        evaluation_lm = lm
        request_checkpoint = None
        if args.request_checkpoint:
            from dart.eval_checkpoint import ExactSamplingCheckpointLM

            request_checkpoint = (
                _result_base(args, model_path)
                / args.task
                / run_name
                / f"repeat_{repeat}_requests.pt"
            )
            evaluation_lm = ExactSamplingCheckpointLM(
                lm,
                request_checkpoint,
                {**experiment_signature, "repeat": repeat},
            )
        result = simple_evaluate(
            model=evaluation_lm,
            tasks=task_spec,
            num_fewshot=_task_fewshot(args.task),
            batch_size=args.batch_size,
            limit=args.limit,
            log_samples=True,
            apply_chat_template="aime" in task_lower,
            fewshot_as_multiturn=False,
            gen_kwargs=gen_kwargs,
            random_seed=random_seed,
            numpy_random_seed=args.numpy_seed + repeat,
            torch_random_seed=args.torch_seed + repeat,
            fewshot_random_seed=args.fewshot_seed + repeat,
            confirm_run_unsafe_code=args.confirm_run_unsafe_code,
        )
        if result is None:
            raise RuntimeError("lm-eval returned no result on the local rank")
        repeat_results.append(_jsonable(result.get("results", {})))
        repeat_path.parent.mkdir(parents=True, exist_ok=True)
        repeat_path.write_text(json.dumps({
            "variant": variant,
            "run_name": run_name,
            "label": VARIANTS[variant]["label"],
            "resolved_variant": resolved_variant,
            "backend": args.backend,
            "protocol": args.protocol,
            "experiment_signature": experiment_signature,
            "repeat": repeat,
            "seeds": {
                "random": random_seed,
                "numpy": args.numpy_seed + repeat,
                "torch": args.torch_seed + repeat,
                "fewshot": args.fewshot_seed + repeat,
            },
            "model": str(model_path),
            "task": args.task,
            "config": _jsonable(vars(args)),
            "cache_config": _jsonable(cache_config.__dict__) if cache_config is not None else None,
            "local_task_data": local_task_data,
            "results": _jsonable(result.get("results", {})),
            "samples": _jsonable(result.get("samples", {})),
        }, ensure_ascii=False, indent=2) + "\n")
        if request_checkpoint is not None:
            evaluation_lm.discard()
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    output = {
        "variant": variant,
        "run_name": run_name,
        "label": VARIANTS[variant]["label"],
        "resolved_variant": resolved_variant,
        "backend": args.backend,
        "protocol": args.protocol,
        "protocol_config": PROTOCOLS[args.protocol],
        "experiment_signature": experiment_signature,
        "model": str(model_path),
        "task": args.task,
        "repeats": args.repeats,
        "limit": args.limit,
        "dtype": args.dtype,
        "device": str(device),
        "seeds": {
            "random": args.random_seed,
            "numpy": args.numpy_seed,
            "torch": args.torch_seed,
            "fewshot": args.fewshot_seed,
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "lm_eval": importlib.metadata.version("lm-eval"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_memory_bytes": peak_memory,
        "results": _jsonable(result.get("results", {})),
        "repeat_statistics": _summarize_repeats(repeat_results),
        "cache_config": _jsonable(cache_config.__dict__) if cache_config is not None else None,
        "local_task_data": local_task_data,
    }
    del lm, model, cache
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    summary_path = (
        _result_base(args, model_path)
        / args.task
        / f"{run_name}_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    output["result_path"] = str(summary_path)
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.repeats <= 0 or args.max_new_tokens <= 0:
        raise SystemExit("batch-size, repeats, and max-new-tokens must be positive")
    if args.promote_ratio is not None and not 0.0 <= args.promote_ratio <= 1.0:
        raise SystemExit("promote-ratio must be between 0 and 1")
    variants = tuple(VARIANTS) if args.variant == "all" else (args.variant,)
    outputs = [run_variant(args, variant) for variant in variants]
    if len(outputs) > 1:
        matrix_path = (
            _result_base(args, Path(args.model).expanduser())
            / args.task
            / "variant_matrix.json"
        )
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        matrix_path.write_text(json.dumps({
            "model": args.model,
            "task": args.task,
            "protocol": args.protocol,
            "backend": args.backend,
            "seeds": {
                "random": args.random_seed,
                "numpy": args.numpy_seed,
                "torch": args.torch_seed,
                "fewshot": args.fewshot_seed,
            },
            "variants": outputs,
        }, ensure_ascii=False, indent=2) + "\n")
        print(f"variant matrix: {matrix_path}")
    print(json.dumps(outputs[0] if len(outputs) == 1 else outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
