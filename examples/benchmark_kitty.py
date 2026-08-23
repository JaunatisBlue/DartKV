"""Benchmark Kitty-style dense/local-buffer generation on a local model.

This is the Dart-side counterpart of ``reference/code/Kitty/latency_benchmarking``.
It fixes the prompt, sequence length, batch size, warmup, repeat count, device,
and seed in the result JSON, then reports generation throughput and KV storage.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

from dart import DartKVCacheConfig
from dart.integrations import DartHFCache


DEFAULT_PROMPT = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--protocol",
        choices=("paper", "artifact"),
        default="paper",
        help="paper uses the ~100-token Figure 5 prompt length; artifact keeps the full checked-in prompt",
    )
    parser.add_argument(
        "--cache",
        choices=("dense", "dart", "kitty-reference", "kitty-engine"),
        default="dart",
        help="kitty-engine uses the checked-in Triton/custom Qwen3 implementation",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Override the Kitty reference prompt; omitted uses prompt-choice below",
    )
    parser.add_argument("--prompt-choice", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--paper-prompt-tokens",
        type=int,
        default=100,
        help="Exact prompt-token budget used for the paper protocol",
    )
    parser.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the model chat template as in Kitty's latency script",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--bits", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--promote-ratio", type=float, default=0.25)
    parser.add_argument("--sink-tokens", type=int, default=32)
    parser.add_argument("--buffer-length", type=int, default=128)
    parser.add_argument("--key-group-size", type=int, default=128)
    parser.add_argument("--value-group-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="flash_attention_2",
        help="Prefill attention backend; kitty-engine uses its Triton kernel after prefill",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--output", default="results/kitty_latency")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the first measured repeat and export a CUDA summary table",
    )
    parser.add_argument(
        "--profile-trace",
        action="store_true",
        help="Also export a large Chrome trace (disabled by default)",
    )
    parser.add_argument("--profile-row-limit", type=int, default=40)
    return parser


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _limit_prompt_inputs(inputs, target_tokens: int, tail_tokens: int = 8):
    """Fit a reference prompt to Figure 5's token budget while retaining its chat suffix."""
    current_tokens = inputs.input_ids.shape[-1]
    if target_tokens <= 0:
        raise ValueError("paper-prompt-tokens must be positive")
    if current_tokens < target_tokens:
        raise ValueError(
            f"reference prompt has {current_tokens} tokens, fewer than requested {target_tokens}"
        )
    if current_tokens == target_tokens:
        return inputs
    preserved_tail = min(tail_tokens, target_tokens // 2)
    head_tokens = target_tokens - preserved_tail
    for name in ("input_ids", "attention_mask"):
        tensor = getattr(inputs, name, None)
        if tensor is not None:
            setattr(inputs, name, torch.cat((tensor[:, :head_tokens], tensor[:, -preserved_tail:]), dim=-1))
    return inputs


def _dense_storage(cache) -> int:
    tensors = []
    for key_cache, value_cache in zip(cache.key_cache, cache.value_cache, strict=True):
        tensors.extend((key_cache, value_cache))
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _kitty_engine_storage(cache) -> int:
    tensors = []
    for layer in cache.kv_cache:
        for name in (
            "KeyCache",
            "KeyCache_metadata",
            "ValueCache",
            "ValueCache_metadata",
            "PageTable_K",
            "PageTable_V",
            "Sink_Buffer_K",
            "Sink_Buffer_V",
            "Q_Buffer_K",
            "Q_Buffer_V",
            "Local_Buffer_V",
        ):
            tensor = getattr(layer, name, None)
            if tensor is not None:
                tensors.append(tensor)
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _new_cache(args: argparse.Namespace):
    if args.cache == "dense":
        from transformers import DynamicCache

        return DynamicCache()
    if args.cache == "kitty-reference":
        reference_src = Path(__file__).resolve().parents[1] / "reference" / "code" / "Kitty" / "src"
        sys.path.insert(0, str(reference_src))
        from kitty_sim import KittyKVCache, KittyKVCacheConfig

        return KittyKVCache(KittyKVCacheConfig(
            sink_length=args.sink_tokens,
            buffer_length=args.buffer_length,
            group_size=args.key_group_size,
            kbits=args.bits,
            vbits=args.bits,
            promote_ratio=args.promote_ratio,
            promote_bit=4,
            channel_selection=1,
        ))
    if args.cache == "kitty-engine":
        raise RuntimeError("kitty-engine cache requires _new_engine_cache(model_config, args)")
    return DartHFCache(DartKVCacheConfig(
        bits=args.bits,
        key_bits=args.bits,
        value_bits=args.bits,
        key_group_size=args.key_group_size,
        value_group_size=args.value_group_size,
        sink_tokens=args.sink_tokens,
        page_size=args.buffer_length,
        local_tokens=1,
        value_local_tokens=args.buffer_length,
        hold_partial_pages=True,
        promote_bits=4,
        promote_ratio=args.promote_ratio,
        metadata_dtype=torch.float16,
    ))


def _new_engine_cache(args: argparse.Namespace, model_config):
    reference_src = Path(__file__).resolve().parents[1] / "reference" / "code" / "Kitty" / "src"
    if str(reference_src) not in sys.path:
        sys.path.insert(0, str(reference_src))
    from kitty.kvcache import get_kvcache_kitty

    return get_kvcache_kitty(model_config, args.batch_size, args.max_seq_len)


def _kitty_prompt(choice: int) -> tuple[str, str]:
    reference_src = Path(__file__).resolve().parents[1] / "reference" / "code" / "Kitty" / "src"
    if str(reference_src) not in sys.path:
        sys.path.insert(0, str(reference_src))
    from kitty_sim.cli.utils_cli import get_prompt

    return get_prompt(choice)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.batch_size, args.max_seq_len, args.warmup_runs, args.repeat_runs) <= 0:
        raise SystemExit("batch-size, max-seq-len, warmup-runs, and repeat-runs must be positive")
    model_path = Path(args.model).expanduser()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"no local config.json found under {model_path}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if args.cache == "kitty-engine":
        reference_src = Path(__file__).resolve().parents[1] / "reference" / "code" / "Kitty" / "src"
        if str(reference_src) not in sys.path:
            sys.path.insert(0, str(reference_src))
        from kitty.models.qwen3 import Qwen3ForCausalLM_Kitty
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        model = Qwen3ForCausalLM_Kitty.from_pretrained(
            model_path,
            config=model_config,
            torch_dtype=_dtype(args.dtype),
            local_files_only=True,
            attn_implementation=args.attn_implementation,
        ).to(device).eval()
    else:
        model_config = None
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=_dtype(args.dtype),
            local_files_only=True,
            attn_implementation=args.attn_implementation,
        ).to(device).eval()
    task_name, reference_prompt = _kitty_prompt(args.prompt_choice)
    prompt = args.prompt if args.prompt is not None else reference_prompt
    texts = [prompt] * args.batch_size
    if args.chat_template:
        texts = tokenizer.apply_chat_template(
            [[{"role": "user", "content": text}] for text in texts],
            add_generation_prompt=True,
            tokenize=False,
        )
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    if args.protocol == "paper" and args.prompt is None:
        inputs = _limit_prompt_inputs(inputs, args.paper_prompt_tokens)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    if input_ids.shape[-1] >= args.max_seq_len:
        raise SystemExit("max-seq-len must exceed the tokenized prompt length")

    def generate_once():
        cache = _new_engine_cache(args, model_config) if args.cache == "kitty-engine" else _new_cache(args)
        with torch.inference_mode():
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_length": args.max_seq_len,
                "max_new_tokens": None,
                "return_dict_in_generate": False,
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "eos_token_id": None,
                "use_cache": True,
                "past_key_values": cache,
            }
            if args.cache == "kitty-engine":
                generate_kwargs["disable_compile"] = True
            output = model.generate(**generate_kwargs)
        _sync(device)
        if isinstance(cache, DartHFCache):
            storage_bytes = cache.storage_bytes
            dense_bytes = cache.dense_bytes
            ratio = cache.compression_ratio
        elif args.cache == "kitty-engine":
            storage_bytes = _kitty_engine_storage(cache)
            dense_bytes = (
                model.config.num_hidden_layers
                * 2
                * args.batch_size
                * model.config.num_key_value_heads
                * model.config.head_dim
                * output.shape[-1]
                * torch.tensor([], dtype=_dtype(args.dtype)).element_size()
            )
            ratio = dense_bytes / storage_bytes if storage_bytes else 1.0
        else:
            storage_bytes = _dense_storage(cache)
            dense_bytes = storage_bytes
            ratio = 1.0
        del cache
        return output, storage_bytes, dense_bytes, ratio

    for _ in range(args.warmup_runs):
        generate_once()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    records = []
    profile = None
    for repeat_index in range(args.repeat_runs):
        if args.profile and repeat_index == 0:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profile = torch.profiler.profile(
                activities=activities,
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
            )
            profile.__enter__()
        try:
            start = time.perf_counter()
            output, storage_bytes, dense_bytes, ratio = generate_once()
            elapsed_ms = (time.perf_counter() - start) * 1000
        finally:
            if profile is not None and repeat_index == 0:
                profile.__exit__(None, None, None)
        timings.append(elapsed_ms)
        records.append({
            "elapsed_ms": elapsed_ms,
            "storage_bytes": storage_bytes,
            "dense_bytes": dense_bytes,
            "compression_ratio": ratio,
            "output_tokens": int(output.shape[-1]),
            "first_output_token_ids": output[0, : min(32, output.shape[-1])].tolist(),
        })
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    avg_ms = sum(timings) / len(timings)
    generated_tokens = args.batch_size * (args.max_seq_len - input_ids.shape[-1])
    sequence_tokens = args.batch_size * args.max_seq_len
    elapsed_seconds = avg_ms / 1000
    result = {
        "model": str(model_path),
        "cache": args.cache,
        "task_name": task_name,
        "protocol": args.protocol,
        "prompt_choice": args.prompt_choice,
        "prompt_source": "custom" if args.prompt is not None else "kitty-reference",
        "chat_template": args.chat_template,
        "batch_size": args.batch_size,
        "prompt_tokens": int(input_ids.shape[-1]),
        "max_seq_len": args.max_seq_len,
        "generated_tokens": generated_tokens,
        "sequence_tokens": sequence_tokens,
        "warmup_runs": args.warmup_runs,
        "repeat_runs": args.repeat_runs,
        "device": str(device),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "seed": args.seed,
        "average_elapsed_ms": avg_ms,
        "tokens_per_second": sequence_tokens / elapsed_seconds,
        "sequence_tokens_per_second": sequence_tokens / elapsed_seconds,
        "generated_tokens_per_second": generated_tokens / elapsed_seconds,
        "peak_memory_bytes": peak_memory,
        "records": records,
        "dart_config": vars(args) if args.cache == "dart" else None,
        "reference_simulation": args.cache == "kitty-reference",
        "reference_engine": args.cache == "kitty-engine",
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_trace = None
    profile_table = None
    output_stem = f"{model_path.name}_{args.cache}_{args.protocol}_b{args.batch_size}_l{args.max_seq_len}"
    if profile is not None:
        profile_table = output_dir / f"{output_stem}_profile.txt"
        if args.profile_trace:
            profile_trace = output_dir / f"{output_stem}_profile.json"
            profile.export_chrome_trace(str(profile_trace))
        sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        profile_table.write_text(
            profile.key_averages().table(sort_by=sort_by, row_limit=args.profile_row_limit) + "\n"
        )
        result["profile_trace"] = str(profile_trace) if profile_trace is not None else None
        result["profile_table"] = str(profile_table)
    output_path = output_dir / (
        f"{output_stem}.json"
    )
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "result_path": str(output_path)}, indent=2))
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
