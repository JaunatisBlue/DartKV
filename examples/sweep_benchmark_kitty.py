"""Sweep Figure 5 batch sizes until the local model/cache reaches OOM."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", choices=("dense", "dart", "kitty-reference"), default="dart")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--output", default="results/kitty_latency_sweep")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Path(__file__).with_name("benchmark_kitty.py")
    records: list[dict] = []
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for batch in args.batches:
        command = [
            sys.executable,
            str(runner),
            "--model", args.model,
            "--cache", args.cache,
            "--batch-size", str(batch),
            "--max-seq-len", str(args.max_seq_len),
            "--warmup-runs", str(args.warmup_runs),
            "--repeat-runs", str(args.repeat_runs),
            "--device", args.device,
            "--dtype", args.dtype,
            "--output", str(output_dir),
        ]
        print("[benchmark sweep]", " ".join(command), flush=True)
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        record: dict = {"batch_size": batch, "returncode": completed.returncode}
        if completed.returncode == 0:
            path = output_dir / f"{Path(args.model).name}_{args.cache}_b{batch}_l{args.max_seq_len}.json"
            if path.is_file():
                record.update(json.loads(path.read_text()))
        else:
            record["stderr_tail"] = completed.stderr[-4000:]
            record["stdout_tail"] = completed.stdout[-1000:]
        records.append(record)
        (output_dir / "sweep_progress.json").write_text(json.dumps({
            "model": args.model,
            "cache": args.cache,
            "device": args.device,
            "dtype": args.dtype,
            "max_seq_len": args.max_seq_len,
            "records": records,
        }, ensure_ascii=False, indent=2) + "\n")
        if completed.returncode != 0 and "out of memory" in (completed.stderr + completed.stdout).lower():
            print(f"[benchmark sweep] OOM at batch={batch}; stopping", flush=True)
            break
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
