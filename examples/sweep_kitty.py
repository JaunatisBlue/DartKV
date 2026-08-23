"""Run the Figure 4 promotion-ratio sweep with resumable child experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_RATIOS = tuple(index / 10 for index in range(11))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["gsm8k_cot_llama", "minerva_math_algebra"],
    )
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--selections", nargs="+", choices=("random", "magnitude"), default=["random", "magnitude"])
    parser.add_argument("--backend", choices=("kitty-reference", "dart"), default="kitty-reference")
    parser.add_argument("--protocol", choices=("paper", "artifact"), default="paper")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--output", default="results/kitty_sweep")
    parser.add_argument("--confirm-run-unsafe-code", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if any(not 0.0 <= ratio <= 1.0 for ratio in args.ratios):
        raise SystemExit("all ratios must be between 0 and 1")
    records: list[dict] = []
    runner = Path(__file__).with_name("reproduce_kitty.py")
    for task in args.tasks:
        for selection in args.selections:
            for ratio in args.ratios:
                command = [
                    sys.executable,
                    str(runner),
                    "--model", args.model,
                    "--task", task,
                    "--variant", "kitty-pro",
                    "--backend", args.backend,
                    "--protocol", args.protocol,
                    "--device", args.device,
                    "--dtype", args.dtype,
                    "--batch-size", str(args.batch_size),
                    "--repeats", str(args.repeats),
                    "--max-new-tokens", str(args.max_new_tokens),
                    "--promote-ratio", str(ratio),
                    "--channel-selection", selection,
                    "--output", args.output,
                ]
                if args.limit is not None:
                    command.extend(("--limit", str(args.limit)))
                if args.confirm_run_unsafe_code:
                    command.append("--confirm-run-unsafe-code")
                print("[sweep]", " ".join(command), flush=True)
                completed = subprocess.run(command, check=False)
                record = {
                    "task": task,
                    "selection": selection,
                    "promote_ratio": ratio,
                    "returncode": completed.returncode,
                }
                records.append(record)
                Path(args.output).mkdir(parents=True, exist_ok=True)
                (Path(args.output) / "sweep_progress.json").write_text(
                    json.dumps({
                        "model": args.model,
                        "backend": args.backend,
                        "protocol": args.protocol,
                        "tasks": args.tasks,
                        "ratios": args.ratios,
                        "selections": args.selections,
                        "records": records,
                    }, ensure_ascii=False, indent=2) + "\n"
                )
                if completed.returncode != 0:
                    return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
