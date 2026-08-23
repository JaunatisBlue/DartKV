"""Run a fixed local prompt suite for one Qwen3 cache configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_qwen3 import build_parser, run


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.add_argument("--prompt-file", default="examples/prompts.txt")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    prompt_path = Path(args.prompt_file)
    prompts = [line.strip() for line in prompt_path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("limit must be positive")
        prompts = prompts[: args.limit]
    if not prompts:
        raise SystemExit(f"no prompts found in {prompt_path}")

    records = []
    for index, prompt in enumerate(prompts):
        args.prompt = prompt
        record = run(args)
        record["prompt_id"] = index
        records.append(record)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    variant = f"{args.cache}_{'mixed' if args.promote_ratio > 0 else 'uniform'}" if args.cache == "dart" else "dense"
    output_path = output_dir / f"suite_{variant}_{args.seed}.json"
    summary = {
        "model": args.model,
        "cache": args.cache,
        "prompt_file": str(prompt_path),
        "count": len(records),
        "records": records,
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({**summary, "result_path": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
