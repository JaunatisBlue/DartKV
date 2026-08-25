"""Compare local Kitty reproduction summaries with the paper's reported cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments" / "kitty_paper_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=Path("results/kitty_repro"))
    parser.add_argument("--table", choices=("table3", "table4"), default="table3")
    parser.add_argument("--model-key", default="Qwen3-8B")
    parser.add_argument("--model-dir", default="Qwen-8B")
    parser.add_argument("--protocol", choices=("paper", "artifact"), default="paper")
    parser.add_argument("--backend", choices=("kitty-reference", "dart"), default="kitty-reference")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Only audit these method variants (for example: kitty kitty-pro)",
    )
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=None,
        help="Accuracy-point tolerance; by default each cell uses its published max deviation",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _observed_mean(summary: dict[str, Any], task: str, metric: str) -> float | None:
    metric_summary = summary.get("repeat_statistics", {}).get(task, {}).get(metric)
    if isinstance(metric_summary, dict) and isinstance(metric_summary.get("mean"), (int, float)):
        return float(metric_summary["mean"]) * 100.0
    value = summary.get("results", {}).get(task, {}).get(metric)
    if isinstance(value, (int, float)):
        return float(value) * 100.0
    return None


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text())
    targets = manifest[args.table][args.model_key]
    tasks = manifest["tasks"]
    base_dir = args.results / args.protocol / args.backend / args.model_dir
    cells: list[dict[str, Any]] = []
    for variant, variant_targets in targets.items():
        if getattr(args, "variants", None) and variant not in args.variants:
            continue
        for task, target in variant_targets.items():
            if task == "average":
                continue
            target_mean, paper_max_deviation = target
            metric = tasks[task]["metric"]
            summary_path = base_dir / task / f"{variant}_summary.json"
            observed = None
            if summary_path.is_file():
                observed = _observed_mean(json.loads(summary_path.read_text()), task, metric)
            tolerance = args.absolute_tolerance
            if tolerance is None:
                tolerance = float(paper_max_deviation)
            delta = None if observed is None else observed - float(target_mean)
            if observed is None:
                status = "missing"
            elif abs(delta) <= tolerance:
                status = "matched"
            else:
                status = "outside_tolerance"
            cells.append({
                "variant": variant,
                "task": task,
                "metric": metric,
                "target_percent": target_mean,
                "paper_max_deviation": paper_max_deviation,
                "observed_percent": observed,
                "delta_points": delta,
                "tolerance_points": tolerance,
                "status": status,
                "summary_path": str(summary_path),
            })
    counts = {
        status: sum(cell["status"] == status for cell in cells)
        for status in ("matched", "outside_tolerance", "missing")
    }
    return {
        "table": args.table,
        "model": args.model_key,
        "protocol": args.protocol,
        "backend": args.backend,
        "results_base": str(base_dir),
        "criterion": (
            f"absolute delta <= {args.absolute_tolerance} percentage points"
            if args.absolute_tolerance is not None
            else "absolute delta <= the paper's reported maximum deviation for each cell"
        ),
        "counts": counts,
        "complete": counts["missing"] == 0,
        "reproduced": counts["missing"] == 0 and counts["outside_tolerance"] == 0,
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0 if report["reproduced"] or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
