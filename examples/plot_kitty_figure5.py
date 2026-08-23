"""Aggregate local Kitty Figure 5 JSON results and render the two paper panels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


LABELS = {
    "dense": "HF Dynamic FP16",
    "static": "HF Static FP16",
    "quanto": "HF KIVI INT4 (Quanto)",
    "hqq": "HF KIVI INT4 (HQQ)",
    "kitty-engine": "Kitty-Pro",
}

COLORS = {
    "dense": "#4c72b0",
    "static": "#55a868",
    "quanto": "#c44e52",
    "hqq": "#8172b2",
    "kitty-engine": "#dd3d3d",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/kitty_figure5_paper"))
    parser.add_argument("--protocol", default="paper")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=Path("results/kitty_figure5_paper/figure5.png"))
    parser.add_argument("--memory-metric", choices=("allocated", "reserved"), default="allocated")
    parser.add_argument(
        "--caches",
        nargs="+",
        default=["dense", "static", "quanto", "hqq", "kitty-engine"],
        choices=tuple(LABELS),
    )
    return parser


def load_points(
    result_dir: Path,
    protocol: str,
    max_seq_len: int,
    caches: set[str],
) -> list[dict]:
    points = []
    for path in sorted(result_dir.glob("*.json")):
        try:
            result = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        cache = result.get("cache")
        if (
            cache not in caches
            or result.get("protocol") != protocol
            or result.get("max_seq_len") != max_seq_len
        ):
            continue
        points.append({
            "cache": cache,
            "label": LABELS[cache],
            "batch_size": result["batch_size"],
            "generated_tokens_per_second": result["generated_tokens_per_second"],
            "sequence_tokens_per_second": result["sequence_tokens_per_second"],
            "peak_memory_allocated_bytes": result.get("peak_memory_bytes"),
            "peak_memory_reserved_bytes": result.get("peak_memory_reserved_bytes"),
            "result": str(path),
        })
    return sorted(points, key=lambda point: (point["cache"], point["batch_size"]))


def render(points: list[dict], output: Path, memory_metric: str) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict]] = defaultdict(list)
    for point in points:
        grouped[point["cache"]].append(point)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.25))
    for cache, cache_points in grouped.items():
        memory_key = f"peak_memory_{memory_metric}_bytes"
        memory_points = [point for point in cache_points if point[memory_key] is not None]
        memory_batches = [point["batch_size"] for point in memory_points]
        memory_gb = [point[memory_key] / 1e9 for point in memory_points]
        batches = [point["batch_size"] for point in cache_points]
        throughput = [point["generated_tokens_per_second"] for point in cache_points]
        style = {"marker": "o", "linewidth": 2, "color": COLORS[cache], "label": LABELS[cache]}
        axes[0].plot(memory_batches, memory_gb, **style)
        axes[1].plot(batches, throughput, **style)
    axes[0].set(
        xlabel="Batch Size (# requests)",
        ylabel=f"Peak GPU Memory ({memory_metric}, GB)",
    )
    axes[1].set(xlabel="Batch Size (# requests)", ylabel="Throughput (generated tokens/s)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Kitty Figure 5 local reproduction: Qwen3-8B, sequence length 8192")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    points = load_points(args.results, args.protocol, args.max_seq_len, set(args.caches))
    if not points:
        raise SystemExit(f"no matching benchmark JSON files under {args.results}")
    render(points, args.output, args.memory_metric)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps({
        "protocol": args.protocol,
        "max_seq_len": args.max_seq_len,
        "memory_metric": args.memory_metric,
        "points": points,
        "plot": str(args.output),
    }, indent=2) + "\n")
    print(f"plot: {args.output}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
