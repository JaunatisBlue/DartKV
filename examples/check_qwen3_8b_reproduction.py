"""Audit the complete Qwen3-8B-only Kitty reproduction deliverables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from examples.check_kitty_reproduction import DEFAULT_MANIFEST, audit as audit_table
except ModuleNotFoundError:  # direct ``python examples/...py`` execution
    from check_kitty_reproduction import DEFAULT_MANIFEST, audit as audit_table


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPERATOR_AUDIT = REPO_ROOT / "experiments" / "kitty_operator_audit.json"
DEFAULT_FIGURE5 = REPO_ROOT / "experiments" / "kitty_figure5_reproduction.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--accuracy-results", type=Path, default=Path("results/kitty_full16_native"))
    parser.add_argument("--figure4-results", type=Path, default=Path("results/kitty_sweep"))
    parser.add_argument("--operator-audit", type=Path, default=DEFAULT_OPERATOR_AUDIT)
    parser.add_argument("--figure5", type=Path, default=DEFAULT_FIGURE5)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments" / "qwen3_8b_reproduction_status.json",
    )
    return parser


def figure4_expected_paths(base: Path, manifest: dict) -> list[Path]:
    figure = manifest["figure4"]
    root = base / "paper" / "kitty-reference" / "Qwen-8B"
    paths = []
    for task in figure["tasks"]:
        for selection in figure["selection"]:
            for ratio in figure["promotion_ratios"]:
                ratio_name = str(ratio).replace(".", "_")
                paths.append(
                    root / task / f"kitty-pro__{selection}__pr{ratio_name}_summary.json"
                )
    return paths


def check_figure5(path: Path) -> dict:
    if not path.is_file():
        return {"passed": False, "reason": "missing manifest", "path": str(path)}
    report = json.loads(path.read_text())
    final = report.get("final_native_kernel_reproduction", {})
    required_points = ("batch32", "batch64", "batch128", "batch256")
    missing_points = [name for name in required_points if name not in final]
    result_files = {
        name: Path(final[name].get("result", ""))
        for name in required_points
        if name in final
    }
    missing_results = [name for name, result_path in result_files.items() if not result_path.is_file()]
    boundary = final.get("batch512_boundary", {})
    passed = (
        not missing_points
        and not missing_results
        and final.get("batch_size_gain") == 8.0
        and final.get("throughput_within_paper_range") is True
        and boundary.get("status") == "oom"
        and final.get("operator_facade") == "dart.kitty_kernels"
    )
    return {
        "passed": passed,
        "missing_points": missing_points,
        "missing_result_files": missing_results,
        "batch_size_gain": final.get("batch_size_gain"),
        "generated_throughput_gain": final.get("generated_throughput_gain"),
        "batch512_status": boundary.get("status"),
        "operator_facade": final.get("operator_facade"),
    }


def check(args: argparse.Namespace) -> dict:
    manifest = json.loads(args.manifest.read_text())
    table_args = {
        "manifest": args.manifest,
        "results": args.accuracy_results,
        "model_key": "Qwen3-8B",
        "model_dir": "Qwen-8B",
        "protocol": "paper",
        "backend": "kitty-reference",
        "absolute_tolerance": None,
    }
    table3 = audit_table(argparse.Namespace(table="table3", **table_args))
    table4 = audit_table(argparse.Namespace(table="table4", **table_args))

    expected_figure4 = figure4_expected_paths(args.figure4_results, manifest)
    missing_figure4 = [str(path) for path in expected_figure4 if not path.is_file()]
    figure4 = {
        "passed": not missing_figure4,
        "expected": len(expected_figure4),
        "completed": len(expected_figure4) - len(missing_figure4),
        "missing": missing_figure4,
    }

    operator = (
        json.loads(args.operator_audit.read_text())
        if args.operator_audit.is_file()
        else {"passed": False, "reason": "missing operator audit"}
    )
    figure5 = check_figure5(args.figure5)
    passed = all((
        operator.get("passed") is True,
        figure5["passed"],
        table3["reproduced"],
        table4["reproduced"],
        figure4["passed"],
    ))
    return {
        "scope": "Qwen3-8B only",
        "passed": passed,
        "operator_audit": {
            "passed": operator.get("passed", False),
            "path": str(args.operator_audit),
        },
        "figure5": figure5,
        "table3": {
            "passed": table3["reproduced"],
            "counts": table3["counts"],
            "incomplete_cells": [cell for cell in table3["cells"] if cell["status"] != "matched"],
        },
        "table4": {
            "passed": table4["reproduced"],
            "counts": table4["counts"],
            "incomplete_cells": [cell for cell in table4["cells"] if cell["status"] != "matched"],
        },
        "figure4": figure4,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = check(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
