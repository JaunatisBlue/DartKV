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
GPQA_DIAMOND_SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"
DEFAULT_OPERATOR_AUDIT = REPO_ROOT / "experiments" / "kitty_operator_audit.json"
DEFAULT_FIGURE5 = REPO_ROOT / "experiments" / "kitty_figure5_reproduction.json"
REQUIRED_TABLE3_CELLS = (
    "kitty:humaneval_instruct",
    "kitty-pro:gsm8k_cot_llama",
    "kitty-pro:minerva_math_algebra",
    "kitty-pro:humaneval_instruct",
    "kitty-pro:gpqa_diamond_cot_n_shot",
)
REQUIRED_TABLE4_CELLS = (
    "kitty:aime24",
    "kitty:aime25",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--accuracy-results", type=Path, default=Path("results/kitty_qwen8_paper_b16"))
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


def accuracy_signature_issues(table_report: dict, table_name: str) -> list[dict]:
    """Reject results produced with a different stochastic accuracy protocol."""

    expected_max_new_tokens = 4096 if table_name == "table3" else 32768
    issues = []
    checked: set[str] = set()
    for cell in table_report["cells"]:
        summary_path = cell["summary_path"]
        if summary_path in checked or not Path(summary_path).is_file():
            continue
        checked.add(summary_path)
        summary = json.loads(Path(summary_path).read_text())
        signature = summary.get("experiment_signature", {})
        expected = {
            "model_name": "Qwen-8B",
            "backend": "kitty-reference",
            "protocol": "paper",
            "batch_size": 16,
            "limit": None,
            "max_new_tokens": expected_max_new_tokens,
            "minimum_repeats": 1,
        }
        actual = {
            "model_name": Path(str(signature.get("model", ""))).name,
            "backend": signature.get("backend"),
            "protocol": signature.get("protocol"),
            "batch_size": signature.get("batch_size"),
            "limit": signature.get("limit"),
            "max_new_tokens": signature.get("max_new_tokens"),
            "repeats": summary.get("repeats"),
        }
        if cell["task"] == "gpqa_diamond_cot_n_shot":
            expected["gpqa_data_sha256"] = GPQA_DIAMOND_SHA256
            actual["gpqa_data_sha256"] = signature.get("gpqa_data_sha256")
        valid = (
            actual["model_name"] == expected["model_name"]
            and actual["backend"] == expected["backend"]
            and actual["protocol"] == expected["protocol"]
            and actual["batch_size"] == expected["batch_size"]
            and actual["limit"] is None
            and actual["max_new_tokens"] == expected["max_new_tokens"]
            and isinstance(actual["repeats"], int)
            and actual["repeats"] >= expected["minimum_repeats"]
            and actual.get("gpqa_data_sha256") == expected.get("gpqa_data_sha256")
        )
        if not valid:
            issues.append({"summary_path": summary_path, "expected": expected, "actual": actual})
    return issues


def check(args: argparse.Namespace) -> dict:
    manifest = json.loads(args.manifest.read_text())
    common_table_args = {
        "manifest": args.manifest,
        "results": args.accuracy_results,
        "model_key": "Qwen3-8B",
        "model_dir": "Qwen-8B",
        "protocol": "paper",
        "backend": "kitty-reference",
        "absolute_tolerance": None,
    }
    # The user-selected fast scope intentionally excludes FP16/KIVI, Figure 4,
    # and Kitty base cells other than HumanEval.
    table3 = audit_table(argparse.Namespace(
        table="table3", variants=None, cells=REQUIRED_TABLE3_CELLS, **common_table_args
    ))
    table4 = audit_table(argparse.Namespace(
        table="table4", variants=None, cells=REQUIRED_TABLE4_CELLS, **common_table_args
    ))
    table3_signature_issues = accuracy_signature_issues(table3, "table3")
    table4_signature_issues = accuracy_signature_issues(table4, "table4")

    figure4 = {
        "required": False,
        "reason": "excluded from the user-selected fast reproduction scope",
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
        table3["reproduced"] and not table3_signature_issues,
        table4["reproduced"] and not table4_signature_issues,
    ))
    return {
        "scope": "user-selected Qwen3-8B Kitty cells; one complete repeat minimum",
        "required_accuracy_cells": {
            "table3": list(REQUIRED_TABLE3_CELLS),
            "table4": list(REQUIRED_TABLE4_CELLS),
        },
        "passed": passed,
        "operator_audit": {
            "passed": operator.get("passed", False),
            "path": str(args.operator_audit),
        },
        "figure5": figure5,
        "table3": {
            "passed": table3["reproduced"] and not table3_signature_issues,
            "counts": table3["counts"],
            "signature_issues": table3_signature_issues,
            "incomplete_cells": [cell for cell in table3["cells"] if cell["status"] != "matched"],
        },
        "table4": {
            "passed": table4["reproduced"] and not table4_signature_issues,
            "counts": table4["counts"],
            "signature_issues": table4_signature_issues,
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
