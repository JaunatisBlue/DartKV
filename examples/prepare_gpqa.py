"""Download and validate the official GPQA release for local lm-eval use."""

from __future__ import annotations

import argparse
import csv
import hashlib
import urllib.request
import zipfile
from pathlib import Path


OFFICIAL_URL = "https://raw.githubusercontent.com/idavidrein/gpqa/main/dataset.zip"
ZIP_SHA256 = "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc"
DIAMOND_SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"
PASSWORD = b"deserted-untie-orchid"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "dataset.zip"
    if not archive.is_file() or _sha256(archive) != ZIP_SHA256:
        urllib.request.urlretrieve(OFFICIAL_URL, archive)
    if _sha256(archive) != ZIP_SHA256:
        raise RuntimeError("official GPQA archive SHA-256 mismatch")
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(output, pwd=PASSWORD)
    diamond = output / "dataset" / "gpqa_diamond.csv"
    if _sha256(diamond) != DIAMOND_SHA256:
        raise RuntimeError("GPQA-Diamond CSV SHA-256 mismatch")
    with diamond.open(newline="") as handle:
        rows = sum(1 for _ in csv.DictReader(handle))
    if rows != 198:
        raise RuntimeError(f"GPQA-Diamond should contain 198 rows, found {rows}")
    return {
        "source": OFFICIAL_URL,
        "archive": str(archive),
        "archive_sha256": ZIP_SHA256,
        "diamond": str(diamond),
        "diamond_sha256": DIAMOND_SHA256,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/home/yx/.cache/dartkv/gpqa"))
    args = parser.parse_args(argv)
    for key, value in prepare(args.output).items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
