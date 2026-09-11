from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_output(path: Path) -> dict[int, tuple[float, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            int(row["year"]): (
                float(row["raw_score"]),
                float(row["published_global_standardized_score"]),
            )
            for row in csv.DictReader(handle)
        }


def read_reference(path: Path) -> dict[int, tuple[float, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            int(row["year"]): (float(row["score"]), float(row["all_stdscore"]))
            for row in csv.DictReader(handle, delimiter="\t")
            if row["lang"].strip('"') == "fre"
        }


def report(label: str, reproduced: np.ndarray, reference: np.ndarray) -> None:
    correlation = np.corrcoef(reproduced, reference)[0, 1]
    difference = np.abs(reproduced - reference)
    print(
        f"{label}: correlation={correlation:.8f}, "
        f"MAE={difference.mean():.8g}, max_abs_error={difference.max():.8g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare reproduced French scores with the authors' f.tab."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", default=Path("data/reference/f.tab"), type=Path)
    args = parser.parse_args()

    output = read_output(args.output)
    reference = read_reference(args.reference)
    years = sorted(output.keys() & reference.keys())
    if not years:
        raise SystemExit("No overlapping French years found.")
    reproduced = np.array([output[year] for year in years])
    expected = np.array([reference[year] for year in years])
    print(f"Compared {len(years)} years ({years[0]}-{years[-1]})")
    report("raw", reproduced[:, 0], expected[:, 0])
    report("standardized", reproduced[:, 1], expected[:, 1])


if __name__ == "__main__":
    main()

