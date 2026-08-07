#!/usr/bin/env python3
"""Build small, auditable publication tables from the frozen result archive."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def summarize_synthetic(source: Path, output_dir: Path) -> None:
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    detailed: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        control = row["control"]
        if control == "positive":
            rank = int(row["rank"])
            true_rank = int(row["true_rank"])
            relation = "lower" if rank < true_rank else "true" if rank == true_rank else "higher"
            theta = f'{float(row["theta_degrees"]):.0f}'
            p = row["p"]
        else:
            relation = "searched"
            theta = "negative"
            p = row["p"]
        grouped[(control, theta, relation)].append(row)
        detailed[(control, p, theta, relation)].append(row)

    fields = [
        "control", "p", "theta_degrees", "rank_relation", "n", "passes",
        "pass_rate", "median_selective_gap", "median_recovered_ceiling_fraction",
    ]

    def write(path: Path, groups: dict[tuple[str, ...], list[dict[str, str]]], include_p: bool) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for key in sorted(groups, key=lambda item: tuple(str(x) for x in item)):
                group = groups[key]
                if include_p:
                    control, p, theta, relation = key
                else:
                    control, theta, relation = key
                    p = "all"
                passes = sum(as_bool(row["final_operational_pass"]) for row in group)
                gaps = [float(row["final_selective_gap"]) for row in group]
                ceiling = [
                    float(row["oracle_gap_fraction_recovered"])
                    if control == "positive" and row.get("oracle_gap_fraction_recovered", "")
                    else float(row["final_recovered_ceiling_fraction"])
                    for row in group
                ]
                writer.writerow({
                    "control": control,
                    "p": p,
                    "theta_degrees": theta,
                    "rank_relation": relation,
                    "n": len(group),
                    "passes": passes,
                    "pass_rate": passes / len(group),
                    "median_selective_gap": median(gaps),
                    "median_recovered_ceiling_fraction": median(ceiling),
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    write(output_dir / "synthetic_detection_summary.csv", grouped, include_p=False)
    write(output_dir / "synthetic_angle_rank_summary.csv", detailed, include_p=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive-root",
        type=Path,
        required=True,
        help="Path to the frozen calibrated_separability_2026 result namespace.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/calibrated"))
    args = parser.parse_args()
    source = args.archive_root / "synthetic_controls/evaluation/synthetic_learned_results.csv"
    if not source.is_file():
        raise SystemExit(f"missing frozen source: {source}")
    summarize_synthetic(source, args.output_dir)
    print(f"wrote synthetic summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
