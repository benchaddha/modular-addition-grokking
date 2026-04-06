#!/usr/bin/env python3
"""Analyze Fourier top-5 frequency identities across paired t0 / t1e-04 bridge runs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REFERENCE_FREQS = {14, 35, 41, 42, 52}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-files",
        nargs="+",
        required=True,
        help="JSONL files containing bridge frequency importance scores.",
    )
    parser.add_argument(
        "--output-stem",
        default="frequency_identity_bridge",
        help="Stem used for metrics/report outputs.",
    )
    return parser.parse_args()


def _parse_seed(checkpoint_path: str) -> int:
    match = re.search(r"seed_(\d+)", checkpoint_path)
    if not match:
        raise ValueError(f"Could not parse seed from checkpoint path: {checkpoint_path}")
    return int(match.group(1))


def _parse_condition(checkpoint_path: str) -> str:
    match = re.search(r"/(t0|t1e-04)/", checkpoint_path)
    if match:
        return match.group(1)
    match = re.search(r"_((?:t0)|(?:t1e-04))_", checkpoint_path)
    if match:
        return match.group(1)
    raise ValueError(f"Could not parse condition from checkpoint path: {checkpoint_path}")


def _parse_stage(row: dict[str, Any]) -> str:
    threshold = row.get("checkpoint_threshold")
    if threshold is not None:
        pct = int(round(float(threshold) * 100))
        return f"testacc_{pct}"
    checkpoint_path = str(row.get("checkpoint_path", ""))
    stage_match = re.search(r"(testacc_\d+|best|final)", checkpoint_path)
    if stage_match:
        return stage_match.group(1)
    checkpoint_type = row.get("checkpoint_type")
    if checkpoint_type:
        return str(checkpoint_type)
    raise ValueError(f"Could not determine checkpoint stage for row: {row}")


def _stage_sort_key(stage: str) -> tuple[int, float, str]:
    match = re.match(r"testacc_(\d+)", stage)
    if match:
        return (0, float(match.group(1)), stage)
    if stage == "best":
        return (1, math.inf, stage)
    if stage == "final":
        return (2, math.inf, stage)
    return (3, math.inf, stage)


def _format_freqs(values: list[int] | set[int]) -> str:
    ordered = sorted(values) if isinstance(values, set) else list(values)
    return "[" + ", ".join(str(v) for v in ordered) + "]"


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        formatted = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                formatted.append(f"{value:.3f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    metrics_dir = repo_root / "results" / "metrics"
    reports_dir = repo_root / "results" / "reports"
    output_stem = args.output_stem

    score_files = [str((repo_root / path).resolve()) if not Path(path).is_absolute() else path for path in args.score_files]

    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for score_file in score_files:
        with open(score_file, "r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                checkpoint_path = row["checkpoint_path"]
                seed = _parse_seed(checkpoint_path)
                condition = _parse_condition(checkpoint_path)
                stage = _parse_stage(row)
                grouped[(seed, condition, stage)].append(row)

    top5_records: list[dict[str, Any]] = []
    for (seed, condition, stage), rows in grouped.items():
        top_rows = sorted(rows, key=lambda item: int(item["rank"]))[:5]
        top_freqs = [int(item["frequency"]) for item in top_rows]
        top_set = set(top_freqs)
        top5_records.append(
            {
                "seed": seed,
                "condition": condition,
                "stage": stage,
                "checkpoint_epoch": int(top_rows[0].get("checkpoint_epoch", 0)),
                "checkpoint_path": str(top_rows[0]["checkpoint_path"]),
                "top5_frequencies": top_freqs,
                "top5_display": _format_freqs(top_freqs),
                "top5_set": top_set,
                "reference_overlap": _format_freqs(top_set & REFERENCE_FREQS),
                "reference_overlap_count": len(top_set & REFERENCE_FREQS),
            }
        )
    top5_records.sort(key=lambda row: (row["seed"], row["condition"], _stage_sort_key(row["stage"])))

    top5_rows = [
        {
            "seed": row["seed"],
            "condition": row["condition"],
            "stage": row["stage"],
            "checkpoint_epoch": row["checkpoint_epoch"],
            "top5_frequencies": row["top5_display"],
            "reference_overlap_count": row["reference_overlap_count"],
            "reference_overlap": row["reference_overlap"],
            "checkpoint_path": row["checkpoint_path"],
        }
        for row in top5_records
    ]

    indexed = {(row["seed"], row["condition"], row["stage"]): row for row in top5_records}
    seeds = sorted({row["seed"] for row in top5_records})
    stages = sorted({row["stage"] for row in top5_records}, key=_stage_sort_key)
    conditions = sorted({row["condition"] for row in top5_records})

    cross_condition_rows: list[dict[str, Any]] = []
    seed_summary_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_jaccards = []
        changed_stages = []
        for stage in stages:
            t0 = indexed.get((seed, "t0", stage))
            t1 = indexed.get((seed, "t1e-04", stage))
            if not t0 or not t1:
                continue
            jaccard = _jaccard(t0["top5_set"], t1["top5_set"])
            seed_jaccards.append(jaccard)
            same_set = t0["top5_set"] == t1["top5_set"]
            if not same_set:
                changed_stages.append(stage)
            cross_condition_rows.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "t0_top5": t0["top5_display"],
                    "t1e_04_top5": t1["top5_display"],
                    "jaccard": jaccard,
                    "intersection": _format_freqs(t0["top5_set"] & t1["top5_set"]),
                    "union": _format_freqs(t0["top5_set"] | t1["top5_set"]),
                    "same_set": same_set,
                    "same_order": t0["top5_frequencies"] == t1["top5_frequencies"],
                    "t0_reference_overlap": t0["reference_overlap"],
                    "t1_reference_overlap": t1["reference_overlap"],
                }
            )
        seed_summary_rows.append(
            {
                "seed": seed,
                "mean_cross_condition_jaccard": sum(seed_jaccards) / len(seed_jaccards) if seed_jaccards else float("nan"),
                "min_cross_condition_jaccard": min(seed_jaccards) if seed_jaccards else float("nan"),
                "max_cross_condition_jaccard": max(seed_jaccards) if seed_jaccards else float("nan"),
                "changed_stages": "[" + ", ".join(changed_stages) + "]" if changed_stages else "[]",
                "all_stages_same_set": len(changed_stages) == 0 and bool(seed_jaccards),
            }
        )

    condition_pairwise_rows: list[dict[str, Any]] = []
    prevalence_counter: Counter[tuple[str, int]] = Counter()
    for condition in conditions:
        condition_records = [row for row in top5_records if row["condition"] == condition]
        by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in condition_records:
            by_stage[row["stage"]].append(row)
            for freq in row["top5_frequencies"]:
                prevalence_counter[(condition, freq)] += 1
        for stage, rows in sorted(by_stage.items(), key=lambda item: _stage_sort_key(item[0])):
            rows = sorted(rows, key=lambda row: row["seed"])
            for left, right in itertools.combinations(rows, 2):
                condition_pairwise_rows.append(
                    {
                        "condition": condition,
                        "stage": stage,
                        "seed_a": left["seed"],
                        "seed_b": right["seed"],
                        "jaccard": _jaccard(left["top5_set"], right["top5_set"]),
                        "intersection": _format_freqs(left["top5_set"] & right["top5_set"]),
                        "union": _format_freqs(left["top5_set"] | right["top5_set"]),
                    }
                )

    prevalence_rows: list[dict[str, Any]] = []
    for (condition, freq), appearances in sorted(prevalence_counter.items(), key=lambda item: (item[0][0], -item[1], item[0][1])):
        prevalence_rows.append(
            {
                "condition": condition,
                "frequency": freq,
                "appearances": appearances,
                "matches_reference": freq in REFERENCE_FREQS,
            }
        )

    top5_csv = metrics_dir / f"{output_stem}_top5.csv"
    cross_condition_csv = metrics_dir / f"{output_stem}_cross_condition.csv"
    seed_summary_csv = metrics_dir / f"{output_stem}_seed_summary.csv"
    pairwise_csv = metrics_dir / f"{output_stem}_condition_pairwise.csv"
    prevalence_csv = metrics_dir / f"{output_stem}_prevalence.csv"
    report_md = reports_dir / f"{output_stem}.md"

    _write_csv(
        top5_csv,
        top5_rows,
        [
            "seed",
            "condition",
            "stage",
            "checkpoint_epoch",
            "top5_frequencies",
            "reference_overlap_count",
            "reference_overlap",
            "checkpoint_path",
        ],
    )
    _write_csv(
        cross_condition_csv,
        cross_condition_rows,
        [
            "seed",
            "stage",
            "t0_top5",
            "t1e_04_top5",
            "jaccard",
            "intersection",
            "union",
            "same_set",
            "same_order",
            "t0_reference_overlap",
            "t1_reference_overlap",
        ],
    )
    _write_csv(
        seed_summary_csv,
        seed_summary_rows,
        [
            "seed",
            "mean_cross_condition_jaccard",
            "min_cross_condition_jaccard",
            "max_cross_condition_jaccard",
            "changed_stages",
            "all_stages_same_set",
        ],
    )
    _write_csv(
        pairwise_csv,
        condition_pairwise_rows,
        [
            "condition",
            "stage",
            "seed_a",
            "seed_b",
            "jaccard",
            "intersection",
            "union",
        ],
    )
    _write_csv(
        prevalence_csv,
        prevalence_rows,
        ["condition", "frequency", "appearances", "matches_reference"],
    )

    top_99_rows = [row for row in cross_condition_rows if row["stage"] == "testacc_99"]
    report = [
        "# Frequency Identity Bridge Analysis",
        "",
        "## Inputs",
        f"- Score files: {', '.join(score_files)}",
        f"- Seeds: {_format_freqs(seeds)}",
        f"- Conditions: [{', '.join(conditions)}]",
        f"- Stages: [{', '.join(stages)}]",
        f"- Reference comparison set: {_format_freqs(sorted(REFERENCE_FREQS))}",
        "",
        "## Cross-condition Summary",
        _markdown_table(
            seed_summary_rows,
            [
                ("seed", "Seed"),
                ("mean_cross_condition_jaccard", "Mean Jaccard"),
                ("min_cross_condition_jaccard", "Min Jaccard"),
                ("max_cross_condition_jaccard", "Max Jaccard"),
                ("changed_stages", "Changed Stages"),
                ("all_stages_same_set", "All Same Set"),
            ],
        ),
        "",
        "## T0 vs T=1e-4 By Stage",
        _markdown_table(
            cross_condition_rows,
            [
                ("seed", "Seed"),
                ("stage", "Stage"),
                ("t0_top5", "T=0 Top-5"),
                ("t1e_04_top5", "T=1e-4 Top-5"),
                ("jaccard", "Jaccard"),
                ("intersection", "Intersection"),
            ],
        ),
        "",
        "## 99% Checkpoint Comparison",
        _markdown_table(
            top_99_rows,
            [
                ("seed", "Seed"),
                ("t0_top5", "T=0 Top-5"),
                ("t1e_04_top5", "T=1e-4 Top-5"),
                ("jaccard", "Jaccard"),
                ("intersection", "Intersection"),
            ],
        ),
        "",
        "## Within-condition Cross-seed Jaccard",
        _markdown_table(
            condition_pairwise_rows,
            [
                ("condition", "Condition"),
                ("stage", "Stage"),
                ("seed_a", "Seed A"),
                ("seed_b", "Seed B"),
                ("jaccard", "Jaccard"),
                ("intersection", "Intersection"),
            ],
        ),
        "",
        "## Frequency Prevalence",
        _markdown_table(
            prevalence_rows[:12],
            [
                ("condition", "Condition"),
                ("frequency", "Frequency"),
                ("appearances", "Appearances"),
                ("matches_reference", "In Ref Set"),
            ],
        ),
        "",
        "## Output Files",
        f"- {top5_csv}",
        f"- {cross_condition_csv}",
        f"- {seed_summary_csv}",
        f"- {pairwise_csv}",
        f"- {prevalence_csv}",
    ]
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Top-5 summary: {top5_csv.relative_to(repo_root)}")
    print(f"Cross-condition: {cross_condition_csv.relative_to(repo_root)}")
    print(f"Seed summary: {seed_summary_csv.relative_to(repo_root)}")
    print(f"Condition pairwise: {pairwise_csv.relative_to(repo_root)}")
    print(f"Prevalence: {prevalence_csv.relative_to(repo_root)}")
    print(f"Report: {report_md.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
