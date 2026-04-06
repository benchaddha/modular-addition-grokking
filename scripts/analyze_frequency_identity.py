#!/usr/bin/env python3
"""Analyze Fourier frequency identity stability across seeds and checkpoints."""

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
        help="JSONL files containing per-frequency importance scores.",
    )
    parser.add_argument(
        "--output-stem",
        default="frequency_identity_analysis",
        help="Stem used for metrics/report outputs.",
    )
    return parser.parse_args()


def _parse_seed(checkpoint_path: str) -> int:
    match = re.search(r"seed_(\d+)", checkpoint_path)
    if not match:
        raise ValueError(f"Could not parse seed from checkpoint path: {checkpoint_path}")
    return int(match.group(1))


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


def _jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _format_freqs(freqs: list[int] | set[int]) -> str:
    if isinstance(freqs, set):
        values = sorted(freqs)
    else:
        values = list(freqs)
    return "[" + ", ".join(str(v) for v in values) + "]"


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    headers = [label for _, label in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_top5_records(score_files: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for score_file in score_files:
        with open(score_file, "r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                checkpoint_path = row["checkpoint_path"]
                seed = _parse_seed(checkpoint_path)
                stage = _parse_stage(row)
                key = (seed, stage, checkpoint_path)
                row["score_file"] = score_file
                grouped[key].append(row)

    top5_records: list[dict[str, Any]] = []
    for (seed, stage, checkpoint_path), rows in grouped.items():
        top_rows = sorted(rows, key=lambda item: int(item["rank"]))[:5]
        top_freqs = [int(item["frequency"]) for item in top_rows]
        top_set = set(top_freqs)
        top5_records.append(
            {
                "seed": seed,
                "stage": stage,
                "checkpoint_path": checkpoint_path,
                "checkpoint_epoch": int(top_rows[0].get("checkpoint_epoch", 0)),
                "checkpoint_type": str(top_rows[0].get("checkpoint_type", "")),
                "top5_frequencies": top_freqs,
                "top5_set": top_set,
                "top5_display": _format_freqs(top_freqs),
                "reference_overlap": top_set & REFERENCE_FREQS,
                "reference_overlap_count": len(top_set & REFERENCE_FREQS),
                "score_file": str(top_rows[0]["score_file"]),
            }
        )

    top5_records.sort(key=lambda item: (item["seed"], _stage_sort_key(item["stage"])))
    return top5_records


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
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
    top5_records = load_top5_records(score_files)

    top5_by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    top5_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in top5_records:
        top5_by_stage[record["stage"]].append(record)
        top5_by_seed[record["seed"]].append(record)

    top5_rows: list[dict[str, Any]] = []
    for record in top5_records:
        top5_rows.append(
            {
                "seed": record["seed"],
                "stage": record["stage"],
                "checkpoint_epoch": record["checkpoint_epoch"],
                "checkpoint_path": record["checkpoint_path"],
                "top5_frequencies": record["top5_display"],
                "reference_overlap_count": record["reference_overlap_count"],
                "reference_overlap": _format_freqs(record["reference_overlap"]),
            }
        )

    pairwise_rows: list[dict[str, Any]] = []
    checkpoint_summary_rows: list[dict[str, Any]] = []
    for stage, records in sorted(top5_by_stage.items(), key=lambda item: _stage_sort_key(item[0])):
        records = sorted(records, key=lambda item: item["seed"])
        seed_sets = {record["seed"]: record["top5_set"] for record in records}
        seed_lists = {record["seed"]: record["top5_display"] for record in records}
        seed_values = sorted(seed_sets)
        intersections = set.intersection(*(seed_sets[seed] for seed in seed_values)) if seed_values else set()
        unions = set.union(*(seed_sets[seed] for seed in seed_values)) if seed_values else set()
        pairwise_scores = []
        for seed_a, seed_b in itertools.combinations(seed_values, 2):
            set_a = seed_sets[seed_a]
            set_b = seed_sets[seed_b]
            score = _jaccard(set_a, set_b)
            pairwise_scores.append(score)
            pairwise_rows.append(
                {
                    "stage": stage,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "jaccard": score,
                    "intersection_size": len(set_a & set_b),
                    "union_size": len(set_a | set_b),
                    "intersection": _format_freqs(set_a & set_b),
                    "union": _format_freqs(set_a | set_b),
                }
            )

        checkpoint_summary_rows.append(
            {
                "stage": stage,
                "num_seeds": len(seed_values),
                "seed_52": seed_lists.get(52, ""),
                "seed_53": seed_lists.get(53, ""),
                "seed_54": seed_lists.get(54, ""),
                "intersection": _format_freqs(intersections),
                "intersection_size": len(intersections),
                "union": _format_freqs(unions),
                "union_size": len(unions),
                "mean_pairwise_jaccard": sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else float("nan"),
                "reference_overlap_with_union": _format_freqs(unions & REFERENCE_FREQS),
                "reference_overlap_count": len(unions & REFERENCE_FREQS),
            }
        )

    stability_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for seed, records in sorted(top5_by_seed.items()):
        records = sorted(records, key=lambda item: _stage_sort_key(item["stage"]))
        stage_sets = [record["top5_set"] for record in records]
        all_intersection = set.intersection(*stage_sets) if stage_sets else set()
        all_union = set.union(*stage_sets) if stage_sets else set()
        stability_rows.append(
            {
                "seed": seed,
                "stages": ", ".join(record["stage"] for record in records),
                "unique_top5_sets": len({frozenset(record["top5_set"]) for record in records}),
                "unique_top5_orders": len({tuple(record["top5_frequencies"]) for record in records}),
                "all_stage_intersection": _format_freqs(all_intersection),
                "all_stage_intersection_size": len(all_intersection),
                "all_stage_union": _format_freqs(all_union),
                "all_stage_union_size": len(all_union),
                "reference_overlap_with_union": _format_freqs(all_union & REFERENCE_FREQS),
                "reference_overlap_count": len(all_union & REFERENCE_FREQS),
            }
        )
        for first, second in zip(records, records[1:]):
            transition_rows.append(
                {
                    "seed": seed,
                    "from_stage": first["stage"],
                    "to_stage": second["stage"],
                    "jaccard": _jaccard(first["top5_set"], second["top5_set"]),
                    "same_set": first["top5_set"] == second["top5_set"],
                    "same_order": first["top5_frequencies"] == second["top5_frequencies"],
                    "left_only": _format_freqs(first["top5_set"] - second["top5_set"]),
                    "right_only": _format_freqs(second["top5_set"] - first["top5_set"]),
                }
            )

    prevalence_counter = Counter()
    seed_presence: dict[int, set[int]] = defaultdict(set)
    stage_presence: dict[int, set[str]] = defaultdict(set)
    for record in top5_records:
        seed = record["seed"]
        stage = record["stage"]
        for freq in record["top5_frequencies"]:
            prevalence_counter[freq] += 1
            seed_presence[freq].add(seed)
            stage_presence[freq].add(stage)
    prevalence_rows: list[dict[str, Any]] = []
    for freq, appearances in sorted(prevalence_counter.items(), key=lambda item: (-item[1], item[0])):
        prevalence_rows.append(
            {
                "frequency": freq,
                "appearances": appearances,
                "num_seeds": len(seed_presence[freq]),
                "seeds": _format_freqs(seed_presence[freq]),
                "num_stages": len(stage_presence[freq]),
                "stages": "[" + ", ".join(sorted(stage_presence[freq], key=_stage_sort_key)) + "]",
                "matches_reference": freq in REFERENCE_FREQS,
            }
        )

    top5_csv = metrics_dir / f"{output_stem}_top5.csv"
    checkpoint_csv = metrics_dir / f"{output_stem}_checkpoint_summary.csv"
    pairwise_csv = metrics_dir / f"{output_stem}_pairwise_jaccard.csv"
    stability_csv = metrics_dir / f"{output_stem}_stability.csv"
    transition_csv = metrics_dir / f"{output_stem}_transition_jaccard.csv"
    prevalence_csv = metrics_dir / f"{output_stem}_prevalence.csv"
    report_md = reports_dir / f"{output_stem}.md"

    write_csv(
        top5_csv,
        top5_rows,
        [
            "seed",
            "stage",
            "checkpoint_epoch",
            "checkpoint_path",
            "top5_frequencies",
            "reference_overlap_count",
            "reference_overlap",
        ],
    )
    write_csv(
        checkpoint_csv,
        checkpoint_summary_rows,
        [
            "stage",
            "num_seeds",
            "seed_52",
            "seed_53",
            "seed_54",
            "intersection",
            "intersection_size",
            "union",
            "union_size",
            "mean_pairwise_jaccard",
            "reference_overlap_with_union",
            "reference_overlap_count",
        ],
    )
    write_csv(
        pairwise_csv,
        pairwise_rows,
        [
            "stage",
            "seed_a",
            "seed_b",
            "jaccard",
            "intersection_size",
            "union_size",
            "intersection",
            "union",
        ],
    )
    write_csv(
        stability_csv,
        stability_rows,
        [
            "seed",
            "stages",
            "unique_top5_sets",
            "unique_top5_orders",
            "all_stage_intersection",
            "all_stage_intersection_size",
            "all_stage_union",
            "all_stage_union_size",
            "reference_overlap_with_union",
            "reference_overlap_count",
        ],
    )
    write_csv(
        transition_csv,
        transition_rows,
        [
            "seed",
            "from_stage",
            "to_stage",
            "jaccard",
            "same_set",
            "same_order",
            "left_only",
            "right_only",
        ],
    )
    write_csv(
        prevalence_csv,
        prevalence_rows,
        [
            "frequency",
            "appearances",
            "num_seeds",
            "seeds",
            "num_stages",
            "stages",
            "matches_reference",
        ],
    )

    total_records = len(top5_records)
    seed_list = sorted(top5_by_seed)
    stage_list = sorted(top5_by_stage, key=_stage_sort_key)
    strongest_prevalence = prevalence_rows[:10]
    global_intersection = (
        set.intersection(*(record["top5_set"] for record in top5_records)) if top5_records else set()
    )
    global_union = set.union(*(record["top5_set"] for record in top5_records)) if top5_records else set()

    interpretation_lines = []
    if not global_intersection:
        interpretation_lines.append(
            "- No frequency appears in the top 5 for every seed and checkpoint in the baseline cohort."
        )
    else:
        interpretation_lines.append(
            f"- Frequencies shared across every seed/checkpoint: {_format_freqs(global_intersection)}."
        )

    high_overlap_stages = [
        row["stage"] for row in checkpoint_summary_rows if not math.isnan(row["mean_pairwise_jaccard"]) and row["mean_pairwise_jaccard"] >= 0.5
    ]
    if high_overlap_stages:
        interpretation_lines.append(
            f"- Cross-seed agreement is high only at: {', '.join(high_overlap_stages)}."
        )
    else:
        interpretation_lines.append(
            "- Cross-seed agreement stays low at all matched checkpoints, which points toward seed-specific frequency choices rather than a single canonical top-5 set."
        )

    stable_seeds = [str(row["seed"]) for row in stability_rows if row["unique_top5_sets"] == 1]
    if stable_seeds:
        interpretation_lines.append(
            f"- Within a seed, the top-5 set is perfectly stable across available checkpoints for seeds: {', '.join(stable_seeds)}."
        )
    else:
        interpretation_lines.append(
            "- Every seed shows at least one top-5 set change across the transition checkpoints."
        )

    reference_overlap = global_union & REFERENCE_FREQS
    if reference_overlap:
        interpretation_lines.append(
            f"- The cohort-level union overlaps the reference frequency set {_format_freqs(sorted(REFERENCE_FREQS))} on {_format_freqs(reference_overlap)}."
        )
    else:
        interpretation_lines.append(
            f"- The cohort-level union does not overlap the reference frequency set {_format_freqs(sorted(REFERENCE_FREQS))}."
        )

    top_prevalence_summary = ", ".join(
        f"{row['frequency']} ({row['appearances']})" for row in strongest_prevalence[:5]
    )

    report = [
        "# Frequency Identity Analysis",
        "",
        "## Inputs",
        f"- Score files: {', '.join(score_files)}",
        f"- Seeds: {_format_freqs(seed_list)}",
        f"- Stages: [{', '.join(stage_list)}]",
        f"- Seed-stage records analyzed: {total_records}",
        f"- Reference comparison set: {_format_freqs(sorted(REFERENCE_FREQS))}",
        "",
        "## Summary",
        f"- Global top-5 union across all seed-stage records: {_format_freqs(global_union)}",
        f"- Global top-5 intersection across all seed-stage records: {_format_freqs(global_intersection)}",
        f"- Frequencies with highest prevalence: {top_prevalence_summary}",
        "",
        *interpretation_lines,
        "",
        "## Per-checkpoint Top-5 Sets",
        _markdown_table(
            checkpoint_summary_rows,
            [
                ("stage", "Stage"),
                ("num_seeds", "Seeds"),
                ("seed_52", "Seed 52"),
                ("seed_53", "Seed 53"),
                ("seed_54", "Seed 54"),
                ("intersection", "Intersection"),
                ("union", "Union"),
                ("mean_pairwise_jaccard", "Mean Jaccard"),
            ],
        ),
        "",
        "## Pairwise Jaccard By Checkpoint",
        _markdown_table(
            pairwise_rows,
            [
                ("stage", "Stage"),
                ("seed_a", "Seed A"),
                ("seed_b", "Seed B"),
                ("jaccard", "Jaccard"),
                ("intersection", "Intersection"),
                ("union", "Union"),
            ],
        ),
        "",
        "## Within-seed Stability Across Transition",
        _markdown_table(
            stability_rows,
            [
                ("seed", "Seed"),
                ("stages", "Stages"),
                ("unique_top5_sets", "Unique Sets"),
                ("unique_top5_orders", "Unique Orders"),
                ("all_stage_intersection", "All-stage Intersection"),
                ("all_stage_union", "All-stage Union"),
                ("reference_overlap_with_union", "Ref Overlap"),
            ],
        ),
        "",
        "## Consecutive Transition Jaccard",
        _markdown_table(
            transition_rows,
            [
                ("seed", "Seed"),
                ("from_stage", "From"),
                ("to_stage", "To"),
                ("jaccard", "Jaccard"),
                ("same_set", "Same Set"),
                ("same_order", "Same Order"),
                ("left_only", "Dropped"),
                ("right_only", "Added"),
            ],
        ),
        "",
        "## Frequency Prevalence",
        _markdown_table(
            strongest_prevalence,
            [
                ("frequency", "Frequency"),
                ("appearances", "Appearances"),
                ("seeds", "Seeds"),
                ("stages", "Stages"),
                ("matches_reference", "In Ref Set"),
            ],
        ),
        "",
        "## Output Files",
        f"- {top5_csv}",
        f"- {checkpoint_csv}",
        f"- {pairwise_csv}",
        f"- {stability_csv}",
        f"- {transition_csv}",
        f"- {prevalence_csv}",
    ]
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Top-5 summary: {top5_csv.relative_to(repo_root)}")
    print(f"Checkpoint summary: {checkpoint_csv.relative_to(repo_root)}")
    print(f"Pairwise Jaccard: {pairwise_csv.relative_to(repo_root)}")
    print(f"Within-seed stability: {stability_csv.relative_to(repo_root)}")
    print(f"Transition Jaccard: {transition_csv.relative_to(repo_root)}")
    print(f"Prevalence: {prevalence_csv.relative_to(repo_root)}")
    print(f"Report: {report_md.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
