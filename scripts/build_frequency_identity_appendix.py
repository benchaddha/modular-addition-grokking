#!/usr/bin/env python3
"""Build appendix-style markdown tables for frequency identity analyses."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint-summary", required=True)
    parser.add_argument("--baseline-stability", required=True)
    parser.add_argument("--bridge-cross-condition", required=True)
    parser.add_argument("--bridge-seed-summary", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def markdown_table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(key, "") for key, _ in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline_checkpoint = read_csv(args.baseline_checkpoint_summary)
    baseline_stability = read_csv(args.baseline_stability)
    bridge_cross_condition = read_csv(args.bridge_cross_condition)
    bridge_seed_summary = read_csv(args.bridge_seed_summary)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = [
        "# Appendix: Frequency Identity Tables",
        "",
        "## Table G1. Baseline Top-5 Frequency Identities By Checkpoint",
        markdown_table(
            baseline_checkpoint,
            [
                ("stage", "Stage"),
                ("seed_52", "Seed 52"),
                ("seed_53", "Seed 53"),
                ("seed_54", "Seed 54"),
                ("intersection", "Intersection"),
                ("union", "Union"),
                ("mean_pairwise_jaccard", "Mean Jaccard"),
            ],
        ),
        "",
        "## Table G2. Baseline Within-seed Frequency Stability",
        markdown_table(
            baseline_stability,
            [
                ("seed", "Seed"),
                ("stages", "Stages"),
                ("unique_top5_sets", "Unique Sets"),
                ("unique_top5_orders", "Unique Orders"),
                ("all_stage_intersection", "All-stage Intersection"),
                ("all_stage_union", "All-stage Union"),
            ],
        ),
        "",
        "## Table G3. Bridge Cross-condition Frequency Identity",
        markdown_table(
            bridge_cross_condition,
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
        "## Table G4. Bridge Seed-level Cross-condition Summary",
        markdown_table(
            bridge_seed_summary,
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
    ]
    output_path.write_text("\n".join(content) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
