#!/usr/bin/env python3
"""Generate manuscript LaTeX tables from committed machine-readable summaries."""

from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/calibrated"
OUT = ROOT / "generated"

LABEL = {"p97_1layer": "$97$/1", "p113_2layer": "$113$/2"}
ORDER = {"plateau": 0, "05": 1, "20": 2, "50": 3, "80": 4, "95": 5, "99": 6}


def read(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: str, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def milestone_table() -> None:
    rows = read("real_checkpoint_summary.csv")
    rows.sort(key=lambda r: (0 if r["setting"] == "p97_1layer" else 1, ORDER[r["milestone_label"]]))
    lines = [
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Setting & Milestone & Median gap & Seed range & Median retention & Passes \\", r"\midrule",
    ]
    for row in rows:
        milestone = "Plateau" if row["milestone_label"] == "plateau" else row["milestone_label"] + r"\%"
        lines.append(
            f'{LABEL[row["setting"]]} & {milestone} & {f(row["median_final_selective_gap"])} & '
            f'[{f(row["min_final_selective_gap"])}, {f(row["max_final_selective_gap"])}] & '
            f'{f(row["median_final_train_retention"])} & {row["criterion_pass_90_seeds"]}/5 \\\\'
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "milestone_summary.tex").write_text("\n".join(lines) + "\n")


def per_seed_table() -> None:
    rows = read("real_learned_vs_best_control.csv")
    rows.sort(key=lambda r: (0 if r["setting"] == "p97_1layer" else 1, int(r["seed"]), ORDER[r["milestone_label"]]))
    lines = [
        r"\begin{longtable}{lllr@{\hspace{5pt}}rrrr}",
        r"\caption{Full standard-grid final results. The asterisk marks the schema-only pilot-exposed row.}\label{tab:per-seed}\\",
        r"\toprule Setting & Seed & Milestone & Rank & Gap & Retention & Test acc. & Control gap \\",
        r"\midrule\endfirsthead",
        r"\toprule Setting & Seed & Milestone & Rank & Gap & Retention & Test acc. & Control gap \\",
        r"\midrule\endhead",
        r"\bottomrule\endfoot",
    ]
    selected = {(r["job_id"]): r for r in read("selected_final_candidates.csv") if r["comparison"] == "standard" and r["status"] == "selected"}
    for row in rows:
        final = selected[row["job_id"]]
        mark = "$^{*}$" if row["job_id"] == "p97_1layer-seed100-20" else ""
        milestone = "P" if row["milestone_label"] == "plateau" else row["milestone_label"]
        lines.append(
            f'{LABEL[row["setting"]]} & {row["seed"]} & {milestone}{mark} & {row["learned_rank"]} & '
            f'{f(row["learned_final_selective_gap"])} & {f(row["learned_final_train_retention"])} & '
            f'{f(final["final_test_accuracy"])} & {f(row["best_control_final_selective_gap"])} \\\\'
        )
    lines.append(r"\end{longtable}")
    (OUT / "per_seed_calibrated.tex").write_text("\n".join(lines) + "\n")


def control_table() -> None:
    rows = [r for r in read("control_results.csv") if r["comparison"] == "standard"]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Method & Feasible/70 & Passes & Median final gap \\", r"\midrule"]
    for method in ["random", "pca", "fourier", "gradient"]:
        group = groups[method]
        feasible = [r for r in group if r["status"] == "selected"]
        passes = sum(r["criterion_pass_90"] == "True" for r in feasible)
        median = statistics.median(float(r["final_selective_gap"]) for r in feasible) if feasible else float("nan")
        label = {"pca": "Activation PCA", "fourier": "Weight-ranked Fourier", "gradient": "Loss-gradient SVD", "random": "Random subspace"}[method]
        lines.append(f"{label} & {len(feasible)}/70 & {passes} & {median:.3f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "control_by_method.tex").write_text("\n".join(lines) + "\n")


def synthetic_table() -> None:
    rows = [r for r in read("synthetic_detection_summary.csv") if r["control"] == "positive"]
    rows.sort(key=lambda r: (float(r["theta_degrees"]), {"lower": 0, "true": 1, "higher": 2}[r["rank_relation"]]))
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Angle & Search-rank relation & Passes & Median ceiling fraction \\", r"\midrule"]
    for row in rows:
        lines.append(f'{row["theta_degrees"]}$^\\circ$ & {row["rank_relation"].title()} & {row["passes"]}/{row["n"]} & {f(row["median_recovered_ceiling_fraction"])} \\\\')
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "synthetic_angle_rank.tex").write_text("\n".join(lines) + "\n")


def rank64_table() -> None:
    rank32 = [r for r in read("selected_final_candidates.csv") if r["comparison"] == "rank32_only" and r["method"] == "learned"]
    rank64 = [r for r in read("high_rank_stress_results.csv") if r["method"] == "learned"]
    lines = [r"\begin{tabular}{llrrr}", r"\toprule", r"Setting & Milestone & Rank 32 median & Rank 64 median & Rank-64 feasible \\", r"\midrule"]
    for setting in ["p97_1layer", "p113_2layer"]:
        for milestone in ["20", "50", "80"]:
            r32 = [r for r in rank32 if r["setting"] == setting and r["milestone_label"] == milestone and r["status"] == "selected"]
            r64 = [r for r in rank64 if r["setting"] == setting and r["milestone_label"] == milestone and r["status"] == "selected"]
            med32 = statistics.median(float(r["final_selective_gap"]) for r in r32)
            med64 = "---" if not r64 else f'{statistics.median(float(r["final_selective_gap"]) for r in r64):.3f}'
            lines.append(f"{LABEL[setting]} & {milestone}\\% & {med32:.3f} & {med64} & {len(r64)}/5 \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "rank64_summary.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    milestone_table()
    per_seed_table()
    control_table()
    synthetic_table()
    rank64_table()
    print("generated 5 manuscript tables")


if __name__ == "__main__":
    main()
