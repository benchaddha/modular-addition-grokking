#!/usr/bin/env python3
"""Build seed-level tables and figures for the calibrated experiment namespace."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_calibrated_separability")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MILESTONES = ["plateau", "05", "20", "50", "80", "95", "99"]
PHASE = {
    "plateau": "plateau",
    "05": "early transition",
    "20": "early transition",
    "50": "midpoint",
    "80": "late transition",
    "95": "late transition",
    "99": "completion",
}
METHOD_COLORS = {
    "learned": "#1f77b4",
    "random": "#7f7f7f",
    "pca": "#ff7f0e",
    "fourier": "#2ca02c",
    "gradient": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--fourier-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: Dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return float("nan")
    return float(value)


def truthy(value: Any) -> bool:
    return str(value).lower() == "true"


def synthetic_rank_relation(row: Dict[str, Any]) -> str:
    rank = int(float(row["rank"]))
    true_rank = int(float(row["true_rank"]))
    if rank < true_rank:
        return "lower"
    if rank == true_rank:
        return "true"
    return "higher"


def finite(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def median(values: Iterable[float]) -> float:
    valid = finite(values)
    return statistics.median(valid) if valid else float("nan")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def control_selection_key(row: Dict[str, Any]) -> tuple:
    return (
        -f(row, "validation_selective_gap"),
        -f(row, "validation_test_damage"),
        f(row, "validation_train_damage"),
        int(float(row.get("step") or 0)),
        row.get("selected_candidate_id", ""),
    )


def build_expanded_grid_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select over standard ranks plus rank 64 using validation metrics only."""
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("comparison") in {"standard", "rank64_stress"}
            and row.get("status") == "selected"
            and row.get("milestone_label") in {"20", "50", "80"}
        ):
            grouped[(row["job_id"], row["method"])].append(row)
    output: List[Dict[str, Any]] = []
    for (_, _), candidates in sorted(grouped.items()):
        winner = sorted(candidates, key=control_selection_key)[0]
        copied = dict(winner)
        copied["source_comparison"] = winner["comparison"]
        copied["comparison"] = "expanded_rank64_grid"
        output.append(copied)
    return output


def build_learned_control_rows(
    learned: Sequence[Dict[str, Any]], controls: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    controls_by_job: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in controls:
        if row.get("comparison") == "standard" and row.get("status") == "selected":
            controls_by_job[row["job_id"]].append(row)
    result = []
    for row in learned:
        if row.get("comparison") != "standard" or row.get("status") != "selected":
            continue
        choices = sorted(controls_by_job.get(row["job_id"], []), key=control_selection_key)
        control = choices[0] if choices else None
        result.append(
            {
                "job_id": row["job_id"],
                "setting": row["setting"],
                "seed": int(row["seed"]),
                "milestone_label": row["milestone_label"],
                "learned_rank": int(float(row["rank"])),
                "learned_lambda": float(row["lambda"]),
                "learned_restart": int(float(row["restart"])),
                "learned_step": int(float(row["step"])),
                "learned_final_train_retention": f(row, "final_relative_train_retention"),
                "learned_final_test_damage": f(row, "final_test_damage"),
                "learned_final_relative_test_damage": f(row, "final_relative_test_damage"),
                "learned_final_selective_gap": f(row, "final_selective_gap"),
                "learned_final_ceiling_fraction": f(row, "final_recovered_ceiling_fraction"),
                "learned_criterion_pass_80": truthy(row.get("criterion_pass_80")),
                "learned_criterion_pass_90": truthy(row.get("criterion_pass_90")),
                "learned_criterion_pass_95": truthy(row.get("criterion_pass_95")),
                "control_status": "selected" if control else "no_feasible_control",
                "best_control_method": control.get("method") if control else None,
                "best_control_rank": int(float(control["rank"])) if control else None,
                "best_control_final_selective_gap": f(control, "final_selective_gap") if control else None,
                "best_control_final_test_damage": f(control, "final_test_damage") if control else None,
                "learned_minus_control_final_gap": (
                    f(row, "final_selective_gap") - f(control, "final_selective_gap")
                    if control
                    else None
                ),
            }
        )
    return result


def build_checkpoint_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["setting"], row["milestone_label"])].append(row)
    output = []
    for (setting, milestone), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], MILESTONES.index(item[0][1]))
    ):
        gaps = finite(f(row, "final_selective_gap") for row in group)
        output.append(
            {
                "setting": setting,
                "milestone_label": milestone,
                "phase": PHASE[milestone],
                "seed_count": len({row["seed"] for row in group}),
                "median_final_selective_gap": statistics.median(gaps),
                "min_final_selective_gap": min(gaps),
                "max_final_selective_gap": max(gaps),
                "median_final_train_retention": median(
                    f(row, "final_relative_train_retention") for row in group
                ),
                "median_final_test_damage": median(f(row, "final_test_damage") for row in group),
                "median_relative_test_damage": median(
                    f(row, "final_relative_test_damage") for row in group
                ),
                "median_recovered_ceiling_fraction": median(
                    f(row, "final_recovered_ceiling_fraction") for row in group
                ),
                "criterion_pass_80_seeds": sum(truthy(row.get("criterion_pass_80")) for row in group),
                "criterion_pass_90_seeds": sum(truthy(row.get("criterion_pass_90")) for row in group),
                "criterion_pass_95_seeds": sum(truthy(row.get("criterion_pass_95")) for row in group),
            }
        )
    return output


def build_phase_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    per_seed: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_seed[(row["setting"], int(row["seed"]), PHASE[row["milestone_label"]])].append(row)
    reduced = []
    for (setting, seed, phase), group in per_seed.items():
        reduced.append(
            {
                "setting": setting,
                "seed": seed,
                "phase": phase,
                "checkpoint_count": len(group),
                "mean_final_selective_gap": statistics.mean(
                    f(row, "final_selective_gap") for row in group
                ),
                "mean_final_train_retention": statistics.mean(
                    f(row, "final_relative_train_retention") for row in group
                ),
                "mean_final_test_damage": statistics.mean(
                    f(row, "final_test_damage") for row in group
                ),
                "criterion_pass_90_count": sum(
                    truthy(row.get("criterion_pass_90")) for row in group
                ),
            }
        )
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in reduced:
        grouped[(row["setting"], row["phase"])].append(row)
    order = ["plateau", "early transition", "midpoint", "late transition", "completion"]
    output = []
    for (setting, phase), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], order.index(item[0][1]))
    ):
        output.append(
            {
                "setting": setting,
                "phase": phase,
                "seed_count": len(group),
                "checkpoints_per_seed": json.dumps(sorted({row["checkpoint_count"] for row in group})),
                "median_seed_mean_final_selective_gap": median(
                    row["mean_final_selective_gap"] for row in group
                ),
                "min_seed_mean_final_selective_gap": min(
                    row["mean_final_selective_gap"] for row in group
                ),
                "max_seed_mean_final_selective_gap": max(
                    row["mean_final_selective_gap"] for row in group
                ),
                "median_seed_mean_train_retention": median(
                    row["mean_final_train_retention"] for row in group
                ),
                "median_seed_mean_test_damage": median(
                    row["mean_final_test_damage"] for row in group
                ),
                "criterion_pass_90_checkpoint_count": sum(
                    row["criterion_pass_90_count"] for row in group
                ),
            }
        )
    return output


def plot_trajectories(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, setting in zip(axes, ["p97_1layer", "p113_2layer"]):
        for seed in sorted({int(row["seed"]) for row in rows if row["setting"] == setting}):
            subset = {
                row["milestone_label"]: f(row, "final_selective_gap")
                for row in rows
                if row["setting"] == setting and int(row["seed"]) == seed
            }
            axis.plot(range(len(MILESTONES)), [subset[m] for m in MILESTONES], marker="o", label=str(seed))
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(MILESTONES)), MILESTONES)
        axis.set_title(setting.replace("_", " "))
        axis.set_xlabel("checkpoint milestone")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("final selective gap")
    axes[1].legend(title="seed", frameon=False)
    fig.suptitle("Validation-selected attack: per-seed final selective gap")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pareto(
    learned: Sequence[Dict[str, Any]], controls: Sequence[Dict[str, Any]], path: Path
) -> None:
    rows = [row for row in learned if row.get("comparison") == "standard" and row.get("status") == "selected"]
    rows += [row for row in controls if row.get("comparison") == "standard" and row.get("status") == "selected"]
    fig, axes = plt.subplots(2, 7, figsize=(21, 6), sharex=True, sharey=True)
    for col, milestone in enumerate(MILESTONES):
        for row_index, setting in enumerate(["p97_1layer", "p113_2layer"]):
            axis = axes[row_index, col]
            subset = [row for row in rows if row["setting"] == setting and row["milestone_label"] == milestone]
            for method in METHOD_COLORS:
                method_rows = [row for row in subset if row["method"] == method]
                if method_rows:
                    axis.scatter(
                        [f(row, "final_train_damage") for row in method_rows],
                        [f(row, "final_test_damage") for row in method_rows],
                        s=18,
                        alpha=0.75,
                        color=METHOD_COLORS[method],
                        label=method,
                    )
            axis.axline((0, 0), slope=1, color="black", linewidth=0.6, alpha=0.5)
            axis.grid(alpha=0.15)
            if row_index == 0:
                axis.set_title(milestone)
            if col == 0:
                axis.set_ylabel(f"{setting}\nfinal test damage")
            if row_index == 1:
                axis.set_xlabel("final train damage")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle("Final damage after validation selection (diagonal: equal damage)", y=1.03)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_learned_control(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for axis, setting in zip(axes, ["p97_1layer", "p113_2layer"]):
        subset = [row for row in rows if row["setting"] == setting and row["control_status"] == "selected"]
        for milestone in MILESTONES:
            group = [row for row in subset if row["milestone_label"] == milestone]
            axis.scatter(
                [row["best_control_final_selective_gap"] for row in group],
                [row["learned_final_selective_gap"] for row in group],
                label=milestone,
                s=28,
                alpha=0.8,
            )
        limits = [-0.1, 1.0]
        axis.plot(limits, limits, color="black", linewidth=0.8)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_title(setting.replace("_", " "))
        axis.set_xlabel("best validation-selected control: final gap")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("learned attack: final gap")
    axes[1].legend(title="milestone", frameon=False, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rank_comparison(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    by = {(row["job_id"], row["comparison"]): row for row in rows if row.get("status") == "selected"}
    points = []
    for (job_id, comparison), row32 in by.items():
        if comparison != "rank32_only":
            continue
        row64 = by.get((job_id, "rank64_stress"))
        if row64:
            points.append((row32, row64))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for axis, setting in zip(axes, ["p97_1layer", "p113_2layer"]):
        subset = [pair for pair in points if pair[0]["setting"] == setting]
        for milestone in ["20", "50", "80"]:
            group = [pair for pair in subset if pair[0]["milestone_label"] == milestone]
            axis.scatter(
                [f(pair[0], "final_selective_gap") for pair in group],
                [f(pair[1], "final_selective_gap") for pair in group],
                label=milestone,
                s=32,
            )
        limits = [-0.15, 1.0]
        axis.plot(limits, limits, color="black", linewidth=0.8)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_title(setting.replace("_", " "))
        axis.set_xlabel("rank-32 final gap")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("rank-64 final gap")
    axes[1].legend(title="milestone", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_criterion(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    values = []
    labels = []
    for setting in ["p97_1layer", "p113_2layer"]:
        labels.append(setting.replace("_", " "))
        values.append(
            [
                sum(
                    truthy(row.get("criterion_pass_90"))
                    for row in rows
                    if row["setting"] == setting and row["milestone_label"] == milestone
                )
                for milestone in MILESTONES
            ]
        )
    fig, axis = plt.subplots(figsize=(9, 2.8))
    image = axis.imshow(values, vmin=0, vmax=5, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(len(MILESTONES)):
            axis.text(j, i, f"{values[i][j]}/5", ha="center", va="center")
    axis.set_xticks(range(len(MILESTONES)), MILESTONES)
    axis.set_yticks(range(2), labels)
    axis.set_title("Final operational criterion passes at 90% train retention")
    fig.colorbar(image, ax=axis, label="seed count")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_fourier_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["setting"], row["milestone_label"], row["split"], row["variant"], int(row["top_k"]))].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        setting, milestone, split, variant, top_k = key
        output.append(
            {
                "setting": setting,
                "milestone_label": milestone,
                "split": split,
                "variant": variant,
                "top_k": top_k,
                "seed_count": len({row["seed"] for row in group}),
                "median_accuracy": median(float(row["accuracy"]) for row in group),
                "median_cross_entropy": median(float(row["cross_entropy"]) for row in group),
                "median_confidence": median(float(row["confidence"]) for row in group),
                "median_logit_norm": median(float(row["logit_norm"]) for row in group),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    real_dir = Path(args.real_dir)
    synthetic_dir = Path(args.synthetic_dir)
    output_root = Path(args.output_root)
    manifest = json.loads((real_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = len(manifest["expected_jobs"])
    completed = len(manifest["completed_jobs"])
    failed = len(manifest["failed_jobs"])
    plan = {"expected_real_jobs": expected, "completed_real_jobs": completed, "failed_real_jobs": failed}
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    if not args.allow_incomplete and (completed != expected or failed):
        raise RuntimeError(f"Real experiment incomplete: {plan}")

    learned = read_csv(real_dir / "selected_final_candidates.csv")
    controls = read_csv(real_dir / "control_results.csv")
    capacity = read_csv(real_dir / "capacity_control_results.csv")
    stress = read_csv(real_dir / "high_rank_stress_results.csv")
    job_summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((real_dir / "jobs").glob("*/summary.json"))
    ]
    expanded_learned = build_expanded_grid_rows(learned)
    expanded_controls = build_expanded_grid_rows(controls)
    standard = [row for row in learned if row.get("comparison") == "standard" and row.get("status") == "selected"]
    learned_control = build_learned_control_rows(learned, controls)
    checkpoint_summary = build_checkpoint_summary(standard)
    phase_summary = build_phase_summary(standard)
    criterion = [
        row
        for row in learned
        if row.get("status") == "selected" and row.get("comparison") in {"standard", "rank32_only", "rank64_stress"}
    ]
    fourier = read_csv(Path(args.fourier_csv))
    fourier_summary = build_fourier_summary(fourier)

    tables = output_root / "tables"
    write_csv(tables / "real_learned_vs_best_control.csv", learned_control)
    write_csv(tables / "real_checkpoint_summary.csv", checkpoint_summary)
    write_csv(tables / "real_phase_summary.csv", phase_summary)
    write_csv(tables / "final_criterion_pass_table.csv", criterion)
    write_csv(output_root / "high_rank_stress_tests" / "high_rank_stress_results.csv", stress)
    write_csv(
        output_root / "high_rank_stress_tests" / "expanded_grid_selected_final_candidates.csv",
        expanded_learned,
    )
    write_csv(
        output_root / "high_rank_stress_tests" / "expanded_grid_control_results.csv",
        expanded_controls,
    )
    write_csv(tables / "capacity_control_results.csv", capacity)
    write_csv(tables / "fourier_diagnostic_summary.csv", fourier_summary)

    figures = output_root / "figures"
    plot_trajectories(standard, figures / "real_transition_final_selective_gap.png")
    plot_pareto(learned, controls, figures / "real_final_damage_pareto.png")
    plot_learned_control(learned_control, figures / "real_learned_vs_matched_control.png")
    plot_rank_comparison(learned, figures / "real_rank32_vs_rank64.png")
    plot_criterion(standard, figures / "real_final_criterion_passes.png")

    oracle = read_csv(synthetic_dir / "synthetic_oracle_results.csv")
    synthetic_learned = read_csv(synthetic_dir / "synthetic_learned_results.csv")
    true_rank_90 = [
        row
        for row in synthetic_learned
        if row["control"] == "positive"
        and float(row["theta_degrees"]) == 90
        and synthetic_rank_relation(row) == "true"
    ]
    transition_standard = [
        row for row in standard if row["milestone_label"] in {"20", "50", "80"}
    ]
    max_standard = max(standard, key=lambda row: f(row, "final_selective_gap"))
    max_transition = max(
        transition_standard, key=lambda row: f(row, "final_selective_gap")
    )
    rank64_rows = [
        row
        for row in learned
        if row.get("comparison") == "rank64_stress" and row.get("status") == "selected"
    ]
    max_rank64 = max(rank64_rows, key=lambda row: f(row, "final_selective_gap"))
    max_expanded = max(
        expanded_learned, key=lambda row: f(row, "final_selective_gap")
    )
    control_failure_count = sum(
        len(summary.get("control_failures", [])) for summary in job_summaries
    )
    svd_fallback_count = sum(
        bool(diagnostic.get("fallback_used"))
        for summary in job_summaries
        for diagnostic in summary.get("control_diagnostics", [])
    )
    cache_errors = [
        float(summary["cache_diagnostics"]["max_abs_logit_error"])
        for summary in job_summaries
    ]
    run_record_count = sum(len(summary["run_records"]) for summary in job_summaries)
    capacity_both = [
        row
        for row in capacity
        if row.get("comparison") == "standard"
        and row.get("status") == "selected"
        and f(row, "final_relative_train_retention") <= 0.5
        and f(row, "final_relative_test_damage") >= 0.5
    ]

    def identity(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "setting": row["setting"],
            "seed": int(row["seed"]),
            "milestone_label": row["milestone_label"],
            "rank": int(float(row["rank"])),
            "lambda": float(row["lambda"]),
            "restart": int(float(row["restart"])),
            "step": int(float(row["step"])),
            "final_train_retention": f(row, "final_relative_train_retention"),
            "final_test_damage": f(row, "final_test_damage"),
            "final_selective_gap": f(row, "final_selective_gap"),
            "criterion_pass_90": truthy(row.get("criterion_pass_90")),
        }

    central = {
        "real_jobs": plan,
        "standard_final_rows": len(standard),
        "standard_criterion_pass_80": sum(truthy(row.get("criterion_pass_80")) for row in standard),
        "standard_criterion_pass_90": sum(truthy(row.get("criterion_pass_90")) for row in standard),
        "standard_criterion_pass_95": sum(truthy(row.get("criterion_pass_95")) for row in standard),
        "transition_20_50_80_criterion_pass_80": sum(
            truthy(row.get("criterion_pass_80"))
            for row in standard
            if row["milestone_label"] in {"20", "50", "80"}
        ),
        "transition_20_50_80_criterion_pass_90": sum(
            truthy(row.get("criterion_pass_90"))
            for row in standard
            if row["milestone_label"] in {"20", "50", "80"}
        ),
        "transition_20_50_80_criterion_pass_95": sum(
            truthy(row.get("criterion_pass_95"))
            for row in standard
            if row["milestone_label"] in {"20", "50", "80"}
        ),
        "max_standard_result": identity(max_standard),
        "max_transition_20_50_80_result": identity(max_transition),
        "rank64_criterion_pass_90": sum(
            truthy(row.get("criterion_pass_90"))
            for row in learned
            if row.get("comparison") == "rank64_stress"
        ),
        "rank64_selected_rows": len(rank64_rows),
        "rank64_no_feasible_rows": sum(
            row.get("comparison") == "rank64_stress"
            and row.get("status") == "no_feasible_candidate"
            for row in learned
        ),
        "rank64_criterion_pass_80": sum(
            truthy(row.get("criterion_pass_80")) for row in rank64_rows
        ),
        "rank64_criterion_pass_95": sum(
            truthy(row.get("criterion_pass_95")) for row in rank64_rows
        ),
        "max_rank64_result": identity(max_rank64),
        "expanded_rank64_grid_criterion_pass_90": sum(
            truthy(row.get("criterion_pass_90")) for row in expanded_learned
        ),
        "expanded_rank64_grid_selected_rank64": sum(
            int(float(row["rank"])) == 64 for row in expanded_learned
        ),
        "max_expanded_rank64_grid_result": identity(max_expanded),
        "standard_criterion_pass_identities": [
            identity(row) for row in standard if truthy(row.get("criterion_pass_90"))
        ],
        "rank64_criterion_pass_identities": [
            identity(row) for row in rank64_rows if truthy(row.get("criterion_pass_90"))
        ],
        "real_learned_run_records": run_record_count,
        "real_control_failures": control_failure_count,
        "svd_fallback_count": svd_fallback_count,
        "max_cached_prefix_logit_error": max(cache_errors) if cache_errors else None,
        "standard_capacity_damages_both": len(capacity_both),
        "standard_capacity_rows": sum(
            row.get("comparison") == "standard" and row.get("status") == "selected"
            for row in capacity
        ),
        "synthetic_gate_successes": sum(truthy(row["final_operational_pass"]) for row in true_rank_90),
        "synthetic_gate_total": len(true_rank_90),
        "synthetic_oracle_rows": len(oracle),
        "fourier_rows": len(fourier),
    }
    (tables / "calibrated_summary.json").write_text(
        json.dumps(central, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(central, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
