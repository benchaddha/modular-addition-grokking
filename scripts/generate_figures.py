#!/usr/bin/env python3
"""Regenerate all manuscript figures from committed compact tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "figures"

COLORS = {"p97_1layer": "#2b6cb0", "p113_2layer": "#c05621", "original": "#557a46"}
LABELS = {
    "p97_1layer": r"$p=97$, one layer",
    "p113_2layer": r"$p=113$, two layers",
    "original": r"$p=113$, one layer, minibatch",
}
MILESTONES = ["plateau", "05", "20", "50", "80", "95", "99"]


def configure() -> None:
    mpl.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })
    OUT.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)


def calibration() -> None:
    df = pd.read_csv(DATA / "calibrated/synthetic_detection_summary.csv")
    pos = df[df.control == "positive"].copy()
    pos.theta_degrees = pd.to_numeric(pos.theta_degrees)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), constrained_layout=True)
    styles = {
        "lower": ("#718096", "o", "Lower rank"),
        "true": ("#2b6cb0", "s", "True rank"),
        "higher": ("#c05621", "^", "Higher rank"),
    }
    for relation, (color, marker, label) in styles.items():
        part = pos[pos.rank_relation == relation].sort_values("theta_degrees")
        axes[0].plot(part.theta_degrees, part.pass_rate, marker=marker, color=color, label=label)
        axes[1].plot(
            part.theta_degrees,
            part.median_recovered_ceiling_fraction,
            marker=marker,
            color=color,
            label=label,
        )
    axes[0].set(xlabel=r"Synthetic angle $\theta$ (degrees)", ylabel="Operational-pass fraction", ylim=(-0.04, 1.04))
    axes[1].set(xlabel=r"Synthetic angle $\theta$ (degrees)", ylabel="Median recovered ceiling fraction", ylim=(-0.04, 1.08))
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xticks([15, 30, 45, 60, 75, 90])
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].set_title("a  Frozen evaluation detection")
    axes[1].set_title("b  Behavioral recovery")
    axes[1].text(.02, .04, "Negative control: 3/120 passes\n(all at $p=5$)", transform=axes[1].transAxes)
    save(fig, "calibration_detection_boundary.pdf")


def transition_trajectories() -> None:
    df = pd.read_csv(DATA / "calibrated/real_learned_vs_best_control.csv")
    df["x"] = df.milestone_label.astype(str).map({m: i for i, m in enumerate(MILESTONES)})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), sharey=True, constrained_layout=True)
    for ax, setting in zip(axes, ["p97_1layer", "p113_2layer"]):
        part = df[df.setting == setting].copy()
        for _, seed_rows in part.groupby("seed"):
            seed_rows = seed_rows.sort_values("x")
            ax.plot(seed_rows.x, seed_rows.learned_final_selective_gap, color=COLORS[setting], alpha=.28, lw=1)
            ax.scatter(seed_rows.x, seed_rows.learned_final_selective_gap, color=COLORS[setting], alpha=.42, s=11)
        med = part.groupby("x").learned_final_selective_gap.median().sort_index()
        ax.plot(med.index, med.values, color=COLORS[setting], lw=2.4, marker="o", ms=4, label="Seed median")
        ax.axhline(0, color="black", lw=.7)
        ax.set_title(LABELS[setting])
        ax.set_xticks(range(len(MILESTONES)), ["Plateau", "5", "20", "50", "80", "95", "99"])
        ax.set_xlabel("Held-out milestone (%)")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Final selective gap")
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, "calibrated_selective_gap_trajectories.pdf")


def damage_pareto() -> None:
    df = pd.read_csv(DATA / "calibrated/selected_final_candidates.csv")
    df = df[(df.comparison == "standard") & (df.status == "selected")]
    df = df[df.milestone_label.astype(str).isin(["20", "50", "80", "95"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), sharex=True, sharey=True, constrained_layout=True)
    marker = {"20": "o", "50": "s", "80": "^", "95": "D"}
    for ax, setting in zip(axes, ["p97_1layer", "p113_2layer"]):
        part = df[df.setting == setting]
        for milestone in ["20", "50", "80", "95"]:
            q = part[part.milestone_label.astype(str) == milestone]
            ax.scatter(q.final_train_damage, q.final_test_damage, s=28, marker=marker[milestone],
                       color=COLORS[setting], alpha=.72, label=f"{milestone}%")
        x = np.linspace(-.02, .48, 100)
        ax.plot(x, x, "--", color="#4a5568", lw=.8, label="Equal damage")
        ax.axvline(.1, color="#718096", lw=.7, ls=":")
        targets = part.final_clean_test_accuracy.to_numpy() - 2 / part.p.to_numpy()
        ax.scatter(np.zeros_like(targets), targets, marker="*", s=24, color="#9b2c2c", alpha=.25)
        ax.set_title(LABELS[setting])
        ax.set_xlabel("Final train damage")
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Final held-out damage")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[:5], labels[:5], frameon=False, loc="upper left")
    save(fig, "calibrated_damage_pareto.pdf")


def learned_controls() -> None:
    df = pd.read_csv(DATA / "calibrated/real_learned_vs_best_control.csv")
    df = df[df.milestone_label.astype(str).isin(["50", "80", "95"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharex=True, sharey=True, constrained_layout=True)
    markers = {"50": "s", "80": "^", "95": "D"}
    for ax, setting in zip(axes, ["p97_1layer", "p113_2layer"]):
        part = df[df.setting == setting]
        for milestone in ["50", "80", "95"]:
            q = part[part.milestone_label.astype(str) == milestone]
            ax.scatter(q.best_control_final_selective_gap, q.learned_final_selective_gap,
                       marker=markers[milestone], color=COLORS[setting], s=30, alpha=.75, label=f"{milestone}%")
        lo, hi = -.04, .31
        ax.plot([lo, hi], [lo, hi], "--", color="#4a5568", lw=.8)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), title=LABELS[setting], xlabel="Best matched-control gap")
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Learned-search gap")
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, "learned_vs_matched_controls.pdf")


def fourier_trajectory() -> None:
    checkpoints = pd.read_csv(DATA / "fourier/checkpoint_metrics.csv")
    seeds = pd.read_csv(DATA / "fourier/seed_summaries.csv")
    norm = seeds[["setting", "seed", "e_t05", "e_t99"]]
    df = checkpoints.merge(norm, on=["setting", "seed"], how="left")
    df["tau"] = (df.epoch - df.e_t05) / (df.e_t99 - df.e_t05)
    df = df[(df.tau >= -.25) & (df.tau <= 1.15)]
    settings = ["original", "p97_1layer", "p113_2layer"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.25), sharex="col", constrained_layout=True)
    for col, setting in enumerate(settings):
        part = df[df.setting == setting]
        for seed, q in part.groupby("seed"):
            q = q.sort_values("tau")
            axes[0, col].plot(q.tau, q.participation_ratio, color=COLORS[setting], alpha=.3, lw=.8)
            axes[1, col].plot(q.tau, q.k90_test, color=COLORS[setting], alpha=.35, lw=.9)
        for row in range(2):
            axes[row, col].axvline(0, color="#718096", lw=.7, ls=":")
            axes[row, col].axvline(1, color="#718096", lw=.7, ls=":")
            axes[row, col].grid(alpha=.15)
        axes[0, col].set_title(LABELS[setting])
        axes[1, col].set_xlabel(r"Normalized transition time $\tau$")
    axes[0, 0].set_ylabel("Weight participation ratio")
    axes[1, 0].set_ylabel(r"Test $k_{90}$ (conditional)")
    save(fig, "fourier_concentration_trajectories.pdf")


def final_compactness() -> None:
    df = pd.read_csv(DATA / "fourier/seed_summaries.csv")
    settings = ["original", "p97_1layer", "p113_2layer"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    rng = np.random.default_rng(20260722)
    for i, setting in enumerate(settings):
        part = df[df.setting == setting]
        jitter = rng.uniform(-.08, .08, len(part))
        axes[0].scatter(i + jitter, part.k90_test_complete, color=COLORS[setting], s=27, alpha=.8)
        axes[1].scatter(i + jitter, part.pr_complete, color=COLORS[setting], s=27, alpha=.8)
        axes[0].plot([i-.16, i+.16], [part.k90_test_complete.median()]*2, color="black", lw=1.5)
        axes[1].plot([i-.16, i+.16], [part.pr_complete.median()]*2, color="black", lw=1.5)
    short = ["113/1\nmini", "97/1\nfull", "113/2\nfull"]
    axes[0].set(xticks=range(3), xticklabels=short, ylabel=r"Completion test $k_{90}$", title="a  Conditional Fourier sufficiency")
    axes[1].set(xticks=range(3), xticklabels=short, ylabel="Completion participation ratio", title="b  Spectral compactness")
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
    save(fig, "final_fourier_compactness.pdf")


def main() -> None:
    configure()
    calibration()
    transition_trajectories()
    damage_pareto()
    learned_controls()
    fourier_trajectory()
    final_compactness()
    print("generated 6 manuscript figures")


if __name__ == "__main__":
    main()
