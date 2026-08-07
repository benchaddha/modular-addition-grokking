"""Aggregate the adversarial separability search (Analysis B).

Reads results/metrics/adversarial_separability_seed*.csv and reports, per
transition milestone, whether ANY method (adversarially optimized projection,
Fourier top-k, random, PCA, gradient) found a train/test-separating subspace,
plus the achievable test damage subject to a train-retention constraint, and
draws train/test retention Pareto panels.
"""
import argparse
import csv
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

P = 113
CHANCE = 1.0 / P


def _to_float(value: str) -> float:
    return float(value) if value not in ("", None) else float("nan")


def load_rows(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                milestone = int(
                    re.search(r"testacc_(\d+)\.pt$", raw["checkpoint_path"]).group(1)
                )
                seed = int(
                    re.search(r"physics_seed(\d+)_", raw["checkpoint_path"]).group(1)
                )
                clean_train = _to_float(raw["clean_train_acc_heldout"])
                clean_test = _to_float(raw["clean_test_acc_heldout"])
                int_train = _to_float(raw["int_train_acc_heldout"])
                int_test = _to_float(raw["int_test_acc_heldout"])
                row = {
                    "milestone": milestone,
                    "seed": seed,
                    "site": raw["site"],
                    "method": raw["method"],
                    "rank": int(raw["rank"]),
                    "lam": raw["lam"],
                    "clean_train": clean_train,
                    "clean_test": clean_test,
                    "int_train": int_train,
                    "int_test": int_test,
                    "train_ret": int_train / clean_train,
                    "test_ret": int_test / clean_test,
                    "separability_pass": raw["separability_pass"] == "True",
                    "scoped_pass": raw["scoped_separability_pass"] == "True",
                }
                rows.append(row)
                final_train = _to_float(raw.get("final_basis_train_acc_heldout", ""))
                if raw["method"] == "adversarial" and final_train == final_train:
                    final_test = _to_float(raw["final_basis_test_acc_heldout"])
                    rows.append(
                        {
                            **row,
                            "method": "adversarial_final",
                            "int_train": final_train,
                            "int_test": final_test,
                            "train_ret": final_train / clean_train,
                            "test_ret": final_test / clean_test,
                            "separability_pass": final_train >= 0.90
                            and final_test <= 2.0 / P,
                            "scoped_pass": final_train >= 0.9 * clean_train
                            and final_test <= 2.0 / P,
                        }
                    )
    return rows


def constrained_damage_table(rows: List[Dict[str, Any]], out: List[str]) -> None:
    """Max relative test damage among rows whose train retention >= 0.9."""
    grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["train_ret"] >= 0.9:
            grouped[row["method"]][row["milestone"]].append(1.0 - row["test_ret"])
    milestones = sorted({row["milestone"] for row in rows})
    header = "method".ljust(18) + " ".join(f"{m:>7d}" for m in milestones)
    out.append(header)
    out.append("-" * len(header))
    for method in sorted(grouped):
        cells = []
        for m in milestones:
            values = grouped[method].get(m)
            cells.append(f"{max(values):+7.3f}" if values else "      -")
        out.append(method.ljust(18) + " ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=sorted(
            glob.glob(str(PROJECT_ROOT / "results/metrics/adversarial_separability_seed*.csv"))
        ),
    )
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / "results/reports/adversarial_separability_report.md"),
    )
    parser.add_argument(
        "--figure",
        default=str(PROJECT_ROOT / "results/figures/adversarial_pareto.png"),
    )
    args = parser.parse_args()

    rows = load_rows(args.inputs)
    out: List[str] = []
    out.append("# Analysis B: adversarial separability search")
    out.append("")
    out.append(f"Inputs: {args.inputs}")
    out.append(f"Total evaluated interventions: {len(rows)}")
    out.append("")

    hard_passes = [r for r in rows if r["separability_pass"]]
    scoped_passes = [r for r in rows if r["scoped_pass"]]
    out.append(
        f"Separability passes (train>=0.90 abs, test<=2/p): {len(hard_passes)}/{len(rows)}"
    )
    out.append(
        f"Scoped passes (train>=0.9x clean, test<=2/p): {len(scoped_passes)}/{len(rows)}"
    )
    for r in scoped_passes:
        out.append(
            f"  PASS {r['method']} seed {r['seed']} ms {r['milestone']} r={r['rank']} "
            f"lam={r['lam']} train {r['int_train']:.3f}/{r['clean_train']:.3f} "
            f"test {r['int_test']:.3f}/{r['clean_test']:.3f}"
        )
    out.append("")
    out.append(
        "## Max relative test damage subject to train retention >= 0.9 (heldout)"
    )
    out.append("")
    out.append("```")
    constrained_damage_table(rows, out)
    out.append("```")
    out.append("")

    best = defaultdict(lambda: None)
    for r in rows:
        if r["method"] not in ("adversarial", "adversarial_final"):
            continue
        key = r["milestone"]
        gap = r["train_ret"] - r["test_ret"]
        if best[key] is None or gap > best[key]["gap"]:
            best[key] = {**r, "gap": gap}
    out.append("## Best adversarial retention gap (train_ret - test_ret) per milestone")
    out.append("")
    out.append("```")
    for m in sorted(best):
        r = best[m]
        out.append(
            f"ms {m:>2}: gap {r['gap']:+.3f} ({r['method']}, seed {r['seed']}, r={r['rank']}, "
            f"lam={r['lam']}) train {r['int_train']:.3f}/{r['clean_train']:.3f} "
            f"test {r['int_test']:.3f}/{r['clean_test']:.3f}"
        )
    out.append("```")
    out.append("")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        panel_milestones = [5, 25, 50, 70, 90, 99]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
        style = {
            "adversarial": dict(color="crimson", marker="o", s=22, label="adversarial (gated)"),
            "adversarial_final": dict(color="darkorange", marker="o", s=14, label="adversarial (final)"),
            "fourier_topk": dict(color="royalblue", marker="s", s=40, label="Fourier top-k"),
            "random": dict(color="gray", marker="x", s=25, label="random"),
            "pca": dict(color="green", marker="^", s=40, label="PCA"),
            "gradient": dict(color="purple", marker="D", s=30, label="gradient"),
        }
        for ax, milestone in zip(axes.flat, panel_milestones):
            sub = [r for r in rows if r["milestone"] == milestone]
            for method, kw in style.items():
                pts = [r for r in sub if r["method"] == method]
                if pts:
                    ax.scatter(
                        [r["train_ret"] for r in pts],
                        [r["test_ret"] for r in pts],
                        alpha=0.75,
                        **kw,
                    )
            ax.axhspan(0, 0.25, xmin=0.9 / 1.35, color="red", alpha=0.08)
            ax.plot([0, 1.3], [0, 1.3], color="black", lw=0.6, ls=":")
            ax.axvline(0.9, color="black", lw=0.6, ls="--")
            ax.set_title(f"milestone {milestone}% test acc")
            ax.set_xlim(-0.02, 1.35)
            ax.set_ylim(-0.02, 1.6)
        for ax in axes[-1]:
            ax.set_xlabel("train retention (intervened / clean, heldout)")
        for ax in axes[:, 0]:
            ax.set_ylabel("test retention")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=6, frameon=False,
            bbox_to_anchor=(0.5, 0.955),
        )
        fig.suptitle(
            "Train/test damage Pareto: adversarially learned vs fixed project-out subspaces "
            "(seeds 52-54, pre_unembed)",
            y=0.99,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        figure_path = Path(args.figure)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=170)
        out.append(f"Pareto figure: {figure_path}")
    except ImportError:
        out.append("matplotlib unavailable; skipped figure")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[:60]))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
