"""Aggregate early-window intervention sweeps (Analysis A).

Reads the necessity/sufficiency Fourier sweeps and the head-surgery sweeps run
on the dense early-window checkpoints (seeds 52-54, test-accuracy milestones
5%-99%) and asks whether any intervention family is differentially more
damaging to test than to train early in the transition (25-70%) than late
(80-99%).

Because baselines differ across milestones (test baseline is 0.05 at the 5%
checkpoint), raw drops are not comparable across checkpoints. We report:
  - relative retention: intervened_acc / baseline_acc per split
  - retention gap: train_retention - test_retention (positive = intervention
    spares train relative to test, i.e. separability-like asymmetry)
  - a scoped separability criterion: intervened test <= 2/p while intervened
    train >= 0.9 * baseline train.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

P = 113
CHANCE_CEILING = 2.0 / P

EARLY_BAND = {5, 10, 20, 25, 30, 40, 50, 60, 70}
LATE_BAND = {80, 85, 90, 95, 99}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _milestone_from_path(checkpoint_path: str) -> int:
    match = re.search(r"testacc_(\d+)\.pt$", checkpoint_path)
    if not match:
        raise ValueError(f"Cannot parse milestone from {checkpoint_path}")
    return int(match.group(1))


def _seed_from_path(checkpoint_path: str) -> int:
    match = re.search(r"physics_seed(\d+)_", checkpoint_path)
    if not match:
        raise ValueError(f"Cannot parse seed from {checkpoint_path}")
    return int(match.group(1))


def _augment(row: Dict[str, Any], train_key: str = "train_acc", test_key: str = "test_acc") -> Dict[str, Any]:
    base_train = row["baseline_train_acc"]
    base_test = row["baseline_test_acc"]
    train_ret = row[train_key] / base_train if base_train > 0 else float("nan")
    test_ret = row[test_key] / base_test if base_test > 0 else float("nan")
    return {
        **row,
        "milestone": _milestone_from_path(row["checkpoint_path"]),
        "run_seed": _seed_from_path(row["checkpoint_path"]),
        "train_retention": train_ret,
        "test_retention": test_ret,
        "retention_gap": train_ret - test_ret,
        "scoped_separability_pass": (
            row[test_key] <= CHANCE_CEILING and row[train_key] >= 0.9 * base_train
        ),
        "reverse_pass": (
            row[train_key] <= CHANCE_CEILING and row[test_key] >= 0.9 * base_test
        ),
    }


def _fmt(value: float) -> str:
    return f"{value:+.3f}" if value == value else "  nan"


def _summarize(rows: List[Dict[str, Any]], label_key: str, out_lines: List[str]) -> None:
    grouped: Dict[Any, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row[label_key]][row["milestone"]].append(row)

    milestones = sorted({row["milestone"] for row in rows})
    header = "condition".ljust(18) + " ".join(f"{m:>7d}" for m in milestones)
    out_lines.append(header)
    out_lines.append("-" * len(header))
    for label in sorted(grouped):
        cells = []
        for m in milestones:
            group = grouped[label].get(m, [])
            if group:
                mean_gap = sum(r["retention_gap"] for r in group) / len(group)
                cells.append(f"{_fmt(mean_gap):>7s}")
            else:
                cells.append("      -")
        out_lines.append(str(label).ljust(18) + " ".join(cells))

    early = [r for r in rows if r["milestone"] in EARLY_BAND]
    late = [r for r in rows if r["milestone"] in LATE_BAND]
    for name, band in (("early(5-70)", early), ("late(80-99)", late)):
        if band:
            mean_gap = sum(r["retention_gap"] for r in band) / len(band)
            max_row = max(band, key=lambda r: r["retention_gap"])
            out_lines.append(
                f"{name}: mean retention_gap {_fmt(mean_gap)}; "
                f"max {_fmt(max_row['retention_gap'])} "
                f"({max_row.get(label_key)}, seed {max_row['run_seed']}, "
                f"milestone {max_row['milestone']}, "
                f"train {max_row['train_acc']:.3f}/{max_row['baseline_train_acc']:.3f}, "
                f"test {max_row['test_acc']:.3f}/{max_row['baseline_test_acc']:.3f})"
            )
    passes = [r for r in rows if r["scoped_separability_pass"]]
    reverse = [r for r in rows if r["reverse_pass"]]
    out_lines.append(f"scoped separability passes: {len(passes)}/{len(rows)}")
    for r in passes:
        out_lines.append(
            f"  PASS: {r.get(label_key)} seed {r['run_seed']} milestone {r['milestone']} "
            f"train {r['train_acc']:.3f} (base {r['baseline_train_acc']:.3f}) "
            f"test {r['test_acc']:.3f} (base {r['baseline_test_acc']:.3f})"
        )
    out_lines.append(f"reverse (train-killing, test-sparing) passes: {len(reverse)}/{len(rows)}")
    for r in reverse:
        out_lines.append(
            f"  REVERSE: {r.get(label_key)} seed {r['run_seed']} milestone {r['milestone']} "
            f"train {r['train_acc']:.3f} (base {r['baseline_train_acc']:.3f}) "
            f"test {r['test_acc']:.3f} (base {r['baseline_test_acc']:.3f})"
        )
    out_lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "results" / "reports" / "early_window_analysis_a.md"),
    )
    args = parser.parse_args()

    out_lines: List[str] = []
    out_lines.append("# Analysis A: differential train/test sensitivity across the transition")
    out_lines.append("")
    out_lines.append(
        "retention_gap = train_retention - test_retention (mean over seeds 52-54); "
        "positive = intervention spares train relative to test."
    )
    out_lines.append("")

    metrics_dir = PROJECT_ROOT / "results" / "metrics"

    for stem, title in (
        ("early_window_necessity", "Fourier ablate_selected (necessity)"),
        ("early_window_sufficiency", "Fourier keep_only_selected (sufficiency)"),
    ):
        path = metrics_dir / f"{stem}_runs.jsonl"
        if not path.exists():
            out_lines.append(f"## {title}: MISSING ({path})")
            out_lines.append("")
            continue
        rows = [
            _augment(row)
            for row in _read_jsonl(path)
            if row["selection_family"] != "baseline"
        ]
        out_lines.append(f"## {title}")
        out_lines.append("```")
        _summarize(rows, "frequency_set_label", out_lines)
        out_lines.append("```")
        out_lines.append("")

    surgery_rows: List[Dict[str, Any]] = []
    for seed in (52, 53, 54):
        path = metrics_dir / f"early_window_surgery_seed{seed}.jsonl"
        if not path.exists():
            out_lines.append(f"## Head surgery seed {seed}: MISSING ({path})")
            out_lines.append("")
            continue
        for row in _read_jsonl(path):
            if row.get("condition") == "baseline":
                continue
            label = f"{row['condition']}_k{row['k']}"
            surgery_rows.append({**_augment(row), "condition_label": label})
    if surgery_rows:
        out_lines.append("## Head-level ablation (surgery)")
        out_lines.append("```")
        _summarize(surgery_rows, "condition_label", out_lines)
        out_lines.append("```")
        out_lines.append("")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {output_path}")
    print("\n".join(out_lines[:80]))


if __name__ == "__main__":
    main()
