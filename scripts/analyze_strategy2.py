#!/usr/bin/env python3
"""Analyze the Strategy-2 non-separability extension experiments.

Strategy 2 asks whether the non-separability result (no intervention can destroy
test accuracy while sparing train accuracy) survives in *canonical* grokking
regimes. Two full-batch cells were trained:
  - p97_1layer  (p=97, 1 transformer layer, 4 heads)
  - p113_2layer (p=113, 2 transformer layers)
each with 5 seeds (100..104), full-batch AdamW (betas 0.9/0.98, wd 1.0, lr 1e-3,
frac_train 0.3), all exhibiting genuine train~1 / test~chance grokking plateaus.

This script loads the per-seed battery outputs *defensively* (any missing file
becomes a 'pending' marker, never a crash), computes per-seed headline metrics
(NEVER pooling interventions across seeds as independent observations), writes a
tidy headline-metrics CSV, prints a compact table, generates two figures, and
writes a markdown report.

It is idempotent / re-runnable: re-run once the adversarial CSVs finish.

Adds only:
  results/metrics/strategy2_headline_metrics.csv
  results/figures/strategy2_training_curves.png
  results/figures/strategy2_adversarial_pareto.png
  results/reports/strategy2_extension_report.md
"""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Keep matplotlib's font cache in a writable dir (background jobs run headless).
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_strategy2")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

CELLS = ["p97_1layer", "p113_2layer"]
SEEDS = [100, 101, 102, 103, 104]

STRATEGY2_DIR = REPO_ROOT / "results" / "strategy2"
METRICS_DIR = REPO_ROOT / "results" / "metrics"
FIGURES_DIR = REPO_ROOT / "results" / "figures"
REPORTS_DIR = REPO_ROOT / "results" / "reports"

HEADLINE_CSV = METRICS_DIR / "strategy2_headline_metrics.csv"
TRAINING_FIG = FIGURES_DIR / "strategy2_training_curves.png"
PARETO_FIG = FIGURES_DIR / "strategy2_adversarial_pareto.png"
REPORT_MD = REPORTS_DIR / "strategy2_extension_report.md"

# Learned/adversarial optimisation vs. non-adaptive baseline bases.
LEARNED_METHODS = ("learned", "adversarial")
BASELINE_METHODS = ("random", "pca", "fourier", "gradient")

# Checkpoint-class buckets keyed off the milestone the checkpoint was saved at.
# Pre-grok snapshots and testacc<=0.10 checkpoints sit on the pre-grok plateau.
CLASS_ORDER = ["plateau", "onset", "mid", "late", "other"]
CLASS_COLORS = {
    "plateau": "#6c757d",
    "onset": "#1f77b4",
    "mid": "#ff7f0e",
    "late": "#2ca02c",
    "other": "#9467bd",
}

PENDING = "pending"


# --------------------------------------------------------------------------- #
# Small defensive helpers
# --------------------------------------------------------------------------- #
def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if (isinstance(value, float) and math.isnan(value)) else float(value)
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> List[dict]:
    """Tolerate a partially written final line (job may still be running)."""
    rows: List[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # Truncated trailing line from an in-flight writer -> stop.
                    break
    except OSError:
        return []
    return rows


def _read_csv_dicts(path: Path) -> List[dict]:
    """Tolerate missing file / partial last row."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for row in reader:
                # A partially written row has a None key (short line) -> skip it.
                if None in row:
                    continue
                rows.append(row)
            return rows
    except OSError:
        return []


def _seed_dir(cell: str, seed: int) -> Path:
    return STRATEGY2_DIR / cell / f"seed_{seed}"


# Records how many exact-duplicate rows were dropped per (cell, seed) during the
# dedup pass, so the coverage/report can disclose overlapping checkpoint coverage.
_DEDUP_STATS: Dict[Tuple[str, int], int] = {}


def _dedup_key(r: dict) -> Tuple[Any, ...]:
    """Identity of an adversarial evaluation row.

    Overlapping shard coverage (e.g. a checkpoint re-run in a later part file, or
    a duplicated milestone) writes byte-identical rows. We dedup on the intended
    evaluation identity: (checkpoint, site, method, rank, lam, repeat).
    """
    return (
        r.get("checkpoint_path"),
        r.get("site"),
        r.get("method"),
        r.get("rank"),
        r.get("lam"),
        r.get("repeat"),
    )


def _read_adversarial_rows(cell: str, seed: int) -> Tuple[List[dict], List[Path]]:
    """Glob and concatenate all adversarial CSV shards for a seed.

    Covers the original plateau-first run (adversarial_separability.csv), the
    late-first relaunch (adversarial_separability_part2.csv), and the
    --skip-baselines relaunches (part3/part4) used where the PCA baseline crashed
    on ill-conditioned plateau/post_embed activations. Overlapping shards can
    re-emit byte-identical rows; we dedup on evaluation identity, keeping the
    first occurrence (all observed duplicates have identical metrics). Returns
    the deduped rows and the sorted list of source files (empty if none).
    """
    battery_dir = _seed_dir(cell, seed) / "battery"
    files = sorted(battery_dir.glob("adversarial_separability*.csv"))
    rows: List[dict] = []
    seen: set = set()
    n_dupes = 0
    for fp in files:
        for r in _read_csv_dicts(fp):
            key = _dedup_key(r)
            if key in seen:
                n_dupes += 1
                continue
            seen.add(key)
            rows.append(r)
    _DEDUP_STATS[(cell, seed)] = n_dupes
    return rows, files


def _find_summary_csv(cell: str, seed: int, kind: str) -> Optional[Path]:
    """kind in {'necessity','sufficiency'}. Handle the seed-100 no-suffix note."""
    candidates = [
        METRICS_DIR / f"strategy2_{cell}_seed{seed}_{kind}_summary.csv",
        METRICS_DIR / f"strategy2_{cell}_{kind}_summary.csv",  # seed-100 variant
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


# --------------------------------------------------------------------------- #
# Checkpoint-class inference
# --------------------------------------------------------------------------- #
def checkpoint_class(name: str, ckpt_type: Any, threshold: Any) -> str:
    """Bucket a checkpoint into plateau/onset/mid/late from its basename."""
    base = Path(str(name)).name
    if "epoch_" in base or "_best" in base or str(ckpt_type).lower() == "snapshot":
        return "plateau"
    thr = _to_float(threshold)
    if thr is None and "testacc_" in base:
        try:
            thr = int(base.split("testacc_")[1].split(".")[0].split("_")[0]) / 100.0
        except (ValueError, IndexError):
            thr = None
    if thr is None:
        if "trainacc" in base or "final" in base:
            return "late"
        return "other"
    if thr <= 0.10:
        return "plateau"
    if thr <= 0.40:
        return "onset"
    if thr <= 0.70:
        return "mid"
    return "late"


# Explicit per-milestone label for the coverage matrix (finer than the
# plateau/onset/mid/late bucket used for the Pareto analysis).
COVERAGE_COLS = ["plateau", "t05", "t50", "t80", "t95", "t99"]
SITE_ABBR = {"pre_unembed": "U", "post_embed": "E"}


def checkpoint_milestone(name: str, ckpt_type: Any, threshold: Any) -> str:
    """Named milestone (plateau / t05 / t50 / t80 / t95 / t99 / t<pct>) for a row."""
    base = Path(str(name)).name
    if "epoch_" in base or "_best" in base or str(ckpt_type).lower() == "snapshot":
        return "plateau"
    if "testacc_" in base:
        try:
            pct = int(base.split("testacc_")[1].split(".")[0].split("_")[0])
            return f"t{pct:02d}"
        except (ValueError, IndexError):
            pass
    thr = _to_float(threshold)
    if thr is not None:
        return f"t{int(round(thr * 100)):02d}"
    return "other"


def coverage_block(cell: str, seed: int) -> Dict[str, Any]:
    """Per-seed adversarial coverage: row count and sites per checkpoint milestone."""
    rows, files = _read_adversarial_rows(cell, seed)
    if not files:
        return {"status": PENDING, "per_milestone": {}, "n_dupes_dropped": 0}
    per: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        ms = checkpoint_milestone(
            r.get("checkpoint_path", ""),
            r.get("checkpoint_type"),
            r.get("checkpoint_threshold"),
        )
        site = str(r.get("site", ""))
        d = per.setdefault(ms, {"n": 0, "sites": set()})
        d["n"] += 1
        if site:
            d["sites"].add(site)
    for d in per.values():
        d["sites"] = sorted(d["sites"])
    return {
        "status": "ok",
        "per_milestone": per,
        "n_dupes_dropped": _DEDUP_STATS.get((cell, seed), 0),
    }


# --------------------------------------------------------------------------- #
# Metric blocks (all strictly per-seed)
# --------------------------------------------------------------------------- #
def adversarial_block(cell: str, seed: int) -> Dict[str, Any]:
    """Block (a): separability from the adversarial CSV, train-preserving rows."""
    raw, files = _read_adversarial_rows(cell, seed)
    if not files:
        return {"status": PENDING, "rows_total": 0, "rows_analyzed": 0}

    rows_total = len(raw)

    kept: List[dict] = []
    for r in raw:
        clean_train = _to_float(r.get("clean_train_acc_heldout"))
        int_train = _to_float(r.get("int_train_acc_heldout"))
        clean_test = _to_float(r.get("clean_test_acc_heldout"))
        int_test = _to_float(r.get("int_test_acc_heldout"))
        if None in (clean_train, int_train, clean_test, int_test):
            continue
        # Train-preserving interventions only.
        if int_train < 0.9 * clean_train:
            continue
        method = str(r.get("method", "")).lower()
        is_learned = any(tok in method for tok in LEARNED_METHODS)
        is_baseline = any(tok in method for tok in BASELINE_METHODS)
        group = "learned" if is_learned else ("baseline" if is_baseline else "other")
        train_drop = _to_float(r.get("train_drop"))
        test_drop = _to_float(r.get("test_drop"))
        if train_drop is None:
            train_drop = clean_train - int_train
        if test_drop is None:
            test_drop = clean_test - int_test
        sel_gap = _to_float(r.get("selective_gap"))
        if sel_gap is None:
            sel_gap = test_drop - train_drop
        rel_test_damage = (clean_test - int_test) / clean_test if clean_test > 1e-9 else None
        kept.append(
            {
                "group": group,
                "cls": checkpoint_class(
                    r.get("checkpoint_path", ""),
                    r.get("checkpoint_type"),
                    r.get("checkpoint_threshold"),
                ),
                "train_drop": train_drop,
                "test_drop": test_drop,
                "selective_gap": sel_gap,
                "above_diag": test_drop - train_drop,
                "rel_test_damage": rel_test_damage,
                "sep_pass": _to_bool(r.get("separability_pass")),
                "scoped_sep_pass": _to_bool(r.get("scoped_separability_pass")),
            }
        )

    def _agg(sub: List[dict]) -> Dict[str, Any]:
        if not sub:
            return {
                "n": 0,
                "max_rel_test_damage": None,
                "max_selective_gap": None,
                "max_above_diag": None,
                "n_sep_pass": 0,
                "n_scoped_sep_pass": 0,
            }
        rel = [x["rel_test_damage"] for x in sub if x["rel_test_damage"] is not None]
        return {
            "n": len(sub),
            "max_rel_test_damage": max(rel) if rel else None,
            "max_selective_gap": max(x["selective_gap"] for x in sub),
            "max_above_diag": max(x["above_diag"] for x in sub),
            "n_sep_pass": sum(1 for x in sub if x["sep_pass"]),
            "n_scoped_sep_pass": sum(1 for x in sub if x["scoped_sep_pass"]),
        }

    learned = [x for x in kept if x["group"] == "learned"]
    baseline = [x for x in kept if x["group"] == "baseline"]

    per_class = {}
    for cls in CLASS_ORDER:
        sub = [x for x in kept if x["cls"] == cls]
        rel = [x["rel_test_damage"] for x in sub if x["rel_test_damage"] is not None]
        if sub:
            per_class[cls] = {
                "n": len(sub),
                "max_rel_test_damage": max(rel) if rel else None,
                "max_selective_gap": max(x["selective_gap"] for x in sub),
            }

    return {
        "status": "ok" if kept else "empty",
        "rows_total": rows_total,
        "rows_analyzed": len(kept),
        "classes_present": sorted({x["cls"] for x in kept}),
        "overall": _agg(kept),
        "learned": _agg(learned),
        "baseline": _agg(baseline),
        "per_class": per_class,
    }


def _late_milestone_pareto(cell: str, seed: int) -> Dict[str, Any]:
    """Max above-diagonal (test_drop - train_drop) at testacc_95/99, train-preserving."""
    rows, files = _read_adversarial_rows(cell, seed)
    late_all: List[float] = []
    late_learned: List[float] = []
    for r in rows:
        ms = checkpoint_milestone(
            r.get("checkpoint_path", ""),
            r.get("checkpoint_type"),
            r.get("checkpoint_threshold"),
        )
        if ms not in ("t95", "t99"):
            continue
        clean_train = _to_float(r.get("clean_train_acc_heldout"))
        int_train = _to_float(r.get("int_train_acc_heldout"))
        train_drop = _to_float(r.get("train_drop"))
        test_drop = _to_float(r.get("test_drop"))
        if None in (clean_train, int_train, train_drop, test_drop):
            continue
        if int_train < 0.9 * clean_train:
            continue
        above = test_drop - train_drop
        late_all.append(above)
        method = str(r.get("method", "")).lower()
        if any(tok in method for tok in LEARNED_METHODS):
            late_learned.append(above)
    return {
        "n_late": len(late_all),
        "max_above_diag_all": max(late_all) if late_all else None,
        "max_above_diag_learned": max(late_learned) if late_learned else None,
    }


def _select_threshold_row(rows: List[dict], threshold: float) -> Optional[dict]:
    for r in rows:
        if _to_float(r.get("checkpoint_threshold")) == threshold:
            return r
    for r in rows:  # fall back to path match
        if "testacc_99" in str(r.get("checkpoint_path", "")):
            return r
    return None


def sufficiency_block(cell: str, seed: int) -> Dict[str, Any]:
    """Block (b): smallest keep-only-top-k reaching test>=0.9 and >=0.8 at testacc_99."""
    path = _find_summary_csv(cell, seed, "sufficiency")
    if path is None:
        return {"status": PENDING}
    rows = _read_csv_dicts(path)
    row = _select_threshold_row(rows, 0.99)
    if row is None:
        return {"status": PENDING}
    grid = [1, 2, 3, 5, 10]  # finite k grid, excluding "full"
    accs = {k: _to_float(row.get(f"top_{k}_test_acc")) for k in grid}
    full_acc = _to_float(row.get("top_full_test_acc"))

    def _smallest_k(target: float) -> str:
        for k in grid:
            v = accs.get(k)
            if v is not None and v >= target:
                return str(k)
        return ">max_tested"

    return {
        "status": "ok",
        "grid_accs": accs,
        "full_acc": full_acc,
        "baseline_test_acc": _to_float(row.get("baseline_test_acc")),
        "smallest_k_90": _smallest_k(0.90),
        "smallest_k_80": _smallest_k(0.80),
    }


def necessity_block(cell: str, seed: int) -> Dict[str, Any]:
    """Block (c): ablate-top-5 vs random-5 contrast at testacc_99."""
    path = _find_summary_csv(cell, seed, "necessity")
    if path is None:
        return {"status": PENDING}
    rows = _read_csv_dicts(path)
    row = _select_threshold_row(rows, 0.99)
    if row is None:
        return {"status": PENDING}
    return {
        "status": "ok",
        "top5_train": _to_float(row.get("top_5_train_acc")),
        "top5_test": _to_float(row.get("top_5_test_acc")),
        "random5_train": _to_float(row.get("random_5_mean_train_acc")),
        "random5_test": _to_float(row.get("random_5_mean_test_acc")),
        "baseline_test_acc": _to_float(row.get("baseline_test_acc")),
    }


def surgery_block(cell: str, seed: int) -> Dict[str, Any]:
    """Block (d): top-3 head ablation vs random-3 mean at testacc_99."""
    path = _seed_dir(cell, seed) / "battery" / "surgery_ablations.jsonl"
    if not path.exists():
        return {"status": PENDING}
    rows = _read_jsonl(path)
    sub = [r for r in rows if "testacc_99" in str(r.get("checkpoint_path", ""))]
    if not sub:
        return {"status": PENDING}
    n_heads = None
    baseline_test = None
    for r in sub:
        if r.get("condition") == "all_heads" and r.get("k") is not None:
            n_heads = int(r["k"])
        if r.get("condition") == "baseline":
            baseline_test = _to_float(r.get("test_acc"))
    top3 = next((r for r in sub if r.get("condition") == "top_k" and r.get("k") == 3), None)
    rand3 = [r for r in sub if r.get("condition") == "random_k" and r.get("k") == 3]

    def _mean(vals: List[Optional[float]]) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "status": "ok",
        "n_heads": n_heads,
        "baseline_test": baseline_test,
        "top3_train": _to_float(top3.get("train_acc")) if top3 else None,
        "top3_test": _to_float(top3.get("test_acc")) if top3 else None,
        "random3_train": _mean([_to_float(r.get("train_acc")) for r in rand3]),
        "random3_test": _mean([_to_float(r.get("test_acc")) for r in rand3]),
        "n_random3": len(rand3),
    }


def training_block(cell: str, seed: int) -> Dict[str, Any]:
    """Block (e): plateau characterisation from summary.json + eval jsonl."""
    summ = _read_json(_seed_dir(cell, seed) / "summary.json")
    if summ is None:
        return {"status": PENDING}
    milestones = summ.get("milestone_epochs", {}) or {}
    metrics_file = summ.get("metrics_file")
    eval_rows: List[dict] = []
    if metrics_file:
        eval_rows = _read_jsonl(REPO_ROOT / metrics_file)
    if not eval_rows:
        # Fallback to conventional artifact filename.
        alt = _seed_dir(cell, seed) / "artifacts" / f"strategy2_{cell}_seed{seed}.jsonl"
        eval_rows = _read_jsonl(alt)

    max_gap = None
    plateau_depth_test = None
    first_train99_epoch = None
    for r in eval_rows:
        tr = _to_float(r.get("train_acc"))
        te = _to_float(r.get("test_acc"))
        ep = _to_float(r.get("epoch"))
        if tr is None or te is None:
            continue
        gap = tr - te
        if max_gap is None or gap > max_gap:
            max_gap = gap
        if first_train99_epoch is None and tr >= 0.99:
            first_train99_epoch = ep
            plateau_depth_test = te

    return {
        "status": "ok",
        "grok_epoch_99": summ.get("grok_epoch_99"),
        "grok_epoch_95": summ.get("grok_epoch_95"),
        "plateau_evals": summ.get("plateau_evals"),
        "plateau_epoch_span": summ.get("plateau_epoch_span"),
        "plateau_detected": summ.get("plateau_detected"),
        "testacc_99_epoch": milestones.get("testacc_99"),
        "best_test_acc": summ.get("best_test_acc"),
        "max_train_test_gap": max_gap,
        "first_train99_epoch": first_train99_epoch,
        "plateau_depth_test": plateau_depth_test,
        "n_eval_rows": len(eval_rows),
    }


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def collect() -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for cell in CELLS:
        for seed in SEEDS:
            if not _seed_dir(cell, seed).exists():
                out[(cell, seed)] = {"exists": False}
                continue
            out[(cell, seed)] = {
                "exists": True,
                "training": training_block(cell, seed),
                "adversarial": adversarial_block(cell, seed),
                "coverage": coverage_block(cell, seed),
                "sufficiency": sufficiency_block(cell, seed),
                "necessity": necessity_block(cell, seed),
                "surgery": surgery_block(cell, seed),
            }
    return out


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_headline_csv(data: Dict[Tuple[str, int], Dict[str, Any]]) -> None:
    """Tidy long format: one row per (cell, seed, block, metric)."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    def add(cell, seed, block, status, metric, value):
        rows.append(
            {
                "cell": cell,
                "seed": seed,
                "block": block,
                "status": status,
                "metric": metric,
                "value": value,
            }
        )

    for (cell, seed), rec in sorted(data.items()):
        if not rec.get("exists"):
            add(cell, seed, "meta", "absent", "seed_dir_exists", 0)
            continue

        tr = rec["training"]
        st = tr.get("status")
        for m in (
            "grok_epoch_99",
            "plateau_evals",
            "plateau_epoch_span",
            "testacc_99_epoch",
            "best_test_acc",
            "max_train_test_gap",
            "first_train99_epoch",
            "plateau_depth_test",
        ):
            add(cell, seed, "training", st, m, tr.get(m))

        adv = rec["adversarial"]
        st = adv.get("status")
        add(cell, seed, "adversarial", st, "rows_total", adv.get("rows_total"))
        add(cell, seed, "adversarial", st, "rows_analyzed", adv.get("rows_analyzed"))
        add(cell, seed, "adversarial", st, "classes_present",
            "|".join(adv.get("classes_present", [])) or None)
        for grp in ("overall", "learned", "baseline"):
            g = adv.get(grp)
            if g:
                add(cell, seed, "adversarial", st, f"{grp}_n", g["n"])
                add(cell, seed, "adversarial", st, f"{grp}_max_rel_test_damage",
                    g["max_rel_test_damage"])
                add(cell, seed, "adversarial", st, f"{grp}_max_selective_gap",
                    g["max_selective_gap"])
                add(cell, seed, "adversarial", st, f"{grp}_max_above_diag",
                    g["max_above_diag"])
                add(cell, seed, "adversarial", st, f"{grp}_n_sep_pass", g["n_sep_pass"])
                add(cell, seed, "adversarial", st, f"{grp}_n_scoped_sep_pass",
                    g["n_scoped_sep_pass"])
        for cls, cg in (adv.get("per_class") or {}).items():
            add(cell, seed, "adversarial", st, f"class_{cls}_n", cg["n"])
            add(cell, seed, "adversarial", st, f"class_{cls}_max_rel_test_damage",
                cg["max_rel_test_damage"])
            add(cell, seed, "adversarial", st, f"class_{cls}_max_selective_gap",
                cg["max_selective_gap"])

        suf = rec["sufficiency"]
        st = suf.get("status")
        add(cell, seed, "sufficiency", st, "smallest_k_test90", suf.get("smallest_k_90"))
        add(cell, seed, "sufficiency", st, "smallest_k_test80", suf.get("smallest_k_80"))
        add(cell, seed, "sufficiency", st, "top_full_test_acc", suf.get("full_acc"))

        nec = rec["necessity"]
        st = nec.get("status")
        for m in ("top5_train", "top5_test", "random5_train", "random5_test"):
            add(cell, seed, "necessity", st, m, nec.get(m))

        sur = rec["surgery"]
        st = sur.get("status")
        for m in ("n_heads", "top3_train", "top3_test", "random3_train", "random3_test"):
            add(cell, seed, "surgery", st, m, sur.get(m))

    with HEADLINE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["cell", "seed", "block", "status", "metric", "value"]
        )
        writer.writeheader()
        writer.writerows(rows)


def print_table(data: Dict[Tuple[str, int], Dict[str, Any]]) -> None:
    print("\n" + "=" * 118)
    print("STRATEGY 2 -- per-seed headline metrics (interventions NOT pooled across seeds)")
    print("=" * 118)
    header = (
        f"{'cell':<13}{'seed':>4} | {'grok99':>7}{'pl.ev':>6}{'span':>6}{'gap':>6} | "
        f"{'adv_rows':>8}{'sepPass':>8}{'maxSelGap':>10} | "
        f"{'suf_k90':>8}{'suf_k80':>8} | {'nec_t5T':>8}{'nec_r5T':>8} | "
        f"{'sur_t3T':>8}{'sur_r3T':>8}"
    )
    print(header)
    print("-" * 118)
    for (cell, seed), rec in sorted(data.items()):
        if not rec.get("exists"):
            print(f"{cell:<13}{seed:>4} |  (seed directory absent)")
            continue
        tr, adv = rec["training"], rec["adversarial"]
        suf, nec, sur = rec["sufficiency"], rec["necessity"], rec["surgery"]

        adv_rows = adv.get("rows_analyzed", 0) if adv.get("status") != PENDING else "pend"
        if adv.get("status") == PENDING:
            sep = "pend"
            selgap = "pend"
        else:
            sep = adv["overall"]["n_sep_pass"] + adv["overall"]["n_scoped_sep_pass"]
            selgap = _fmt(adv["overall"]["max_selective_gap"])
        line = (
            f"{cell:<13}{seed:>4} | "
            f"{_fmt(tr.get('grok_epoch_99'),0):>7}"
            f"{_fmt(tr.get('plateau_evals'),0):>6}"
            f"{_fmt(tr.get('plateau_epoch_span'),0):>6}"
            f"{_fmt(tr.get('max_train_test_gap')):>6} | "
            f"{str(adv_rows):>8}{str(sep):>8}{str(selgap):>10} | "
            f"{_fmt(suf.get('smallest_k_90')):>8}{_fmt(suf.get('smallest_k_80')):>8} | "
            f"{_fmt(nec.get('top5_test')):>8}{_fmt(nec.get('random5_test')):>8} | "
            f"{_fmt(sur.get('top3_test')):>8}{_fmt(sur.get('random3_test')):>8}"
        )
        print(line)
    print("=" * 118)
    print("Legend: gap=max(train-test) during training; adv_rows=train-preserving rows analyzed;")
    print("        sepPass=count of separability_pass+scoped (expected 0); maxSelGap=max(test_drop-train_drop);")
    print("        suf_k90/80=smallest keep-only-top-k with test>=0.9/0.8 at testacc_99;")
    print("        nec_t5T/r5T=ablate-top-5 / random-5 test acc; sur_t3T/r3T=top-3 / random-3 head-ablation test acc.")
    print()


def print_coverage(data: Dict[Tuple[str, int], Dict[str, Any]]) -> None:
    print("=" * 118)
    print("ADVERSARIAL COVERAGE MATRIX -- rows analyzed per checkpoint milestone (U=pre_unembed, E=post_embed)")
    print("=" * 118)
    for cell in CELLS:
        milestones_seen = set()
        for seed in SEEDS:
            rec = data.get((cell, seed), {})
            cov = rec.get("coverage", {}) if rec.get("exists") else {}
            milestones_seen.update((cov.get("per_milestone") or {}).keys())
        cols = [c for c in COVERAGE_COLS if c in milestones_seen]
        cols += sorted(m for m in milestones_seen if m not in COVERAGE_COLS)
        print(f"\n{cell}")
        header = f"{'seed':>5} | " + "".join(f"{c:>12}" for c in cols) + f"{'dupes':>7}"
        print(header)
        print("-" * len(header))
        for seed in SEEDS:
            rec = data.get((cell, seed), {})
            if not rec.get("exists"):
                print(f"{seed:>5} |  (seed directory absent)")
                continue
            cov = rec.get("coverage", {})
            per = cov.get("per_milestone", {})
            cells = []
            for c in cols:
                if c in per:
                    sites = "".join(SITE_ABBR.get(s, "?") for s in per[c]["sites"])
                    cells.append(f"{per[c]['n']}({sites})")
                else:
                    cells.append("-")
            row = f"{seed:>5} | " + "".join(f"{c:>12}" for c in cells)
            row += f"{cov.get('n_dupes_dropped', 0):>7}"
            print(row)
    print("=" * 118)
    print()


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_training_curves(data: Dict[Tuple[str, int], Dict[str, Any]]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(CELLS), figsize=(14, 5.2), squeeze=False)
    cmap = plt.get_cmap("viridis")
    any_data = False
    for ci, cell in enumerate(CELLS):
        ax = axes[0][ci]
        for si, seed in enumerate(SEEDS):
            rows = _read_jsonl(
                _seed_dir(cell, seed) / "artifacts" / f"strategy2_{cell}_seed{seed}.jsonl"
            )
            if not rows:
                continue
            any_data = True
            color = cmap(si / max(1, len(SEEDS) - 1))
            epochs = [max(_to_float(r.get("epoch")) or 0.0, 1.0) for r in rows]
            train = [_to_float(r.get("train_acc")) for r in rows]
            test = [_to_float(r.get("test_acc")) for r in rows]
            ax.plot(epochs, train, "-", color=color, lw=1.4, alpha=0.85,
                    label=f"seed {seed}")
            ax.plot(epochs, test, "--", color=color, lw=1.4, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("epoch (log)")
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{cell}\n(solid=train, dashed=test)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="center left")
    fig.suptitle("Strategy 2: canonical grokking training curves (5 seeds/cell)", y=1.02)
    fig.tight_layout()
    fig.savefig(TRAINING_FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    if not any_data:
        print(f"[warn] no eval rows found; {TRAINING_FIG.name} is empty")


def make_pareto(data: Dict[Tuple[str, int], Dict[str, Any]]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(CELLS), figsize=(14, 6), squeeze=False)
    any_pts = False
    for ci, cell in enumerate(CELLS):
        ax = axes[0][ci]
        seen_labels = set()
        for seed in SEEDS:
            adv_rows, adv_files = _read_adversarial_rows(cell, seed)
            if not adv_files:
                continue
            for r in adv_rows:
                td = _to_float(r.get("train_drop"))
                pd_ = _to_float(r.get("test_drop"))
                ct = _to_float(r.get("clean_train_acc_heldout"))
                it = _to_float(r.get("int_train_acc_heldout"))
                if td is None or pd_ is None:
                    continue
                cls = checkpoint_class(
                    r.get("checkpoint_path", ""),
                    r.get("checkpoint_type"),
                    r.get("checkpoint_threshold"),
                )
                method = str(r.get("method", "")).lower()
                learned = any(tok in method for tok in LEARNED_METHODS)
                color = CLASS_COLORS.get(cls, "#333333")
                marker = "*" if learned else "o"
                size = 70 if learned else 26
                lbl = f"{cls} ({'learned' if learned else 'baseline'})"
                ax.scatter(
                    td, pd_, c=[color], marker=marker, s=size,
                    alpha=0.6, edgecolors="none",
                    label=lbl if lbl not in seen_labels else None,
                )
                seen_labels.add(lbl)
                any_pts = True
        # y = x damage diagonal
        lo, hi = -0.05, 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="y=x (equal damage)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("train_drop")
        ax.set_ylabel("test_drop")
        ax.set_title(f"{cell}\n(above diagonal = selective test damage)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Strategy 2: adversarial + baseline separability Pareto (train vs test damage)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(PARETO_FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    if not any_pts:
        print(f"[warn] no adversarial rows found; {PARETO_FIG.name} is empty")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_report(data: Dict[Tuple[str, int], Dict[str, Any]]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Strategy 2 extension: non-separability in canonical grokking regimes\n")
    lines.append(
        "This report tests whether the non-separability result -- *no intervention can "
        "destroy test accuracy while sparing train accuracy* -- survives in canonical "
        "grokking regimes, using two newly trained full-batch cells.\n"
    )
    lines.append("## Cells and regime\n")
    lines.append(
        "- **p97_1layer**: p=97, 1 transformer layer (4 attention heads).\n"
        "- **p113_2layer**: p=113, 2 transformer layers.\n"
        "- Optimizer: full-batch **AdamW**, betas (0.9, 0.98), weight decay **1.0**, "
        "lr **1e-3**, frac_train **0.3**.\n"
        "- 5 seeds each (100-104); every seed shows a genuine train~1 / test~chance "
        "grokking plateau followed by delayed generalization.\n"
        "- All interventions are analyzed **strictly per seed**; battery evaluations "
        "within a seed are correlated and are never pooled as independent replicates.\n"
    )

    # Plateau characterisation
    lines.append("## Per-cell plateau characterisation\n")
    for cell in CELLS:
        rows = []
        for seed in SEEDS:
            rec = data.get((cell, seed), {})
            if not rec.get("exists"):
                rows.append([seed, "absent", "-", "-", "-", "-", "-"])
                continue
            tr = rec["training"]
            if tr.get("status") == PENDING:
                rows.append([seed, "pending", "-", "-", "-", "-", "-"])
                continue
            rows.append([
                seed,
                _fmt(tr.get("grok_epoch_99"), 0),
                _fmt(tr.get("plateau_evals"), 0),
                _fmt(tr.get("plateau_epoch_span"), 0),
                _fmt(tr.get("max_train_test_gap")),
                _fmt(tr.get("plateau_depth_test")),
                _fmt(tr.get("best_test_acc")),
            ])
        lines.append(f"### {cell}\n")
        lines.append(_md_table(
            ["seed", "grok_epoch_99", "plateau_evals", "plateau_epoch_span",
             "max(train-test) gap", "test@first_train>=0.99 (plateau depth)", "best_test_acc"],
            rows,
        ))
        lines.append("")

    # Adversarial separability
    lines.append("## Adversarial separability (per seed, train-preserving rows only)\n")
    lines.append(
        "Train-preserving = `int_train_acc_heldout >= 0.9 * clean_train_acc_heldout`. "
        "`selective_gap = test_drop - train_drop` (distance above the y=x damage "
        "diagonal). `separability_pass`/`scoped_separability_pass` counts are expected "
        "to be **0** (non-separability holds).\n"
    )
    for cell in CELLS:
        rows = []
        for seed in SEEDS:
            rec = data.get((cell, seed), {})
            if not rec.get("exists"):
                rows.append([seed, "absent", "-", "-", "-", "-", "-", "-"])
                continue
            adv = rec["adversarial"]
            if adv.get("status") == PENDING:
                rows.append([seed, "PENDING", "-", "-", "-", "-", "-", "-"])
                continue
            o = adv["overall"]
            lrn = adv["learned"]
            rows.append([
                seed,
                f"{adv['rows_analyzed']}/{adv['rows_total']}",
                "|".join(adv.get("classes_present", [])) or "-",
                _fmt(o["max_selective_gap"]),
                _fmt(lrn["max_selective_gap"]) if lrn["n"] else "-",
                _fmt(o["max_above_diag"]),
                _fmt(o["max_rel_test_damage"]),
                f"{o['n_sep_pass']}/{o['n_scoped_sep_pass']}",
            ])
        lines.append(f"### {cell}\n")
        lines.append(_md_table(
            ["seed", "rows analyzed/total", "checkpoint classes",
             "max selective_gap (all)", "max selective_gap (learned)",
             "max above-diagonal", "max rel. test damage", "sep_pass/scoped"],
            rows,
        ))
        lines.append("")

    # Sufficiency / necessity / surgery
    lines.append("## Sufficiency, necessity, and head-surgery contrasts (at testacc_99)\n")
    for cell in CELLS:
        rows = []
        for seed in SEEDS:
            rec = data.get((cell, seed), {})
            if not rec.get("exists"):
                rows.append([seed] + ["absent"] * 8)
                continue
            suf, nec, sur = rec["sufficiency"], rec["necessity"], rec["surgery"]
            rows.append([
                seed,
                _fmt(suf.get("smallest_k_90")) if suf.get("status") != PENDING else "pend",
                _fmt(suf.get("smallest_k_80")) if suf.get("status") != PENDING else "pend",
                _fmt(nec.get("top5_test")) if nec.get("status") != PENDING else "pend",
                _fmt(nec.get("random5_test")) if nec.get("status") != PENDING else "pend",
                _fmt(sur.get("top3_train")) if sur.get("status") != PENDING else "pend",
                _fmt(sur.get("top3_test")) if sur.get("status") != PENDING else "pend",
                _fmt(sur.get("random3_train")) if sur.get("status") != PENDING else "pend",
                _fmt(sur.get("random3_test")) if sur.get("status") != PENDING else "pend",
            ])
        lines.append(f"### {cell}\n")
        lines.append(_md_table(
            ["seed", "suf k(test>=0.9)", "suf k(test>=0.8)",
             "nec ablate-top5 test", "nec random-5 test",
             "surgery top-3 train", "surgery top-3 test",
             "surgery random-3 train", "surgery random-3 test"],
            rows,
        ))
        lines.append("")

    # Coverage matrix
    lines.append("## Adversarial coverage matrix (rows analyzed per checkpoint milestone)\n")
    lines.append(
        "Each cell is the number of adversarial+baseline rows analyzed at that "
        "checkpoint (after deduping overlapping shards), with the sites covered in "
        "parentheses (`U`=pre_unembed, `E`=post_embed). `plateau` = pre-grok "
        "snapshot (epoch_002000); `t05` = testacc_05 (also on the plateau, at "
        "chance); `t50` = mid; `t80/t95/t99` = late milestones. `dupes` = "
        "exact-duplicate rows dropped during dedup.\n"
    )
    for cell in CELLS:
        rows = []
        milestones_seen = set()
        for seed in SEEDS:
            rec = data.get((cell, seed), {})
            cov = rec.get("coverage", {}) if rec.get("exists") else {}
            for ms in (cov.get("per_milestone") or {}):
                milestones_seen.add(ms)
        cols = [c for c in COVERAGE_COLS if c in milestones_seen]
        cols += sorted(m for m in milestones_seen if m not in COVERAGE_COLS)
        for seed in SEEDS:
            rec = data.get((cell, seed), {})
            if not rec.get("exists"):
                rows.append([seed] + ["absent"] * len(cols) + ["-"])
                continue
            cov = rec.get("coverage", {})
            if cov.get("status") == PENDING:
                rows.append([seed] + ["PENDING"] * len(cols) + ["-"])
                continue
            per = cov.get("per_milestone", {})
            cells = []
            for c in cols:
                if c in per:
                    sites = "".join(SITE_ABBR.get(s, "?") for s in per[c]["sites"])
                    cells.append(f"{per[c]['n']} ({sites})")
                else:
                    cells.append("-")
            rows.append([seed] + cells + [cov.get("n_dupes_dropped", 0)])
        lines.append(f"### {cell}\n")
        lines.append(_md_table(["seed"] + cols + ["dupes"], rows))
        lines.append("")

    # Late-milestone Pareto (diagonal hugging) check
    lines.append("## Late-milestone Pareto check (testacc_95 / testacc_99)\n")
    lines.append(
        "Among train-preserving rows at the two latest milestones, the maximum "
        "above-diagonal distance `max(test_drop - train_drop)` -- small/negative "
        "values mean points hug the y=x diagonal (damage hits train and test "
        "together; no selective test destruction).\n"
    )
    prows = []
    for cell in CELLS:
        for seed in SEEDS:
            rec = data.get((cell, seed), {})
            if not rec.get("exists"):
                continue
            lm = _late_milestone_pareto(cell, seed)
            prows.append([
                cell, seed,
                lm["n_late"],
                _fmt(lm["max_above_diag_all"]),
                _fmt(lm["max_above_diag_learned"]),
            ])
    lines.append(_md_table(
        ["cell", "seed", "n late-milestone train-preserving rows",
         "max above-diag (all)", "max above-diag (learned)"],
        prows,
    ))
    lines.append("")

    # Caveats
    lines.append("## Caveats\n")
    lines.append(
        "- **p97_1layer seed 103 plateau snapshot** had test = 0.064, slightly above the "
        "0.05 plateau criterion; the grokking plateau is genuine but this particular "
        "snapshot sits marginally above chance.\n"
        "- **p113_2layer seed 104 plateau snapshot** had test = 0.072, above 0.05; it is a "
        "fallback snapshot (the strict <=0.05 snapshot was unavailable) and likewise sits "
        "marginally above chance.\n"
        "- The **1-layer cell has only 4 attention heads**, so the top-3-vs-random-3 head "
        "ablation contrast is weakly powered: top-3 and random-3 draw from the same tiny "
        "pool of 4 heads and frequently coincide. Treat p97 surgery contrasts as "
        "underpowered.\n"
        "- **Mixed adversarial grids across seeds.** p97 seed 100 used the full grid "
        "(lambdas {0, 0.1, 0.3, 1, 3, 10} x 2 restarts) on its plateau + testacc_05 "
        "checkpoints and the reduced grid ({0, 0.3, 1, 3} x 1 restart) elsewhere; all "
        "other seeds use the reduced grid throughout. Cross-seed adversarial magnitudes "
        "are therefore not directly comparable; see the coverage matrix for exact per-seed "
        "coverage.\n"
        "- **PCA/random/fourier/gradient baselines are missing for the plateau x "
        "post_embed combination on p113_2layer seeds 101, 102, and 104.** The PCA baseline "
        "crashes deterministically there (torch.linalg.svd non-convergence on "
        "ill-conditioned repeated embedding activations); those plateau/post_embed rows "
        "come from --skip-baselines relaunches and contain only the adversarial (learned) "
        "method. All pre_unembed baselines exist everywhere, and p113 seeds 100/103 have "
        "complete plateau/post_embed baselines.\n"
        "- **p97 sufficiency requires a larger keep-only-top-k than p113** to recover test "
        "accuracy -- a genuine cross-cell difference in circuit compactness (the p113 "
        "2-layer circuit is more concentrated in its top Fourier modes), not a bug.\n"
        "- **Battery evaluations within a seed are correlated** (shared checkpoint, shared "
        "held-out split) and are NOT independent replicates. Headline metrics are reported "
        "strictly per seed and never pooled across evaluations.\n"
        "- **Determinism.** These batteries were run on CPU torch (deterministic), unlike "
        "earlier MPS runs which were non-deterministic; re-running reproduces these numbers "
        "exactly.\n"
    )

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = collect()
    write_headline_csv(data)
    print_table(data)
    print_coverage(data)
    make_training_curves(data)
    make_pareto(data)
    write_report(data)
    print(f"Wrote {HEADLINE_CSV.relative_to(REPO_ROOT)}")
    print(f"Wrote {TRAINING_FIG.relative_to(REPO_ROOT)}")
    print(f"Wrote {PARETO_FIG.relative_to(REPO_ROOT)}")
    print(f"Wrote {REPORT_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
