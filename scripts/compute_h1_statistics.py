"""
T4: H1 acceleration statistics script.

Produces:
  results/metrics/h1_statistics.md
  results/metrics/h1_statistics.csv
  results/metrics/rerun_variance.csv

(a) Exact Wilcoxon signed-rank (two-sided) on paired grok-epoch deltas per temperature vs T=0,
    both thresholds, from results/metrics/physics_runs.jsonl.
(b) Recompute for T=1e-4 pooling new seeds from run4/run5 summary.csv together with original 10.
(c) Bridge compression deltas: Δ = top3_test(SGLD) - top3_test(T=0) at 99% milestone,
    and min k reaching ≥0.99·baseline retention.
(d) Rerun-variance table: all (seed, temperature) instances across all physics data sources.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Exact Wilcoxon signed-rank (two-sided), no scipy dependency
# ============================================================

def _wilcoxon_signed_rank(differences: List[float]) -> Tuple[float, float]:
    """
    Exact two-sided Wilcoxon signed-rank test.
    Returns (W_stat, p_value).
    Drops zeros.  Uses exact distribution for small n.
    """
    diffs = [d for d in differences if d != 0]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0

    # Rank absolute values
    abs_diffs = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    # Handle ties with average ranks
    i = 0
    while i < n:
        j = i
        while j < n and abs(diffs[abs_diffs[j]]) == abs(diffs[abs_diffs[i]]):
            j += 1
        avg_rank = (i + 1 + j) / 2  # 1-indexed average
        for k in range(i, j):
            ranks[abs_diffs[k]] = avg_rank
        i = j

    W_plus = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    W_minus = sum(ranks[i] for i in range(n) if diffs[i] < 0)
    W_stat = min(W_plus, W_minus)

    # Exact p-value via enumeration for n <= 25, else normal approximation
    if n <= 25:
        total = 1 << n  # 2^n sign assignments
        count_leq = 0
        for mask in range(total):
            w = 0.0
            for bit in range(n):
                if mask & (1 << bit):
                    w += ranks[bit]
            if w <= W_stat or w >= (W_plus + W_minus - W_stat):
                count_leq += 1
        p_value = count_leq / total
    else:
        # Normal approximation
        import math
        mu = n * (n + 1) / 4
        sigma2 = n * (n + 1) * (2 * n + 1) / 24
        z = (W_stat - mu) / (sigma2 ** 0.5)
        # Two-tailed
        p_value = 2 * (1 - _norm_cdf(abs(z)))

    return W_stat, p_value


def _norm_cdf(x: float) -> float:
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


# ============================================================
# Data loading
# ============================================================

def load_physics_runs(path: Path) -> List[Dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_summary_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ============================================================
# (a) Wilcoxon on physics_runs.jsonl
# ============================================================

def compute_wilcoxon_from_physics_runs(
    runs: List[Dict],
    temperatures: List[float],
    thresholds: List[float],
) -> List[Dict]:
    """
    For each (temperature, threshold): compute paired deltas = grok_epoch(T) - grok_epoch(T=0),
    only for seeds where both conditions are successful, then run Wilcoxon.
    """
    results = []
    # Index by (seed, temperature, threshold)
    index: Dict[Tuple, Dict] = {}
    for row in runs:
        key = (row["seed"], row["temperature"], row["threshold"])
        index[key] = row

    seeds = sorted(set(r["seed"] for r in runs))

    for threshold in thresholds:
        for temperature in temperatures:
            if temperature == 0.0:
                continue
            deltas = []
            seed_details = []
            for seed in seeds:
                t0_row = index.get((seed, 0.0, threshold))
                tx_row = index.get((seed, temperature, threshold))
                if t0_row is None or tx_row is None:
                    continue
                if not t0_row["success"] or not tx_row["success"]:
                    continue
                d = float(tx_row["event_epoch"]) - float(t0_row["event_epoch"])
                deltas.append(d)
                seed_details.append({
                    "seed": seed,
                    "t0_epoch": t0_row["event_epoch"],
                    "tx_epoch": tx_row["event_epoch"],
                    "delta": d,
                })

            n_pairs = len(deltas)
            if n_pairs >= 2:
                W, p = _wilcoxon_signed_rank(deltas)
            else:
                W, p = None, None

            results.append({
                "source": "physics_runs.jsonl",
                "temperature": temperature,
                "threshold": threshold,
                "n_pairs": n_pairs,
                "deltas": deltas,
                "mean_delta": sum(deltas) / len(deltas) if deltas else None,
                "W_stat": W,
                "p_value": p,
                "seed_details": seed_details,
            })
    return results


# ============================================================
# (b) Extended Wilcoxon with run4/run5 new seeds
# ============================================================

def compute_extended_wilcoxon(
    original_runs: List[Dict],
    run4_rows: List[Dict],
    run5_rows: List[Dict],
    threshold: float = 0.95,
    temperature: float = 1e-4,
) -> Dict:
    """
    Pool original 10 seeds from physics_runs.jsonl with new seeds from run4/run5.
    Returns stats dict.
    """
    # Build lookup from original
    orig_index: Dict[Tuple, Optional[float]] = {}
    for row in original_runs:
        if row["temperature"] in (0.0, temperature) and row["threshold"] == threshold:
            orig_index[(row["seed"], row["temperature"])] = (
                float(row["event_epoch"]) if row["success"] else None
            )

    # New seeds from run4+run5 summary CSVs
    new_index: Dict[Tuple, Optional[float]] = {}
    grok_col = "grok_epoch_95" if threshold == 0.95 else "grok_epoch_99"
    for row in run4_rows + run5_rows:
        seed = int(row["seed"])
        temp = float(row["temperature"])
        if temp not in (0.0, temperature):
            continue
        val_str = row.get(grok_col, "")
        epoch = float(val_str) if val_str and val_str.strip() else None
        new_index[(seed, temp)] = epoch

    # Compute deltas
    all_seeds = sorted(set(s for (s, _) in list(orig_index.keys()) + list(new_index.keys())))
    deltas = []
    details = []
    for seed in all_seeds:
        # Prefer original data; supplement with new
        t0 = orig_index.get((seed, 0.0), new_index.get((seed, 0.0)))
        tx = orig_index.get((seed, temperature), new_index.get((seed, temperature)))
        if t0 is None or tx is None:
            continue
        d = tx - t0
        source = "original" if (seed, 0.0) in orig_index else "run4/5"
        deltas.append(d)
        details.append({"seed": seed, "t0_epoch": t0, "tx_epoch": tx, "delta": d, "source": source})

    n_pairs = len(deltas)
    W, p = None, None
    if n_pairs >= 2:
        W, p = _wilcoxon_signed_rank(deltas)

    return {
        "source": "pooled_original+run4+run5",
        "temperature": temperature,
        "threshold": threshold,
        "n_pairs": n_pairs,
        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
        "W_stat": W,
        "p_value": p,
        "details": details,
    }


# ============================================================
# (c) Bridge compression deltas
# ============================================================

def compute_compression_deltas(metrics_dir: Path) -> List[Dict]:
    """
    For each paired seed with 99% milestone data in the sufficiency runs,
    compute Δ = top3_test(SGLD) - top3_test(T=0) and min k reaching ≥0.99·baseline.
    """
    bridge_files = [
        ("seed6",  "fourier_sufficiency_sgld_bridge_runs.jsonl",         0.99),
        ("seed7",  "fourier_sufficiency_sgld_bridge_run3_seed7_runs.jsonl", 0.99),
        ("seed9",  "fourier_sufficiency_sgld_bridge_run3_seed9_runs.jsonl", 0.99),
    ]
    # Also check final-checkpoint bridge runs if already complete
    final_stems = {
        "seed4_final": "fourier_sufficiency_final_seed4_runs.jsonl",
        "seed6_final": "fourier_sufficiency_final_seed6_runs.jsonl",
        "seed7_final": "fourier_sufficiency_final_seed7_runs.jsonl",
        "seed8_final": "fourier_sufficiency_final_seed8_runs.jsonl",
        "seed9_final": "fourier_sufficiency_final_seed9_runs.jsonl",
    }

    def _extract_topk_acc(rows: List[Dict], checkpoint_identifier: str,
                          k: int, threshold_label: str = "0.99") -> Optional[float]:
        """Get top-k test accuracy for a checkpoint matching checkpoint_identifier at 99% milestone."""
        target_label = f"top_{k}"
        for row in rows:
            cp = row.get("checkpoint_path", "")
            th = row.get("checkpoint_threshold", None)
            if (checkpoint_identifier in cp and
                    row.get("selection_family") == "top" and
                    row.get("frequency_set_label") == target_label):
                if th is not None and abs(float(th) - 0.99) < 0.001:
                    return float(row["test_acc"])
        return None

    def _extract_baseline_acc(rows: List[Dict], checkpoint_identifier: str) -> Optional[float]:
        for row in rows:
            cp = row.get("checkpoint_path", "")
            th = row.get("checkpoint_threshold", None)
            if (checkpoint_identifier in cp and
                    row.get("selection_family") == "baseline"):
                if th is not None and abs(float(th) - 0.99) < 0.001:
                    return float(row["test_acc"])
        return None

    def _min_k_threshold(rows: List[Dict], checkpoint_identifier: str,
                         retention_fraction: float = 0.99) -> Optional[int]:
        """Min k such that top_k test_acc >= retention_fraction * baseline."""
        baseline = _extract_baseline_acc(rows, checkpoint_identifier)
        if baseline is None:
            return None
        target = retention_fraction * baseline
        for k in [1, 2, 3, 5, 10, 56]:
            acc = _extract_topk_acc(rows, checkpoint_identifier, k)
            if acc is not None and acc >= target:
                return k
        return None

    results = []

    # Milestone (99%) bridge analysis
    for label, fname, thr in bridge_files:
        fpath = metrics_dir / fname
        if not fpath.exists():
            continue
        rows = [json.loads(l) for l in fpath.open() if l.strip()]
        seed_num = int(label.replace("seed", ""))

        t0_id = f"seed{seed_num}_t0"
        t1_id = f"seed{seed_num}_t1e-04"

        t0_top3 = _extract_topk_acc(rows, t0_id, 3)
        t1_top3 = _extract_topk_acc(rows, t1_id, 3)
        t0_base = _extract_baseline_acc(rows, t0_id)
        t1_base = _extract_baseline_acc(rows, t1_id)
        t0_mink = _min_k_threshold(rows, t0_id)
        t1_mink = _min_k_threshold(rows, t1_id)
        delta = None
        if t0_top3 is not None and t1_top3 is not None:
            delta = t1_top3 - t0_top3

        results.append({
            "label": label,
            "checkpoint_type": "milestone_99",
            "seed": seed_num,
            "t0_baseline": t0_base,
            "t1_baseline": t1_base,
            "t0_top3": t0_top3,
            "t1_top3": t1_top3,
            "delta_top3": delta,
            "t0_min_k_99pct": t0_mink,
            "t1_min_k_99pct": t1_mink,
        })

    # Final-checkpoint analysis (T2 outputs)
    def _extract_final_topk(rows: List[Dict], temp_tag: str, k: int) -> Optional[float]:
        target_label = f"top_{k}"
        for row in rows:
            cp = row.get("checkpoint_path", "")
            if (temp_tag in cp and "_final" in cp and
                    row.get("selection_family") == "top" and
                    row.get("frequency_set_label") == target_label):
                return float(row["test_acc"])
        return None

    def _extract_final_baseline(rows: List[Dict], temp_tag: str) -> Optional[float]:
        for row in rows:
            cp = row.get("checkpoint_path", "")
            if (temp_tag in cp and "_final" in cp and
                    row.get("selection_family") == "baseline"):
                return float(row["test_acc"])
        return None

    def _min_k_final(rows: List[Dict], temp_tag: str, ret: float = 0.99) -> Optional[int]:
        base = _extract_final_baseline(rows, temp_tag)
        if base is None:
            return None
        target = ret * base
        for k in [1, 2, 3, 5, 10, 56]:
            acc = _extract_final_topk(rows, temp_tag, k)
            if acc is not None and acc >= target:
                return k
        return None

    for label, fname in final_stems.items():
        fpath = metrics_dir / fname
        if not fpath.exists():
            continue
        rows = [json.loads(l) for l in fpath.open() if l.strip()]
        seed_num = int(label.replace("seed", "").replace("_final", ""))

        t0_top3 = _extract_final_topk(rows, "t0", 3)
        t1_top3 = _extract_final_topk(rows, "t1e-04", 3)
        t0_base = _extract_final_baseline(rows, "t0")
        t1_base = _extract_final_baseline(rows, "t1e-04")
        t0_mink = _min_k_final(rows, "t0")
        t1_mink = _min_k_final(rows, "t1e-04")
        delta = None
        if t0_top3 is not None and t1_top3 is not None:
            delta = t1_top3 - t0_top3

        results.append({
            "label": label,
            "checkpoint_type": "final_epoch20k",
            "seed": seed_num,
            "t0_baseline": t0_base,
            "t1_baseline": t1_base,
            "t0_top3": t0_top3,
            "t1_top3": t1_top3,
            "delta_top3": delta,
            "t0_min_k_99pct": t0_mink,
            "t1_min_k_99pct": t1_mink,
        })

    return results


# ============================================================
# (d) Rerun variance table
# ============================================================

def compute_rerun_variance(
    physics_runs_path: Path,
    bridge_run_dirs: List[Tuple[str, Path]],
) -> List[Dict]:
    """
    Collect all (seed, temperature) instances across all data sources.
    Returns rows where same (seed, temperature) appears more than once.
    """
    all_instances = []

    # From physics_runs.jsonl
    if physics_runs_path.exists():
        runs = load_physics_runs(physics_runs_path)
        for row in runs:
            all_instances.append({
                "source": "physics_runs.jsonl",
                "seed": row["seed"],
                "temperature": row["temperature"],
                "threshold": row["threshold"],
                "grok_epoch": row.get("grok_epoch"),
                "event_epoch": row.get("event_epoch"),
                "success": row.get("success"),
                "censored": row.get("censored"),
            })

    # From bridge summary CSVs
    for source_label, csv_path in bridge_run_dirs:
        if not csv_path.exists():
            continue
        rows = load_summary_csv(csv_path)
        for row in rows:
            all_instances.append({
                "source": source_label,
                "seed": int(row["seed"]),
                "temperature": float(row["temperature"]),
                "threshold": 0.95,  # placeholder
                "grok_epoch": row.get("grok_epoch_95") or None,
                "event_epoch": row.get("grok_epoch_95") or None,
                "success": bool(row.get("grok_epoch_95")),
                "censored": not bool(row.get("grok_epoch_95")),
            })
            all_instances.append({
                "source": source_label,
                "seed": int(row["seed"]),
                "temperature": float(row["temperature"]),
                "threshold": 0.99,
                "grok_epoch": row.get("grok_epoch_99") or None,
                "event_epoch": row.get("grok_epoch_99") or None,
                "success": bool(row.get("grok_epoch_99")),
                "censored": not bool(row.get("grok_epoch_99")),
            })

    # Find (seed, temperature, threshold) combinations with multiple entries
    from collections import defaultdict
    groups = defaultdict(list)
    for inst in all_instances:
        key = (inst["seed"], inst["temperature"], inst["threshold"])
        groups[key].append(inst)

    variance_rows = []
    for (seed, temperature, threshold), group in sorted(groups.items()):
        if len(group) > 1:
            for inst in group:
                variance_rows.append(inst)

    return variance_rows


# ============================================================
# Formatting helpers
# ============================================================

def _fmt(v, digits=4):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


# ============================================================
# Main
# ============================================================

def main():
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    physics_runs_path = metrics_dir / "physics_runs.jsonl"
    bridge_run_dirs = [
        ("physics_checkpoint_bridge/summary.csv",
         PROJECT_ROOT / "results" / "physics_checkpoint_bridge" / "summary.csv"),
        ("physics_checkpoint_bridge_run2/summary.csv",
         PROJECT_ROOT / "results" / "physics_checkpoint_bridge_run2" / "summary.csv"),
        ("physics_checkpoint_bridge_run3/summary.csv",
         PROJECT_ROOT / "results" / "physics_checkpoint_bridge_run3" / "summary.csv"),
        ("physics_checkpoint_bridge_run4/summary.csv",
         PROJECT_ROOT / "results" / "physics_checkpoint_bridge_run4" / "summary.csv"),
        ("physics_checkpoint_bridge_run5/summary.csv",
         PROJECT_ROOT / "results" / "physics_checkpoint_bridge_run5" / "summary.csv"),
    ]

    # ---- (a) Wilcoxon on original physics_runs.jsonl ----
    print("=== (a) Wilcoxon on physics_runs.jsonl ===")
    original_runs = load_physics_runs(physics_runs_path)
    temperatures = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    thresholds = [0.95, 0.99]
    wilcoxon_results = compute_wilcoxon_from_physics_runs(
        original_runs, temperatures, thresholds
    )
    for r in wilcoxon_results:
        print(f"  T={r['temperature']:.0e} thr={r['threshold']} n={r['n_pairs']} "
              f"mean_delta={_fmt(r['mean_delta'])} W={_fmt(r['W_stat'])} p={_fmt(r['p_value'])}")

    # ---- (b) Extended Wilcoxon with run4/run5 ----
    print("\n=== (b) Extended Wilcoxon with run4/run5 ===")
    run4_rows = load_summary_csv(
        PROJECT_ROOT / "results" / "physics_checkpoint_bridge_run4" / "summary.csv"
    )
    run5_rows = load_summary_csv(
        PROJECT_ROOT / "results" / "physics_checkpoint_bridge_run5" / "summary.csv"
    )
    extended_results = {}
    for threshold in [0.95, 0.99]:
        res = compute_extended_wilcoxon(
            original_runs, run4_rows, run5_rows,
            threshold=threshold, temperature=1e-4
        )
        extended_results[threshold] = res
        print(f"  T=1e-4 thr={threshold} n={res['n_pairs']} "
              f"mean_delta={_fmt(res['mean_delta'])} W={_fmt(res['W_stat'])} p={_fmt(res['p_value'])}")
        print(f"  Note: includes original 10 seeds + {len(run4_rows)//2} run4 + {len(run5_rows)//2} run5 new seeds")

    # ---- (c) Bridge compression deltas ----
    print("\n=== (c) Bridge compression deltas ===")
    compression_rows = compute_compression_deltas(metrics_dir)
    for r in compression_rows:
        print(f"  {r['label']} ({r['checkpoint_type']}): "
              f"T=0 top3={_fmt(r['t0_top3'])} (base={_fmt(r['t0_baseline'])}) "
              f"T=1e-4 top3={_fmt(r['t1_top3'])} (base={_fmt(r['t1_baseline'])}) "
              f"Δ={_fmt(r['delta_top3'])} "
              f"min_k: T=0 {r['t0_min_k_99pct']} T=1e-4 {r['t1_min_k_99pct']}")

    # Wilcoxon on compression deltas if n>=6
    valid_deltas_milestone = [r["delta_top3"] for r in compression_rows
                              if r["checkpoint_type"] == "milestone_99" and r["delta_top3"] is not None]
    comp_wilcoxon = None
    if len(valid_deltas_milestone) >= 6:
        W, p = _wilcoxon_signed_rank(valid_deltas_milestone)
        comp_wilcoxon = {"n": len(valid_deltas_milestone), "W": W, "p": p}
        print(f"\n  Compression Wilcoxon (milestone_99, n={len(valid_deltas_milestone)}): W={W} p={p:.4f}")
    else:
        print(f"\n  Only {len(valid_deltas_milestone)} valid milestone_99 deltas — below n=6 threshold, no Wilcoxon")

    # ---- (d) Rerun variance ----
    print("\n=== (d) Rerun variance ===")
    variance_rows = compute_rerun_variance(physics_runs_path, bridge_run_dirs)
    print(f"  Found {len(variance_rows)} rows with repeated (seed, temperature) combinations")

    # ---- Write outputs ----
    out_md = metrics_dir / "h1_statistics.md"
    out_csv = metrics_dir / "h1_statistics.csv"
    rerun_csv = metrics_dir / "rerun_variance.csv"

    # --- Write rerun variance CSV ---
    if variance_rows:
        rerun_fields = ["source", "seed", "temperature", "threshold", "grok_epoch", "event_epoch", "success", "censored"]
        with rerun_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rerun_fields)
            writer.writeheader()
            for row in variance_rows:
                writer.writerow({k: row.get(k) for k in rerun_fields})
        print(f"  Wrote {rerun_csv}")

    # --- Write h1_statistics CSV ---
    stats_csv_rows = []
    for r in wilcoxon_results:
        stats_csv_rows.append({
            "analysis": "a_original",
            "temperature": r["temperature"],
            "threshold": r["threshold"],
            "n_pairs": r["n_pairs"],
            "mean_delta": r["mean_delta"],
            "W_stat": r["W_stat"],
            "p_value": r["p_value"],
            "note": "physics_runs.jsonl original 10 seeds",
        })
    for threshold, res in extended_results.items():
        stats_csv_rows.append({
            "analysis": "b_extended",
            "temperature": res["temperature"],
            "threshold": threshold,
            "n_pairs": res["n_pairs"],
            "mean_delta": res["mean_delta"],
            "W_stat": res["W_stat"],
            "p_value": res["p_value"],
            "note": f"original 10 + {len(run4_rows)//2} run4 + {len(run5_rows)//2} run5 (separate batch)",
        })

    if stats_csv_rows:
        fields = list(stats_csv_rows[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(stats_csv_rows)
        print(f"  Wrote {out_csv}")

    # --- Write Markdown report ---
    lines = ["# H1 Acceleration Statistics\n"]
    lines.append("## (a) Wilcoxon on original 10-seed cohort (physics_runs.jsonl)\n")
    lines.append("| Temperature | Threshold | n_pairs | mean_delta | W_stat | p_value |")
    lines.append("|-------------|-----------|---------|------------|--------|---------|")
    for r in wilcoxon_results:
        lines.append(f"| {r['temperature']:.1e} | {r['threshold']} | {r['n_pairs']} | "
                     f"{_fmt(r['mean_delta'])} | {_fmt(r['W_stat'])} | {_fmt(r['p_value'])} |")

    lines.append("")
    lines.append("### Per-seed deltas for T=1e-4\n")
    t1e4_095 = next((r for r in wilcoxon_results if r["temperature"] == 1e-4 and r["threshold"] == 0.95), None)
    t1e4_099 = next((r for r in wilcoxon_results if r["temperature"] == 1e-4 and r["threshold"] == 0.99), None)
    if t1e4_095 and t1e4_095["seed_details"]:
        lines.append("**Threshold 0.95:**")
        lines.append("| seed | T=0 epoch | T=1e-4 epoch | delta |")
        lines.append("|------|-----------|--------------|-------|")
        for d in t1e4_095["seed_details"]:
            lines.append(f"| {d['seed']} | {d['t0_epoch']} | {d['tx_epoch']} | {d['delta']:+.0f} |")
    if t1e4_099 and t1e4_099["seed_details"]:
        lines.append("\n**Threshold 0.99:**")
        lines.append("| seed | T=0 epoch | T=1e-4 epoch | delta |")
        lines.append("|------|-----------|--------------|-------|")
        for d in t1e4_099["seed_details"]:
            lines.append(f"| {d['seed']} | {d['t0_epoch']} | {d['tx_epoch']} | {d['delta']:+.0f} |")

    lines.append("")
    lines.append("## (b) Extended Wilcoxon with run4 + run5 new seeds\n")
    lines.append("_Note: new runs are a separate execution batch (different session/hardware state)._\n")
    lines.append("| Temperature | Threshold | n_pairs | mean_delta | W_stat | p_value | note |")
    lines.append("|-------------|-----------|---------|------------|--------|---------|------|")
    for threshold, res in sorted(extended_results.items()):
        n_new = len(run4_rows) // 2 + len(run5_rows) // 2
        note = f"original 10 + {n_new} new seeds (run4+run5)"
        lines.append(f"| {res['temperature']:.1e} | {threshold} | {res['n_pairs']} | "
                     f"{_fmt(res['mean_delta'])} | {_fmt(res['W_stat'])} | {_fmt(res['p_value'])} | {note} |")

    lines.append("")
    lines.append("## (c) Bridge compression deltas\n")
    lines.append("| label | type | seed | T=0 top3 | T=1e-4 top3 | Δ | T=0 min_k | T=1e-4 min_k |")
    lines.append("|-------|------|------|----------|-------------|---|-----------|--------------|")
    for r in compression_rows:
        lines.append(
            f"| {r['label']} | {r['checkpoint_type']} | {r['seed']} | "
            f"{_fmt(r['t0_top3'])} | {_fmt(r['t1_top3'])} | "
            f"{_fmt(r['delta_top3'],4)} | {r['t0_min_k_99pct']} | {r['t1_min_k_99pct']} |"
        )
    if comp_wilcoxon:
        lines.append(f"\nWilcoxon (milestone_99 deltas, n={comp_wilcoxon['n']}): "
                     f"W={comp_wilcoxon['W']:.1f} p={comp_wilcoxon['p']:.4f}\n")
    else:
        lines.append(f"\nn_milestone_99={len(valid_deltas_milestone)} — below threshold for Wilcoxon\n")

    lines.append("")
    lines.append("## (d) Rerun variance\n")
    lines.append(f"Found {len(variance_rows)} rows across repeated (seed, temperature) executions.")
    lines.append(f"See `rerun_variance.csv` for full table.\n")

    # Show per-key summary
    from collections import defaultdict
    groups = defaultdict(list)
    for vr in variance_rows:
        key = (vr["seed"], vr["temperature"], vr["threshold"])
        groups[key].append(vr)
    lines.append("| seed | temperature | threshold | epochs_across_runs |")
    lines.append("|------|-------------|-----------|-------------------|")
    for (seed, temperature, threshold), grp in sorted(groups.items()):
        epochs = [str(g.get("grok_epoch") or "censored") for g in grp]
        lines.append(f"| {seed} | {temperature:.1e} | {threshold} | {' / '.join(epochs)} |")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {out_md}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
