"""Strategy 3 analysis: sharedness vs compactness vs interference over training.

Inputs:
  - results/strategy3/trajectory_<setting>_seed<S>.jsonl  (run_strategy3_trajectory.py)
  - adversarial separability CSVs from Strategy 1/2 (existing artifacts)

Confirmatory analyses (frozen protocol):
  1. Sharedness versus phase across all three settings (adversarial CSVs).
  2. Compactness versus phase across all three settings (trajectory sweep).
  3. Temporal ordering: shared dependence / frequency stabilization /
     compression / generalization, per seed.
  4. Interference index trajectory, accuracy and cross-entropy versions,
     with calibration checks (confidence, logit norm).
  5. Per-seed (not per-intervention) statistical summaries with bootstrap CIs
     and exact permutation tests across cells.

Optional analysis: diffuse-frequency dependence of plateau training accuracy
(train-side k90 trajectory).

Outputs:
  results/strategy3/checkpoint_metrics.csv
  results/strategy3/sharedness_by_checkpoint.csv
  results/strategy3/seed_summaries.csv
  results/figures/strategy3_*.png
  results/reports/strategy3_report.md
"""

import glob
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRAT3 = PROJECT_ROOT / "results" / "strategy3"
FIGDIR = PROJECT_ROOT / "results" / "figures"
REPORTDIR = PROJECT_ROOT / "results" / "reports"

SETTINGS = {
    "p97_1layer": {
        "display": "p=97, 1 layer, full batch",
        "seeds": [100, 101, 102, 103, 104],
        "adv_glob": "results/strategy2/p97_1layer/seed_{seed}/battery/adversarial_separability*.csv",
        "color": "#0072B2",
    },
    "p113_2layer": {
        "display": "p=113, 2 layers, full batch",
        "seeds": [100, 101, 102, 103, 104],
        "adv_glob": "results/strategy2/p113_2layer/seed_{seed}/battery/adversarial_separability*.csv",
        "color": "#E69F00",
    },
    "original": {
        "display": "p=113, 1 layer, minibatch (original)",
        "seeds": [52, 53, 54],
        "adv_glob": "results/metrics/adversarial_separability_seed{seed}.csv",
        "color": "#009E73",
    },
}

PHASE_ORDER = ["pre_plateau", "plateau", "early_rise", "late_rise", "complete"]
AUC_KMAX = 16
RNG = np.random.default_rng(0)


def assign_phase(train_acc: float, test_acc: float) -> str:
    if train_acc < 0.99:
        return "pre_plateau"
    if test_acc < 0.10:
        return "plateau"
    if test_acc < 0.50:
        return "early_rise"
    if test_acc < 0.95:
        return "late_rise"
    return "complete"


def load_trajectory(setting: str, seed: int):
    path = STRAT3 / f"trajectory_{setting}_seed{seed}.jsonl"
    rows = [json.loads(line) for line in open(path)]
    # dedupe (resume runs may duplicate rows): keep last occurrence
    keyed = {}
    for row in rows:
        key = (
            row["checkpoint_path"],
            row["row_type"],
            row.get("intervention_mode"),
            row.get("label"),
        )
        keyed[key] = row
    return list(keyed.values())


def min_k_recovering(top_rows, split: str, frac: float, baseline_acc: float):
    if baseline_acc < 0.5:
        return None
    for row in sorted(top_rows, key=lambda r: r["k"]):
        if row[split]["acc"] >= frac * baseline_acc:
            return row["k"]
    return None


def checkpoint_metrics(setting: str, seed: int) -> pd.DataFrame:
    rows = load_trajectory(setting, seed)
    by_ckpt = defaultdict(list)
    for row in rows:
        by_ckpt[row["checkpoint_path"]].append(row)

    records = []
    for ckpt, ckpt_rows in by_ckpt.items():
        spectrum = [r for r in ckpt_rows if r["row_type"] == "spectrum"]
        if not spectrum:
            continue
        spec = spectrum[0]
        base_train = spec["baseline_train"]
        base_test = spec["baseline_test"]
        phase = assign_phase(base_train["acc"], base_test["acc"])

        keep_top = [
            r
            for r in ckpt_rows
            if r["row_type"] == "intervention"
            and r["intervention_mode"] == "keep_only_selected"
            and r["family"] == "top"
        ]
        abl_top = [
            r
            for r in ckpt_rows
            if r["row_type"] == "intervention"
            and r["intervention_mode"] == "ablate_selected"
            and r["family"] == "top"
        ]
        abl_rand = [
            r
            for r in ckpt_rows
            if r["row_type"] == "intervention"
            and r["intervention_mode"] == "ablate_selected"
            and r["family"] == "random"
        ]

        rec = {
            "setting": setting,
            "seed": seed,
            "checkpoint_path": ckpt,
            "checkpoint_name": Path(ckpt).name,
            "epoch": spec["checkpoint_epoch"],
            "checkpoint_type": spec["checkpoint_type"],
            "checkpoint_threshold": spec["checkpoint_threshold"],
            "phase": phase,
            "clean_train_acc": base_train["acc"],
            "clean_test_acc": base_test["acc"],
            "clean_train_ce": base_train["ce"],
            "clean_test_ce": base_test["ce"],
            "participation_ratio": spec["participation_ratio"],
            "top3_mass": spec["top3_mass"],
            "top5_mass": spec["top5_mass"],
            "top10_mass": spec["top10_mass"],
            "top5_freqs": tuple(spec["ranked_frequencies_top10"][:5]),
        }

        for frac, tag in [(0.90, "k90"), (0.95, "k95"), (0.99, "k99")]:
            rec[f"{tag}_train"] = min_k_recovering(keep_top, "train", frac, base_train["acc"])
            rec[f"{tag}_test"] = min_k_recovering(keep_top, "test", frac, base_test["acc"])

        auc_rows = [r for r in keep_top if r["k"] <= AUC_KMAX]
        if base_test["acc"] >= 0.5 and auc_rows:
            rec["suff_auc_test"] = float(
                np.mean([min(1.0, r["test"]["acc"] / base_test["acc"]) for r in auc_rows])
            )
        else:
            rec["suff_auc_test"] = None
        if base_train["acc"] >= 0.5 and auc_rows:
            rec["suff_auc_train"] = float(
                np.mean([min(1.0, r["train"]["acc"] / base_train["acc"]) for r in auc_rows])
            )
        else:
            rec["suff_auc_train"] = None

        abl_auc_rows = [r for r in abl_top if r["k"] <= AUC_KMAX]
        if base_test["acc"] >= 0.5 and abl_auc_rows:
            rec["nec_auc_test"] = float(
                np.mean(
                    [
                        np.clip((base_test["acc"] - r["test"]["acc"]) / base_test["acc"], 0, 1)
                        for r in abl_auc_rows
                    ]
                )
            )
        else:
            rec["nec_auc_test"] = None

        # fixed-intervention sharedness differential at k=5 (relative drops)
        abl5 = [r for r in abl_top if r["k"] == 5]
        rand5 = [r for r in abl_rand if r["k"] == 5]
        if abl5 and base_train["acc"] >= 0.5 and base_test["acc"] >= 0.5:
            r5 = abl5[0]
            rel_train = (base_train["acc"] - r5["train"]["acc"]) / base_train["acc"]
            rel_test = (base_test["acc"] - r5["test"]["acc"]) / base_test["acc"]
            rec["abl5_rel_gap"] = rel_test - rel_train
            if rand5:
                gaps = []
                for rr in rand5:
                    rt = (base_train["acc"] - rr["train"]["acc"]) / base_train["acc"]
                    rs = (base_test["acc"] - rr["test"]["acc"]) / base_test["acc"]
                    gaps.append(rs - rt)
                rec["abl5_rel_gap_random"] = float(np.mean(gaps))
        else:
            rec["abl5_rel_gap"] = None
            rec["abl5_rel_gap_random"] = None

        # interference index over keep-only top-k rows
        best = None
        for row in keep_top:
            i_acc = (row["test"]["acc"] - base_test["acc"]) - (
                row["train"]["acc"] - base_train["acc"]
            )
            if best is None or i_acc > best[0]:
                best = (i_acc, row)
        if best is not None:
            i_acc, row = best
            rec["interference_max_acc"] = i_acc
            rec["interference_argmax_k"] = row["k"]
            rec["interference_ce_at_argmax"] = (
                base_test["ce"] - row["test"]["ce"]
            ) - (base_train["ce"] - row["train"]["ce"])
            rec["interference_test_gain_acc"] = row["test"]["acc"] - base_test["acc"]
            rec["interference_test_gain_ce"] = base_test["ce"] - row["test"]["ce"]
            rec["interference_conf_delta_test"] = (
                row["test"]["confidence"] - base_test["confidence"]
            )
            rec["interference_logitnorm_ratio_test"] = (
                row["test"]["logit_norm"] / base_test["logit_norm"]
                if base_test["logit_norm"]
                else None
            )
        # decomposition: best test-side gain from keep-only (the interpretable
        # half of I_k — positive means the discarded residual was actively
        # hurting test performance)
        best_gain = None
        for row in keep_top:
            gain = row["test"]["acc"] - base_test["acc"]
            if best_gain is None or gain > best_gain[0]:
                best_gain = (gain, row)
        if best_gain is not None:
            gain, row = best_gain
            rec["keep_test_gain_max"] = gain
            rec["keep_test_gain_k"] = row["k"]
            rec["keep_test_gain_ce"] = base_test["ce"] - row["test"]["ce"]
            rec["keep_test_gain_conf_delta"] = (
                row["test"]["confidence"] - base_test["confidence"]
            )
            rec["keep_test_gain_logitnorm_ratio"] = (
                row["test"]["logit_norm"] / base_test["logit_norm"]
                if base_test["logit_norm"]
                else None
            )
            rec["keep_test_gain_train_delta"] = (
                row["train"]["acc"] - base_train["acc"]
            )
            rec["interference_at_gain_k"] = gain - (
                row["train"]["acc"] - base_train["acc"]
            )

        k5 = [r for r in keep_top if r["k"] == 5]
        if k5:
            row = k5[0]
            rec["interference_k5_acc"] = (row["test"]["acc"] - base_test["acc"]) - (
                row["train"]["acc"] - base_train["acc"]
            )
            rec["interference_k5_ce"] = (base_test["ce"] - row["test"]["ce"]) - (
                base_train["ce"] - row["train"]["ce"]
            )
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    numeric_cols = [
        c
        for c in df.columns
        if c
        not in (
            "setting",
            "checkpoint_path",
            "checkpoint_name",
            "checkpoint_type",
            "phase",
            "top5_freqs",
        )
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("epoch").reset_index(drop=True)


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def milestone_epoch(df: pd.DataFrame, threshold: float):
    hits = df[
        (df["checkpoint_type"] == "milestone")
        & (df["checkpoint_threshold"].astype(float) == threshold)
        & (df["checkpoint_name"].str.contains("testacc"))
    ]
    if len(hits):
        return float(hits["epoch"].iloc[0])
    crossed = df[df["clean_test_acc"] >= threshold]
    return float(crossed["epoch"].iloc[0]) if len(crossed) else None


def seed_summary(setting: str, seed: int, df: pd.DataFrame, adv: pd.DataFrame) -> dict:
    df = df.sort_values("epoch")
    final = df.iloc[-1]
    e_t05 = milestone_epoch(df, 0.05)
    e_t99 = milestone_epoch(df, 0.99)

    def norm_time(epoch):
        if e_t05 is None or e_t99 is None or e_t99 == e_t05:
            return None
        return (epoch - e_t05) / (e_t99 - e_t05)

    # frequency identity stabilization (top-5 vs final, sustained Jaccard >= 0.6)
    final_top5 = final["top5_freqs"]
    epochs = df["epoch"].to_numpy(dtype=float)
    jacs = np.array([jaccard(t, final_top5) for t in df["top5_freqs"]])
    e_freq_stable = None
    for i in range(len(df)):
        if np.all(jacs[i:] >= 0.6):
            e_freq_stable = float(epochs[i])
            break

    # train-side compression stabilization
    k90_train_final = final["k90_train"]
    e_k90_train_stable = None
    if pd.notna(k90_train_final):
        vals = df["k90_train"].to_numpy(dtype=float)
        for i in range(len(df)):
            tail = vals[i:]
            if np.all(np.isnan(tail) | (tail <= k90_train_final + 2)):
                if not np.isnan(vals[i]):
                    e_k90_train_stable = float(epochs[i])
                    break

    plateau = df[df["phase"] == "plateau"]
    early = df[df["phase"] == "early_rise"]
    complete = df[df["phase"] == "complete"]

    # interference peak over trajectory
    imax_idx = df["interference_max_acc"].astype(float).idxmax()
    imax_row = df.loc[imax_idx]

    summary = {
        "setting": setting,
        "seed": seed,
        "e_t05": e_t05,
        "e_t99": e_t99,
        "e_freq_stable": e_freq_stable,
        "e_freq_stable_norm": norm_time(e_freq_stable) if e_freq_stable is not None else None,
        "e_k90_train_stable": e_k90_train_stable,
        "e_k90_train_stable_norm": (
            norm_time(e_k90_train_stable) if e_k90_train_stable is not None else None
        ),
        "pr_plateau": float(plateau["participation_ratio"].mean()) if len(plateau) else None,
        "pr_early_rise": float(early["participation_ratio"].mean()) if len(early) else None,
        "pr_complete": float(final["participation_ratio"]),
        "delta_pr_plateau_to_onset": (
            float(early["participation_ratio"].mean() - plateau["participation_ratio"].mean())
            if len(plateau) and len(early)
            else None
        ),
        "k90_train_plateau": (
            float(plateau["k90_train"].median()) if len(plateau) and plateau["k90_train"].notna().any() else None
        ),
        "k90_train_complete": float(final["k90_train"]) if pd.notna(final["k90_train"]) else None,
        "k90_test_complete": float(final["k90_test"]) if pd.notna(final["k90_test"]) else None,
        "k95_test_complete": float(final["k95_test"]) if pd.notna(final["k95_test"]) else None,
        "k99_test_complete": float(final["k99_test"]) if pd.notna(final["k99_test"]) else None,
        "top5_mass_complete": float(final["top5_mass"]),
        "suff_auc_test_complete": final["suff_auc_test"],
        "interference_max_acc": float(imax_row["interference_max_acc"]),
        "interference_max_epoch": float(imax_row["epoch"]),
        "interference_max_norm_time": norm_time(float(imax_row["epoch"])),
        "interference_max_k": imax_row["interference_argmax_k"],
        "interference_ce_at_peak": float(imax_row["interference_ce_at_argmax"]),
        "interference_test_gain_acc_at_peak": float(imax_row["interference_test_gain_acc"]),
        "interference_test_gain_ce_at_peak": float(imax_row["interference_test_gain_ce"]),
        "interference_conf_delta_at_peak": float(imax_row["interference_conf_delta_test"]),
        "interference_logitnorm_ratio_at_peak": float(
            imax_row["interference_logitnorm_ratio_test"]
        ),
        "interference_plateau_max": (
            float(plateau["interference_max_acc"].max()) if len(plateau) else None
        ),
        "interference_complete": float(final["interference_max_acc"]),
        "interference_peak_phase": str(imax_row["phase"]),
    }

    # keep-only test-side gain (decomposed interference)
    gmax_idx = df["keep_test_gain_max"].astype(float).idxmax()
    grow = df.loc[gmax_idx]
    summary.update(
        {
            "keep_gain_max": float(grow["keep_test_gain_max"]),
            "keep_gain_max_epoch": float(grow["epoch"]),
            "keep_gain_max_norm_time": norm_time(float(grow["epoch"])),
            "keep_gain_max_phase": str(grow["phase"]),
            "keep_gain_max_k": float(grow["keep_test_gain_k"]),
            "keep_gain_ce_at_peak": float(grow["keep_test_gain_ce"]),
            "keep_gain_conf_delta_at_peak": float(grow["keep_test_gain_conf_delta"]),
            "keep_gain_logitnorm_ratio_at_peak": float(
                grow["keep_test_gain_logitnorm_ratio"]
            ),
            "keep_gain_train_delta_at_peak": float(grow["keep_test_gain_train_delta"]),
            "keep_gain_plateau_max": (
                float(plateau["keep_test_gain_max"].astype(float).max())
                if len(plateau)
                else None
            ),
            "keep_gain_complete": float(final["keep_test_gain_max"]),
        }
    )

    # sharedness (adversarial) per phase
    if adv is not None and len(adv):
        for phase in PHASE_ORDER:
            sub = adv[adv["phase"] == phase]
            learned = sub[sub["method"] == "adversarial"]
            controls = sub[sub["method"].isin(["random", "pca"])]
            if len(learned):
                summary[f"adv_gap_max_{phase}"] = float(learned["selective_gap"].max())
                retained = learned[
                    learned["int_train_acc_heldout"]
                    >= 0.9 * learned["clean_train_acc_heldout"]
                ]
                ct = learned["clean_test_acc_heldout"].iloc[0]
                if len(retained) and ct >= 0.2:
                    summary[f"adv_rel_damage90_{phase}"] = float(
                        (retained["test_drop"] / retained["clean_test_acc_heldout"]).max()
                    )
                else:
                    summary[f"adv_rel_damage90_{phase}"] = None
                summary[f"adv_minus_control_{phase}"] = (
                    float(learned["selective_gap"].max() - controls["selective_gap"].max())
                    if len(controls)
                    else None
                )
            else:
                summary[f"adv_gap_max_{phase}"] = None
                summary[f"adv_rel_damage90_{phase}"] = None
                summary[f"adv_minus_control_{phase}"] = None

    # temporal ordering flags (t50-ish checkpoint: first late_rise checkpoint)
    late = df[df["phase"] == "late_rise"]
    if len(late) and pd.notna(final["k90_test"]):
        mid = late.iloc[0]
        summary["compact_at_t50"] = (
            bool(mid["k90_test"] <= final["k90_test"] + 2) if pd.notna(mid["k90_test"]) else False
        )
        summary["pr_ratio_t50_final"] = float(
            mid["participation_ratio"] / final["participation_ratio"]
        )
    else:
        summary["compact_at_t50"] = None
        summary["pr_ratio_t50_final"] = None
    return summary


def load_adversarial(setting: str, seed: int):
    cfg = SETTINGS[setting]
    paths = []
    for pattern in [cfg["adv_glob"].format(seed=seed)]:
        paths.extend(glob.glob(str(PROJECT_ROOT / pattern)))
    if setting == "original":
        extra = PROJECT_ROOT / f"results/metrics/adversarial_separability_postembed_seed{seed}.csv"
        if extra.exists():
            paths.append(str(extra))
    if not paths:
        return None
    frames = [pd.read_csv(p) for p in sorted(set(paths))]
    adv = pd.concat(frames, ignore_index=True)
    adv = adv[adv["site"] == "pre_unembed"] if "site" in adv else adv
    adv["phase"] = [
        assign_phase(tr, te)
        for tr, te in zip(adv["clean_train_acc_heldout"], adv["clean_test_acc_heldout"])
    ]
    adv["setting"] = setting
    adv["seed"] = seed
    return adv


def bootstrap_ci(values, n_boot=10000, alpha=0.05):
    values = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
    if len(values) == 0:
        return (None, None, None)
    boots = np.median(RNG.choice(values, size=(n_boot, len(values))), axis=1)
    return (
        float(np.median(values)),
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
    )


def exact_permutation_p(a, b):
    """Two-sided exact permutation test on difference of means (small n)."""
    a = [v for v in a if v is not None and not np.isnan(v)]
    b = [v for v in b if v is not None and not np.isnan(v)]
    if not a or not b:
        return None
    pooled = a + b
    n_a = len(a)
    observed = abs(np.mean(a) - np.mean(b))
    count = 0
    total = 0
    for combo in itertools.combinations(range(len(pooled)), n_a):
        grp_a = [pooled[i] for i in combo]
        grp_b = [pooled[i] for i in range(len(pooled)) if i not in combo]
        if abs(np.mean(grp_a) - np.mean(grp_b)) >= observed - 1e-12:
            count += 1
        total += 1
    return count / total


# ---------------------------------------------------------------- figures ---

def fig_compactness(all_ckpt: pd.DataFrame, summaries: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex="col")
    for col, (setting, cfg) in enumerate(SETTINGS.items()):
        color = cfg["color"]
        sub = all_ckpt[all_ckpt["setting"] == setting]
        for row_i, metric, label in [
            (0, "participation_ratio", "Spectral participation ratio"),
            (1, "k90_train", "k90 (train recovery)"),
        ]:
            ax = axes[row_i][col]
            for seed, seed_df in sub.groupby("seed"):
                s = summaries[(summaries["setting"] == setting) & (summaries["seed"] == seed)]
                e05, e99 = s["e_t05"].iloc[0], s["e_t99"].iloc[0]
                if pd.isna(e05) or pd.isna(e99) or e99 == e05:
                    continue
                seed_df = seed_df.sort_values("epoch")
                tau = (seed_df["epoch"] - e05) / (e99 - e05)
                vals = seed_df[metric].astype(float)
                ax.plot(tau, vals, color=color, alpha=0.55, lw=1.4)
                if row_i == 1:
                    k90t = seed_df["k90_test"].astype(float)
                    ax.plot(tau, k90t, color="#555555", alpha=0.45, lw=1.2, ls="--")
            ax.axvline(0, color="#999999", lw=0.7, ls=":")
            ax.axvline(1, color="#999999", lw=0.7, ls=":")
            ax.set_xlim(-1.2, 1.3)
            if row_i == 0:
                ax.set_title(cfg["display"], fontsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row_i == 1:
                ax.set_xlabel("normalized time (0 = test 5%, 1 = test 99%)", fontsize=8)
            ax.grid(alpha=0.25, lw=0.4)
    axes[1][2].plot([], [], color="#555555", ls="--", label="k90 (test recovery)")
    axes[1][2].legend(fontsize=8, loc="upper right")
    fig.suptitle("Strategy 3: circuit compression trajectories (one line per seed)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "strategy3_compactness_trajectories.png", dpi=160)
    plt.close(fig)


def fig_sufficiency(trajectories: dict):
    phases = [("plateau", "plateau"), ("late_rise", "mid transition"), ("complete", "complete")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for col, (setting, cfg) in enumerate(SETTINGS.items()):
        ax = axes[col]
        for (phase, phase_label), ls in zip(phases, ["-", "--", ":"]):
            curves = []
            for (s, seed), rows in trajectories.items():
                if s != setting:
                    continue
                by_ckpt = defaultdict(list)
                for r in rows:
                    by_ckpt[r["checkpoint_path"]].append(r)
                for ckpt_rows in by_ckpt.values():
                    spec = [r for r in ckpt_rows if r["row_type"] == "spectrum"]
                    if not spec:
                        continue
                    spec = spec[0]
                    # test-acc-based bins so the minibatch regime (train<0.99
                    # mid-transition) still contributes a mid-transition curve
                    te = spec["baseline_test"]["acc"]
                    tr = spec["baseline_train"]["acc"]
                    if phase == "plateau":
                        ok = tr >= 0.99 and te < 0.10
                    elif phase == "late_rise":
                        ok = 0.30 <= te < 0.90
                    else:
                        ok = te >= 0.95
                    if not ok:
                        continue
                    keep = sorted(
                        [
                            r
                            for r in ckpt_rows
                            if r["row_type"] == "intervention"
                            and r["intervention_mode"] == "keep_only_selected"
                            and r["family"] == "top"
                        ],
                        key=lambda r: r["k"],
                    )
                    ks = [r["k"] for r in keep]
                    accs = [r["test"]["acc"] for r in keep]
                    curves.append((ks, accs))
            if curves:
                kgrid = curves[0][0]
                mat = np.full((len(curves), len(kgrid)), np.nan)
                for i, (ks, accs) in enumerate(curves):
                    for j, k in enumerate(kgrid):
                        if k in ks:
                            mat[i, j] = accs[ks.index(k)]
                mean = np.nanmean(mat, axis=0)
                ax.plot(kgrid, mean, ls=ls, color=cfg["color"], lw=2, label=phase_label)
        ax.set_xscale("log")
        ax.set_xlabel("k (keep only top-k Fourier modes)", fontsize=9)
        ax.set_title(cfg["display"], fontsize=10)
        ax.grid(alpha=0.25, lw=0.4)
        if col == 0:
            ax.set_ylabel("test accuracy (absolute)", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Sufficiency curves by phase (mean across seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "strategy3_sufficiency_curves.png", dpi=160)
    plt.close(fig)


def fig_interference(all_ckpt: pd.DataFrame, summaries: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex="col")
    for col, (setting, cfg) in enumerate(SETTINGS.items()):
        sub = all_ckpt[all_ckpt["setting"] == setting]
        for row_i, metric, label in [
            (0, "keep_test_gain_max", "best keep-only test gain (accuracy)"),
            (1, "keep_test_gain_ce", "test CE improvement at same k (nats)"),
        ]:
            ax = axes[row_i][col]
            for seed, seed_df in sub.groupby("seed"):
                s = summaries[(summaries["setting"] == setting) & (summaries["seed"] == seed)]
                e05, e99 = s["e_t05"].iloc[0], s["e_t99"].iloc[0]
                if pd.isna(e05) or pd.isna(e99) or e99 == e05:
                    continue
                seed_df = seed_df.sort_values("epoch")
                tau = (seed_df["epoch"] - e05) / (e99 - e05)
                ax.plot(tau, seed_df[metric].astype(float), color=cfg["color"], alpha=0.55, lw=1.4)
            ax.axhline(0, color="#999999", lw=0.7)
            ax.axvline(0, color="#999999", lw=0.7, ls=":")
            ax.axvline(1, color="#999999", lw=0.7, ls=":")
            ax.set_xlim(-1.2, 1.3)
            if row_i == 0:
                ax.set_title(cfg["display"], fontsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row_i == 1:
                ax.set_xlabel("normalized time (0 = test 5%, 1 = test 99%)", fontsize=8)
            ax.grid(alpha=0.25, lw=0.4)
    fig.suptitle(
        "Diffuse-component interference: does keeping only the top-k circuit IMPROVE test? (one line per seed)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "strategy3_interference_index.png", dpi=160)
    plt.close(fig)


def fig_sharedness(adv_all: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for col, (setting, cfg) in enumerate(SETTINGS.items()):
        ax = axes[col]
        sub = adv_all[adv_all["setting"] == setting]
        xticks, xlabels = [], []
        for i, phase in enumerate(PHASE_ORDER):
            psub = sub[sub["phase"] == phase]
            if not len(psub):
                continue
            learned = psub[psub["method"] == "adversarial"]
            controls = psub[psub["method"].isin(["random", "pca"])]
            per_seed_learned = learned.groupby("seed")["selective_gap"].max()
            per_seed_control = controls.groupby("seed")["selective_gap"].max()
            jitter = (np.arange(len(per_seed_learned)) - len(per_seed_learned) / 2) * 0.04
            ax.scatter(
                np.full(len(per_seed_learned), i) + jitter,
                per_seed_learned.values,
                color=cfg["color"],
                s=28,
                zorder=3,
                label="learned attack" if i == 1 and col == 0 else None,
            )
            if len(per_seed_control):
                jitter_c = (np.arange(len(per_seed_control)) - len(per_seed_control) / 2) * 0.04
                ax.scatter(
                    np.full(len(per_seed_control), i) + 0.18 + jitter_c,
                    per_seed_control.values,
                    facecolors="none",
                    edgecolors="#555555",
                    s=28,
                    zorder=3,
                    label="random/PCA control" if i == 1 and col == 0 else None,
                )
            # gap a truly separable circuit would produce at this phase:
            # test falls to chance, train preserved -> gap = clean_test - 2/p
            p_val = 97 if setting == "p97_1layer" else 113
            ceiling = learned["clean_test_acc_heldout"].mean() - 2.0 / p_val
            if ceiling > 0.05:
                ax.plot(
                    [i - 0.15, i + 0.33],
                    [ceiling, ceiling],
                    color="#D55E00",
                    lw=1.2,
                    ls="--",
                    zorder=2,
                )
            xticks.append(i)
            xlabels.append(phase.replace("_", "\n"))
        ax.plot([], [], color="#D55E00", lw=1.2, ls="--", label="gap if truly separable")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_title(cfg["display"], fontsize=10)
        ax.grid(alpha=0.25, lw=0.4, axis="y")
        if col == 0:
            ax.set_ylabel("max selective gap (test drop − train drop)", fontsize=9)
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Sharedness vs phase: best learned attack per seed never separates train from test", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "strategy3_sharedness_vs_phase.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------- main ---

def main() -> None:
    STRAT3.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    REPORTDIR.mkdir(parents=True, exist_ok=True)

    all_ckpt_frames = []
    summaries = []
    adv_frames = []
    trajectories = {}

    for setting, cfg in SETTINGS.items():
        for seed in cfg["seeds"]:
            trajectories[(setting, seed)] = load_trajectory(setting, seed)
            df = checkpoint_metrics(setting, seed)
            all_ckpt_frames.append(df)
            adv = load_adversarial(setting, seed)
            if adv is not None:
                adv_frames.append(adv)
            summaries.append(seed_summary(setting, seed, df, adv))

    all_ckpt = pd.concat(all_ckpt_frames, ignore_index=True)
    summaries_df = pd.DataFrame(summaries)
    adv_all = pd.concat(adv_frames, ignore_index=True)

    all_ckpt.drop(columns=["top5_freqs"]).to_csv(STRAT3 / "checkpoint_metrics.csv", index=False)
    summaries_df.to_csv(STRAT3 / "seed_summaries.csv", index=False)

    shared_cols = [
        "setting",
        "seed",
        "checkpoint_path",
        "phase",
        "method",
        "rank",
        "lam",
        "clean_train_acc_heldout",
        "clean_test_acc_heldout",
        "int_train_acc_heldout",
        "int_test_acc_heldout",
        "train_drop",
        "test_drop",
        "selective_gap",
        "separability_pass",
        "scoped_separability_pass",
    ]
    adv_all[shared_cols].to_csv(STRAT3 / "sharedness_by_checkpoint.csv", index=False)

    fig_compactness(all_ckpt, summaries_df)
    fig_sufficiency(trajectories)
    fig_interference(all_ckpt, summaries_df)
    fig_sharedness(adv_all)

    # ---------------- cross-cell statistics on seed-level summaries ----------
    stats = {}
    for metric in [
        "k90_test_complete",
        "k99_test_complete",
        "pr_complete",
        "top5_mass_complete",
        "interference_max_acc",
        "interference_max_norm_time",
        "keep_gain_max",
        "keep_gain_max_norm_time",
        "keep_gain_ce_at_peak",
        "adv_gap_max_complete",
        "adv_gap_max_plateau",
    ]:
        stats[metric] = {}
        for setting in SETTINGS:
            vals = summaries_df[summaries_df["setting"] == setting][metric]
            vals = [v for v in vals if v is not None]
            stats[metric][setting] = bootstrap_ci(vals)
    perm = {
        "k90_test_complete": exact_permutation_p(
            summaries_df[summaries_df["setting"] == "p97_1layer"]["k90_test_complete"].astype(float).tolist(),
            summaries_df[summaries_df["setting"] == "p113_2layer"]["k90_test_complete"].astype(float).tolist(),
        ),
        "pr_complete": exact_permutation_p(
            summaries_df[summaries_df["setting"] == "p97_1layer"]["pr_complete"].astype(float).tolist(),
            summaries_df[summaries_df["setting"] == "p113_2layer"]["pr_complete"].astype(float).tolist(),
        ),
        "interference_max_acc": exact_permutation_p(
            summaries_df[summaries_df["setting"] == "p97_1layer"]["interference_max_acc"].astype(float).tolist(),
            summaries_df[summaries_df["setting"] == "p113_2layer"]["interference_max_acc"].astype(float).tolist(),
        ),
        "keep_gain_max_fullbatch_vs_original": exact_permutation_p(
            summaries_df[summaries_df["setting"].isin(["p97_1layer", "p113_2layer"])][
                "keep_gain_max"
            ]
            .astype(float)
            .tolist(),
            summaries_df[summaries_df["setting"] == "original"]["keep_gain_max"]
            .astype(float)
            .tolist(),
        ),
    }

    with open(STRAT3 / "cross_cell_stats.json", "w") as handle:
        json.dump({"bootstrap_median_ci": stats, "permutation_p_p97_vs_p113": perm}, handle, indent=2)

    print("Wrote:")
    for path in [
        STRAT3 / "checkpoint_metrics.csv",
        STRAT3 / "seed_summaries.csv",
        STRAT3 / "sharedness_by_checkpoint.csv",
        STRAT3 / "cross_cell_stats.json",
        FIGDIR / "strategy3_compactness_trajectories.png",
        FIGDIR / "strategy3_sufficiency_curves.png",
        FIGDIR / "strategy3_interference_index.png",
        FIGDIR / "strategy3_sharedness_vs_phase.png",
    ]:
        print(" ", path)


if __name__ == "__main__":
    main()
