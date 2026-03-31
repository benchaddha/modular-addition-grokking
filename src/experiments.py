import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import plotly.graph_objects as go
import torch
from tqdm import tqdm

from .config import Config
from .dataset import build_batch_schedule, get_dataset
from .fourier_ablation import run_fourier_ablation_sweep
from .model import get_model
from .surgery import run_exhaustive_subset_ablation, run_surgery_ablation_sweep
from .train_physics import clone_state_dict, train_physics_run


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _median_q1_q3(values: Sequence[float]) -> Tuple[float, float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    median = torch.quantile(tensor, q=0.50).item()
    q1 = torch.quantile(tensor, q=0.25).item()
    q3 = torch.quantile(tensor, q=0.75).item()
    return median, q1, q3


def _km_points(
    event_times: Sequence[int],
    event_observed: Sequence[bool],
    max_time: int,
) -> Tuple[List[float], List[float]]:
    at_risk = len(event_times)
    survival = 1.0

    x_vals: List[float] = [0.0]
    y_vals: List[float] = [1.0]

    observed_times = sorted(
        {time for time, observed in zip(event_times, event_observed) if observed}
    )
    for current_time in observed_times:
        if at_risk <= 0:
            break
        deaths = sum(
            1
            for time, observed in zip(event_times, event_observed)
            if observed and time == current_time
        )
        removed = sum(1 for time in event_times if time == current_time)
        survival *= 1.0 - (deaths / at_risk)
        x_vals.append(float(current_time))
        y_vals.append(float(survival))
        at_risk -= removed

    if x_vals[-1] != float(max_time):
        x_vals.append(float(max_time))
        y_vals.append(y_vals[-1])

    return x_vals, y_vals


def _build_summary_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[float, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["temperature"], row["threshold"])].append(row)

    control_lookup: Dict[Tuple[int, float], Dict[str, Any]] = {}
    for row in rows:
        if row["temperature"] == 0.0:
            control_lookup[(row["seed"], row["threshold"])] = row

    summary_rows: List[Dict[str, Any]] = []
    for (temperature, threshold), group_rows in sorted(grouped.items()):
        total = len(group_rows)
        success_rows = [row for row in group_rows if row["success"]]
        success_epochs = [float(row["event_epoch"]) for row in success_rows]

        median_epoch = None
        iqr_low = None
        iqr_high = None
        if success_epochs:
            median_epoch, iqr_low, iqr_high = _median_q1_q3(success_epochs)

        paired_deltas: List[float] = []
        if temperature != 0.0:
            for row in success_rows:
                control = control_lookup.get((row["seed"], row["threshold"]))
                if control and control["success"]:
                    paired_deltas.append(row["event_epoch"] - control["event_epoch"])

        paired_delta_median = None
        paired_delta_q1 = None
        paired_delta_q3 = None
        if paired_deltas:
            paired_delta_median, paired_delta_q1, paired_delta_q3 = _median_q1_q3(
                paired_deltas
            )

        summary_rows.append(
            {
                "temperature": temperature,
                "threshold": threshold,
                "n_total": total,
                "n_success": len(success_rows),
                "n_censored": total - len(success_rows),
                "success_rate": len(success_rows) / total if total else 0.0,
                "median_grok_epoch": median_epoch,
                "iqr_low": iqr_low,
                "iqr_high": iqr_high,
                "paired_delta_median": paired_delta_median,
                "paired_delta_iqr_low": paired_delta_q1,
                "paired_delta_iqr_high": paired_delta_q3,
            }
        )
    return summary_rows


def _write_survival_figures(
    rows: Sequence[Dict[str, Any]],
    thresholds: Sequence[float],
    max_epochs: int,
    figures_dir: Path,
) -> List[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    for threshold in thresholds:
        threshold_rows = [row for row in rows if row["threshold"] == threshold]
        if not threshold_rows:
            continue

        fig = go.Figure()
        for temperature in sorted({row["temperature"] for row in threshold_rows}):
            group = [row for row in threshold_rows if row["temperature"] == temperature]
            event_times = [int(row["event_epoch"]) for row in group]
            observed = [not bool(row["censored"]) for row in group]
            x_vals, y_vals = _km_points(
                event_times=event_times,
                event_observed=observed,
                max_time=max_epochs,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    line_shape="hv",
                    name=f"T={temperature:g}",
                )
            )

        fig.update_layout(
            title=f"Kaplan-Meier Survival (threshold={threshold:.2f})",
            xaxis_title="Epoch",
            yaxis_title="Fraction Not Yet Grokked",
            yaxis_range=[0.0, 1.0],
            template="plotly_white",
        )
        output_path = figures_dir / f"physics_survival_{int(round(threshold * 100))}.html"
        fig.write_html(output_path)
        paths.append(output_path)

    return paths


def run_paired_physics_sweep(cfg: Config) -> Dict[str, Any]:
    cfg.validate()

    metrics_dir = Path("results") / "metrics"
    figures_dir = Path("results") / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: List[Dict[str, Any]] = []
    thresholds = [float(value) for value in cfg.physics.grok_thresholds]

    progress = tqdm(cfg.physics.seeds, desc="paired-seeds")
    for seed in progress:
        split_generator = torch.Generator().manual_seed(seed)
        dataset = get_dataset(cfg, generator=split_generator)

        train_size = dataset.train_indices.shape[0]
        schedule_generator = torch.Generator().manual_seed(seed)
        batch_schedule = build_batch_schedule(
            train_size=train_size,
            batch_size=cfg.train.batch_size,
            max_epochs=cfg.physics.max_epochs,
            generator=schedule_generator,
        )

        init_model = get_model(cfg, seed=seed)
        initial_state = clone_state_dict(init_model.state_dict())
        del init_model

        for temperature in cfg.physics.temperatures:
            noise_generator = torch.Generator().manual_seed(
                seed + cfg.physics.noise_seed_offset
            )
            run = train_physics_run(
                cfg=cfg,
                dataset=dataset,
                initial_state_dict=initial_state,
                batch_schedule=batch_schedule,
                noise_generator=noise_generator,
                temperature=float(temperature),
                seed=seed,
                max_epochs=cfg.physics.max_epochs,
                eval_every=cfg.physics.eval_every,
                thresholds=thresholds,
            )

            for threshold in thresholds:
                threshold_key = f"{threshold:.2f}"
                grok_epoch = run["grok_epochs"][threshold_key]
                success = grok_epoch is not None
                event_epoch = int(grok_epoch) if success else int(cfg.physics.max_epochs)

                raw_rows.append(
                    {
                        "seed": seed,
                        "temperature": float(temperature),
                        "threshold": threshold,
                        "grok_epoch": grok_epoch,
                        "event_epoch": event_epoch,
                        "success": success,
                        "censored": not success,
                        "max_epochs": cfg.physics.max_epochs,
                        "batch_schedule_hash": run["batch_schedule_hash"],
                    }
                )

    raw_path = metrics_dir / "physics_runs.jsonl"
    _write_jsonl(raw_path, raw_rows)

    summary_rows = _build_summary_rows(raw_rows)
    summary_path = metrics_dir / "physics_summary.csv"
    _write_summary_csv(summary_path, summary_rows)

    figure_paths = _write_survival_figures(
        rows=raw_rows,
        thresholds=thresholds,
        max_epochs=cfg.physics.max_epochs,
        figures_dir=figures_dir,
    )

    return {
        "raw_path": str(raw_path),
        "summary_path": str(summary_path),
        "figure_paths": [str(path) for path in figure_paths],
        "num_runs": len(raw_rows),
    }


def _mean_or_none(values: Sequence[float]) -> Any:
    if not values:
        return None
    return sum(values) / len(values)


def _build_hypothesis_b_summary_rows(
    cfg: Config,
    ablation_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows_by_checkpoint: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in ablation_rows:
        rows_by_checkpoint[row["checkpoint_path"]].append(row)

    chance_threshold = cfg.surgery.causal_test_chance_multiplier / cfg.model.p
    max_top_k = max(cfg.surgery.top_k)
    summary_rows: List[Dict[str, Any]] = []

    for checkpoint_path, checkpoint_rows in sorted(rows_by_checkpoint.items()):
        baseline_rows = [r for r in checkpoint_rows if r["condition"] == "baseline"]
        if len(baseline_rows) != 1:
            raise ValueError(f"Expected 1 baseline row for {checkpoint_path}.")
        baseline = baseline_rows[0]

        gate_pass = (
            baseline["train_acc"] >= cfg.surgery.min_baseline_train_acc
            and baseline["test_acc"] >= cfg.surgery.min_baseline_test_acc
        )

        top_k_rows = [r for r in checkpoint_rows if r["condition"] == "top_k"]
        top_k_rows.sort(key=lambda row: int(row["k"]))
        max_top_k_rows = [r for r in top_k_rows if int(r["k"]) == max_top_k]
        if len(max_top_k_rows) != 1:
            raise ValueError(
                f"Expected one top_k row with k={max_top_k} for {checkpoint_path}."
            )
        max_top_k_row = max_top_k_rows[0]

        random_max_rows = [
            r
            for r in checkpoint_rows
            if r["condition"] == "random_k" and int(r["k"]) == max_top_k
        ]
        all_heads_rows = [r for r in checkpoint_rows if r["condition"] == "all_heads"]
        all_heads_row = all_heads_rows[0] if all_heads_rows else None

        causal_pass = (
            gate_pass
            and max_top_k_row["test_acc"] <= chance_threshold
            and max_top_k_row["train_acc"] >= cfg.surgery.causal_train_floor
        )

        summary_rows.append(
            {
                "checkpoint_path": checkpoint_path,
                "gate_pass": gate_pass,
                "causal_pass": causal_pass,
                "chance_test_threshold": chance_threshold,
                "baseline_train_acc": baseline["train_acc"],
                "baseline_test_acc": baseline["test_acc"],
                "max_top_k": max_top_k,
                "topk_train_acc": max_top_k_row["train_acc"],
                "topk_test_acc": max_top_k_row["test_acc"],
                "random_topk_train_acc_mean": _mean_or_none(
                    [row["train_acc"] for row in random_max_rows]
                ),
                "random_topk_test_acc_mean": _mean_or_none(
                    [row["test_acc"] for row in random_max_rows]
                ),
                "all_heads_train_acc": (
                    all_heads_row["train_acc"] if all_heads_row is not None else None
                ),
                "all_heads_test_acc": (
                    all_heads_row["test_acc"] if all_heads_row is not None else None
                ),
            }
        )

    return summary_rows


def _write_hypothesis_b_figure(
    ablation_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    topk_rows = [row for row in ablation_rows if row["condition"] == "top_k"]
    random_rows = [row for row in ablation_rows if row["condition"] == "random_k"]
    baseline_rows = [row for row in ablation_rows if row["condition"] == "baseline"]

    k_values = sorted({int(row["k"]) for row in topk_rows})
    baseline_train = _mean_or_none([row["train_acc"] for row in baseline_rows])
    baseline_test = _mean_or_none([row["test_acc"] for row in baseline_rows])

    topk_train_y: List[float] = []
    topk_test_y: List[float] = []
    random_train_y: List[float] = []
    random_test_y: List[float] = []
    for k in k_values:
        topk_k = [row for row in topk_rows if int(row["k"]) == k]
        random_k = [row for row in random_rows if int(row["k"]) == k]
        topk_train_y.append(_mean_or_none([row["train_acc"] for row in topk_k]) or 0.0)
        topk_test_y.append(_mean_or_none([row["test_acc"] for row in topk_k]) or 0.0)
        random_train_y.append(_mean_or_none([row["train_acc"] for row in random_k]) or 0.0)
        random_test_y.append(_mean_or_none([row["test_acc"] for row in random_k]) or 0.0)

    fig = go.Figure()
    if baseline_train is not None and baseline_test is not None:
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[baseline_train],
                mode="markers",
                marker=dict(symbol="diamond", size=10),
                name="Baseline Train",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[baseline_test],
                mode="markers",
                marker=dict(symbol="diamond", size=10),
                name="Baseline Test",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=topk_train_y,
            mode="lines+markers",
            name="Top-k Train",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=topk_test_y,
            mode="lines+markers",
            name="Top-k Test",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=random_train_y,
            mode="lines+markers",
            line=dict(dash="dash"),
            name="Random-k Train (mean)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=random_test_y,
            mode="lines+markers",
            line=dict(dash="dash"),
            name="Random-k Test (mean)",
        )
    )

    fig.update_layout(
        title="Hypothesis B: Accuracy Under Top-k vs Random-k Head Ablation",
        xaxis_title="Number of Ablated Heads (k)",
        yaxis_title="Accuracy",
        template="plotly_white",
    )
    fig.write_html(output_path)
    return output_path


def _write_hypothesis_b_report(
    cfg: Config,
    summary_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(summary_rows)
    gate_pass = sum(1 for row in summary_rows if row["gate_pass"])
    causal_pass = sum(1 for row in summary_rows if row["causal_pass"])

    lines = [
        "# Hypothesis B Results (Draft)",
        "",
        "## Setup",
        f"- Checkpoints evaluated: {total}",
        f"- Baseline gate (train/test): {cfg.surgery.min_baseline_train_acc:.2f} / "
        f"{cfg.surgery.min_baseline_test_acc:.2f}",
        f"- Causal test chance threshold: "
        f"{cfg.surgery.causal_test_chance_multiplier:.1f}/p = "
        f"{cfg.surgery.causal_test_chance_multiplier / cfg.model.p:.4f}",
        f"- Causal train floor: {cfg.surgery.causal_train_floor:.2f}",
        "",
        "## Headline",
        f"- Gate pass: {gate_pass}/{total}",
        f"- Causal pass (max top-k): {causal_pass}/{total}",
        "",
        "## Per-checkpoint summary",
        "| checkpoint | gate_pass | causal_pass | baseline_train | baseline_test | "
        "topk_train | topk_test | random_topk_test_mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {Path(row['checkpoint_path']).name} | "
            f"{int(bool(row['gate_pass']))} | "
            f"{int(bool(row['causal_pass']))} | "
            f"{row['baseline_train_acc']:.4f} | "
            f"{row['baseline_test_acc']:.4f} | "
            f"{row['topk_train_acc']:.4f} | "
            f"{row['topk_test_acc']:.4f} | "
            f"{(row['random_topk_test_acc_mean'] or 0.0):.4f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_hypothesis_b_surgery(cfg: Config) -> Dict[str, Any]:
    cfg.validate()
    cfg.validate_surgery()

    metrics_dir = Path("results") / "metrics"
    figures_dir = Path("results") / "figures"
    reports_dir = Path("results") / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    ablation_path = metrics_dir / "surgery_ablations.jsonl"
    ablation_rows = run_surgery_ablation_sweep(cfg=cfg, output_path=ablation_path)

    summary_rows = _build_hypothesis_b_summary_rows(cfg=cfg, ablation_rows=ablation_rows)
    summary_path = metrics_dir / "surgery_summary.csv"
    _write_summary_csv(summary_path, summary_rows)

    figure_path = _write_hypothesis_b_figure(
        ablation_rows=ablation_rows,
        output_path=figures_dir / "hypothesis_b_ablation_curve.html",
    )
    report_path = _write_hypothesis_b_report(
        cfg=cfg,
        summary_rows=summary_rows,
        output_path=reports_dir / "hypothesis_b_results_draft.md",
    )

    return {
        "ablations_path": str(ablation_path),
        "summary_path": str(summary_path),
        "figure_path": str(figure_path),
        "report_path": str(report_path),
        "num_ablation_rows": len(ablation_rows),
        "num_checkpoints": len(summary_rows),
        "num_gate_pass": sum(1 for row in summary_rows if row["gate_pass"]),
        "num_causal_pass": sum(1 for row in summary_rows if row["causal_pass"]),
    }


def _checkpoint_stage_label(row: Dict[str, Any]) -> str:
    threshold = row.get("checkpoint_threshold")
    if threshold is not None:
        return f"testacc_{int(round(float(threshold) * 100)):02d}"
    checkpoint_type = row.get("checkpoint_type")
    if checkpoint_type:
        return str(checkpoint_type)
    return Path(row["checkpoint_path"]).name


def _build_transition_pilot_summary_rows(
    exhaustive_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows_by_checkpoint: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in exhaustive_rows:
        rows_by_checkpoint[row["checkpoint_path"]].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for checkpoint_path, checkpoint_rows in sorted(
        rows_by_checkpoint.items(),
        key=lambda item: (
            item[1][0].get("checkpoint_epoch") is None,
            item[1][0].get("checkpoint_epoch") or 0,
            item[0],
        ),
    ):
        baseline_rows = [row for row in checkpoint_rows if row["num_ablated_heads"] == 0]
        if len(baseline_rows) != 1:
            raise ValueError(f"Expected one baseline row for {checkpoint_path}.")
        baseline = baseline_rows[0]
        candidate_rows = [row for row in checkpoint_rows if row["num_ablated_heads"] > 0]
        best_selective = max(
            candidate_rows,
            key=lambda row: (
                float(row["selective_gap"]),
                -float(row["test_acc"]),
                float(row["train_acc"]),
            ),
        )
        strong_pass_rows = [row for row in candidate_rows if row["strong_h2_pass"]]
        best_strong_pass = None
        if strong_pass_rows:
            best_strong_pass = max(
                strong_pass_rows,
                key=lambda row: (
                    float(row["selective_gap"]),
                    float(row["train_acc"]),
                ),
            )

        summary_rows.append(
            {
                "checkpoint_path": checkpoint_path,
                "checkpoint_epoch": baseline.get("checkpoint_epoch"),
                "checkpoint_type": baseline.get("checkpoint_type"),
                "checkpoint_threshold": baseline.get("checkpoint_threshold"),
                "checkpoint_stage": _checkpoint_stage_label(baseline),
                "baseline_train_acc": baseline["baseline_train_acc"],
                "baseline_test_acc": baseline["baseline_test_acc"],
                "best_subset_label": best_selective["subset_label"],
                "best_subset_train_acc": best_selective["train_acc"],
                "best_subset_test_acc": best_selective["test_acc"],
                "best_subset_selective_gap": best_selective["selective_gap"],
                "strong_h2_any_pass": bool(strong_pass_rows),
                "strong_h2_subset_label": (
                    best_strong_pass["subset_label"] if best_strong_pass else None
                ),
                "strong_h2_train_acc": (
                    best_strong_pass["train_acc"] if best_strong_pass else None
                ),
                "strong_h2_test_acc": (
                    best_strong_pass["test_acc"] if best_strong_pass else None
                ),
            }
        )

    return summary_rows


def _write_transition_pilot_heatmap(
    exhaustive_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_order = sorted(
        {row["checkpoint_path"] for row in exhaustive_rows},
        key=lambda checkpoint_path: (
            next(
                row.get("checkpoint_epoch") is None
                for row in exhaustive_rows
                if row["checkpoint_path"] == checkpoint_path
            ),
            next(
                row.get("checkpoint_epoch") or 0
                for row in exhaustive_rows
                if row["checkpoint_path"] == checkpoint_path
            ),
            checkpoint_path,
        ),
    )
    subset_order = sorted(
        {row["subset_label"] for row in exhaustive_rows},
        key=lambda label: (
            next(
                row["num_ablated_heads"]
                for row in exhaustive_rows
                if row["subset_label"] == label
            ),
            label,
        ),
    )

    z_values: List[List[float]] = []
    y_labels: List[str] = []
    for checkpoint_path in checkpoint_order:
        checkpoint_rows = [
            row for row in exhaustive_rows if row["checkpoint_path"] == checkpoint_path
        ]
        row_lookup = {row["subset_label"]: row for row in checkpoint_rows}
        z_values.append([float(row_lookup[label]["test_acc"]) for label in subset_order])
        y_labels.append(_checkpoint_stage_label(checkpoint_rows[0]))

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=subset_order,
            y=y_labels,
            colorscale="Viridis",
            colorbar=dict(title="Test Accuracy"),
        )
    )
    fig.update_layout(
        title="Transition Pilot: Exhaustive Head-Subset Test Accuracy",
        xaxis_title="Ablated Head Subset",
        yaxis_title="Checkpoint Stage",
        template="plotly_white",
    )
    fig.write_html(output_path)
    return output_path


def _write_transition_pilot_scatter(
    exhaustive_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()

    checkpoint_order = sorted(
        {row["checkpoint_path"] for row in exhaustive_rows},
        key=lambda checkpoint_path: (
            next(
                row.get("checkpoint_epoch") is None
                for row in exhaustive_rows
                if row["checkpoint_path"] == checkpoint_path
            ),
            next(
                row.get("checkpoint_epoch") or 0
                for row in exhaustive_rows
                if row["checkpoint_path"] == checkpoint_path
            ),
            checkpoint_path,
        ),
    )

    for checkpoint_path in checkpoint_order:
        checkpoint_rows = [
            row for row in exhaustive_rows if row["checkpoint_path"] == checkpoint_path
        ]
        fig.add_trace(
            go.Scatter(
                x=[float(row["train_acc"]) for row in checkpoint_rows],
                y=[float(row["test_acc"]) for row in checkpoint_rows],
                mode="markers+text",
                text=[row["subset_label"] for row in checkpoint_rows],
                textposition="top center",
                name=_checkpoint_stage_label(checkpoint_rows[0]),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Train = Test",
        )
    )
    fig.update_layout(
        title="Transition Pilot: Train vs Test Accuracy Under Exhaustive Ablation",
        xaxis_title="Train Accuracy",
        yaxis_title="Test Accuracy",
        template="plotly_white",
    )
    fig.write_html(output_path)
    return output_path


def _write_transition_pilot_report(
    summary_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strong_pass_count = sum(1 for row in summary_rows if row["strong_h2_any_pass"])
    lines = [
        "# Hypothesis B Transition Pilot (Draft)",
        "",
        f"- Checkpoints evaluated: {len(summary_rows)}",
        f"- Strong H2 passes: {strong_pass_count}",
        "",
        "## Per-checkpoint summary",
        "| checkpoint_stage | epoch | baseline_train | baseline_test | best_subset | "
        "best_subset_train | best_subset_test | selective_gap | strong_h2_pass |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['checkpoint_stage']} | "
            f"{row['checkpoint_epoch']} | "
            f"{row['baseline_train_acc']:.4f} | "
            f"{row['baseline_test_acc']:.4f} | "
            f"{row['best_subset_label']} | "
            f"{row['best_subset_train_acc']:.4f} | "
            f"{row['best_subset_test_acc']:.4f} | "
            f"{row['best_subset_selective_gap']:.4f} | "
            f"{int(bool(row['strong_h2_any_pass']))} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_hypothesis_b_transition_pilot(cfg: Config) -> Dict[str, Any]:
    cfg.validate()
    cfg.validate_surgery()

    metrics_dir = Path("results") / "metrics"
    figures_dir = Path("results") / "figures"
    reports_dir = Path("results") / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    exhaustive_path = metrics_dir / "surgery_transition_pilot_exhaustive.jsonl"
    exhaustive_rows = run_exhaustive_subset_ablation(
        cfg=cfg,
        output_path=exhaustive_path,
    )

    summary_rows = _build_transition_pilot_summary_rows(exhaustive_rows=exhaustive_rows)
    summary_path = metrics_dir / "surgery_transition_pilot_summary.csv"
    _write_summary_csv(summary_path, summary_rows)

    heatmap_path = _write_transition_pilot_heatmap(
        exhaustive_rows=exhaustive_rows,
        output_path=figures_dir / "hypothesis_b_transition_pilot_heatmap.html",
    )
    scatter_path = _write_transition_pilot_scatter(
        exhaustive_rows=exhaustive_rows,
        output_path=figures_dir / "hypothesis_b_transition_pilot_scatter.html",
    )
    report_path = _write_transition_pilot_report(
        summary_rows=summary_rows,
        output_path=reports_dir / "hypothesis_b_transition_pilot.md",
    )

    return {
        "exhaustive_path": str(exhaustive_path),
        "summary_path": str(summary_path),
        "heatmap_path": str(heatmap_path),
        "scatter_path": str(scatter_path),
        "report_path": str(report_path),
        "num_checkpoints": len(summary_rows),
        "num_rows": len(exhaustive_rows),
        "num_strong_h2_pass": sum(1 for row in summary_rows if row["strong_h2_any_pass"]),
    }


def _build_fourier_ablation_summary_rows(
    ablation_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows_by_checkpoint_and_site: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in ablation_rows:
        rows_by_checkpoint_and_site[(row["checkpoint_path"], row["site"])].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (checkpoint_path, site), group_rows in sorted(
        rows_by_checkpoint_and_site.items(),
        key=lambda item: (
            item[1][0].get("checkpoint_epoch") is None,
            item[1][0].get("checkpoint_epoch") or 0,
            item[0][1],
            item[0][0],
        ),
    ):
        baseline_rows = [row for row in group_rows if row["frequency_set_label"] == "baseline"]
        if len(baseline_rows) != 1:
            raise ValueError(
                f"Expected one baseline Fourier row for {checkpoint_path} at site={site}."
            )
        baseline = baseline_rows[0]
        candidate_rows = [
            row for row in group_rows if row["frequency_set_label"] != "baseline"
        ]
        best_selective = max(
            candidate_rows,
            key=lambda row: (
                float(row["selective_gap"]),
                -float(row["test_acc"]),
                float(row["train_acc"]),
            ),
        )
        strong_rows = [row for row in candidate_rows if row["strong_h2_pass"]]
        best_strong = None
        if strong_rows:
            best_strong = max(
                strong_rows,
                key=lambda row: (float(row["selective_gap"]), float(row["train_acc"])),
            )

        summary_rows.append(
            {
                "checkpoint_path": checkpoint_path,
                "checkpoint_epoch": baseline.get("checkpoint_epoch"),
                "checkpoint_type": baseline.get("checkpoint_type"),
                "checkpoint_threshold": baseline.get("checkpoint_threshold"),
                "site": site,
                "baseline_train_acc": baseline["baseline_train_acc"],
                "baseline_test_acc": baseline["baseline_test_acc"],
                "best_frequency_set_label": best_selective["frequency_set_label"],
                "best_frequencies": best_selective["frequencies"],
                "best_frequency_train_acc": best_selective["train_acc"],
                "best_frequency_test_acc": best_selective["test_acc"],
                "best_frequency_selective_gap": best_selective["selective_gap"],
                "strong_h2_any_pass": bool(strong_rows),
                "strong_h2_frequency_set_label": (
                    best_strong["frequency_set_label"] if best_strong else None
                ),
                "strong_h2_train_acc": best_strong["train_acc"] if best_strong else None,
                "strong_h2_test_acc": best_strong["test_acc"] if best_strong else None,
            }
        )

    return summary_rows


def _write_fourier_single_frequency_heatmap(
    ablation_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> Path:
    single_rows = [row for row in ablation_rows if row["num_frequencies"] == 1]
    if not single_rows:
        raise ValueError("No single-frequency rows available for heatmap output.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    key_order = sorted(
        {(row["checkpoint_path"], row["site"]) for row in single_rows},
        key=lambda key: (
            next(
                row.get("checkpoint_epoch") is None
                for row in single_rows
                if row["checkpoint_path"] == key[0] and row["site"] == key[1]
            ),
            next(
                row.get("checkpoint_epoch") or 0
                for row in single_rows
                if row["checkpoint_path"] == key[0] and row["site"] == key[1]
            ),
            key[1],
            key[0],
        ),
    )
    frequency_order = sorted(
        {
            int(row["frequencies"][0])
            for row in single_rows
            if row["frequency_set_label"] != "baseline"
        }
    )

    z_values: List[List[float]] = []
    y_labels: List[str] = []
    for checkpoint_path, site in key_order:
        key_rows = [
            row
            for row in single_rows
            if row["checkpoint_path"] == checkpoint_path and row["site"] == site
        ]
        row_lookup = {int(row["frequencies"][0]): row for row in key_rows if row["frequencies"]}
        z_values.append([float(row_lookup[freq]["test_acc"]) for freq in frequency_order])
        stage = _checkpoint_stage_label(key_rows[0])
        y_labels.append(f"{stage}:{site}")

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=frequency_order,
            y=y_labels,
            colorscale="Viridis",
            colorbar=dict(title="Test Accuracy"),
        )
    )
    fig.update_layout(
        title="Fourier Ablation: Single-Frequency Test Accuracy",
        xaxis_title="Frequency",
        yaxis_title="Checkpoint Stage / Site",
        template="plotly_white",
    )
    fig.write_html(output_path)
    return output_path


def _write_fourier_ablation_report(
    summary_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strong_pass_count = sum(1 for row in summary_rows if row["strong_h2_any_pass"])
    lines = [
        "# Fourier Ablation Results (Draft)",
        "",
        f"- Checkpoint/site pairs evaluated: {len(summary_rows)}",
        f"- Strong H2 passes: {strong_pass_count}",
        "",
        "## Per-checkpoint summary",
        "| checkpoint_stage | site | baseline_train | baseline_test | best_frequency_set | "
        "best_train | best_test | selective_gap | strong_h2_pass |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        stage = _checkpoint_stage_label(row)
        lines.append(
            f"| {stage} | "
            f"{row['site']} | "
            f"{row['baseline_train_acc']:.4f} | "
            f"{row['baseline_test_acc']:.4f} | "
            f"{row['best_frequency_set_label']} | "
            f"{row['best_frequency_train_acc']:.4f} | "
            f"{row['best_frequency_test_acc']:.4f} | "
            f"{row['best_frequency_selective_gap']:.4f} | "
            f"{int(bool(row['strong_h2_any_pass']))} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_fourier_ablation(cfg: Config) -> Dict[str, Any]:
    cfg.validate()
    cfg.validate_fourier_ablation()

    metrics_dir = Path("results") / "metrics"
    figures_dir = Path("results") / "figures"
    reports_dir = Path("results") / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    sweep_outputs = run_fourier_ablation_sweep(cfg)
    score_rows = sweep_outputs["score_rows"]
    ablation_rows = sweep_outputs["ablation_rows"]

    scores_path = metrics_dir / "fourier_frequency_scores.jsonl"
    _write_jsonl(scores_path, score_rows)

    ablations_path = metrics_dir / "fourier_ablation_runs.jsonl"
    _write_jsonl(ablations_path, ablation_rows)

    summary_rows = _build_fourier_ablation_summary_rows(ablation_rows=ablation_rows)
    summary_path = metrics_dir / "fourier_ablation_summary.csv"
    _write_summary_csv(summary_path, summary_rows)

    heatmap_path = None
    single_frequency_rows = [row for row in ablation_rows if row["num_frequencies"] == 1]
    if single_frequency_rows:
        heatmap_path = _write_fourier_single_frequency_heatmap(
            ablation_rows=single_frequency_rows,
            output_path=figures_dir / "fourier_ablation_single_frequency_heatmap.html",
        )

    report_path = _write_fourier_ablation_report(
        summary_rows=summary_rows,
        output_path=reports_dir / "fourier_ablation_results_draft.md",
    )

    return {
        "scores_path": str(scores_path),
        "ablations_path": str(ablations_path),
        "summary_path": str(summary_path),
        "heatmap_path": str(heatmap_path) if heatmap_path is not None else None,
        "report_path": str(report_path),
        "num_score_rows": len(score_rows),
        "num_ablation_rows": len(ablation_rows),
        "num_checkpoint_site_pairs": len(summary_rows),
        "num_strong_h2_pass": sum(1 for row in summary_rows if row["strong_h2_any_pass"]),
    }
