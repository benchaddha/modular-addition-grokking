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
from .model import get_model
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
