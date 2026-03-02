import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.metrics import find_grok_epoch
from src.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline training for multiple seeds and summarize outcomes."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to baseline YAML config.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Seed values to run, e.g. --seeds 42 43 44.",
    )
    parser.add_argument(
        "--grok-threshold",
        type=float,
        default=0.95,
        help="Test accuracy threshold used to define grok epoch.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "baseline_seed_sweep"),
        help="Directory for seed-organized artifacts and summary files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun seeds even if prior summaries exist.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="offline",
        choices=["offline", "online", "disabled"],
        help="Weights & Biases mode used during training runs.",
    )
    return parser.parse_args()


def _snapshot_artifacts() -> Tuple[Set[Path], Set[Path]]:
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    checkpoints_dir = PROJECT_ROOT / "results" / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    metric_files = set(metrics_dir.glob("*.jsonl"))
    checkpoint_files = set(checkpoints_dir.glob("*.pt"))
    return metric_files, checkpoint_files


def _find_new_or_recent(
    before: Set[Path],
    after: Set[Path],
    start_time: float,
    suffixes: Sequence[str],
) -> List[Path]:
    created = sorted(after - before)
    if created:
        return created

    # Fallback: files modified during the run window.
    candidates = [
        path
        for path in after
        if path.suffix in suffixes and path.stat().st_mtime >= (start_time - 1.0)
    ]
    return sorted(candidates, key=lambda p: p.stat().st_mtime)


def _copy_if_exists(paths: Sequence[Path], dst_dir: Path) -> List[str]:
    copied: List[str] = []
    for path in paths:
        if path.exists():
            dst = dst_dir / path.name
            shutil.copy2(path, dst)
            copied.append(str(dst))
    return copied


def _summarize_history(
    history: List[Dict[str, float]],
    threshold: float,
) -> Dict[str, Optional[float]]:
    if not history:
        return {
            "best_test_acc": None,
            "last_test_acc": None,
            "grok_epoch": None,
        }
    best_test_acc = max(float(row["test_acc"]) for row in history)
    last_test_acc = float(history[-1]["test_acc"])
    grok_epoch = find_grok_epoch(history, threshold=threshold)
    return {
        "best_test_acc": best_test_acc,
        "last_test_acc": last_test_acc,
        "grok_epoch": grok_epoch,
    }


def main() -> None:
    args = parse_args()
    os.environ["WANDB_MODE"] = args.wandb_mode

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "summary.csv"
    summary_jsonl = output_dir / "summary.jsonl"

    base_cfg = Config.from_yaml(args.config)
    rows: List[Dict[str, object]] = []

    print(f"Running baseline seed sweep for seeds: {args.seeds}")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        artifacts_dir = seed_dir / "artifacts"
        seed_summary_path = seed_dir / "summary.json"

        if seed_summary_path.exists() and not args.force:
            existing = json.loads(seed_summary_path.read_text(encoding="utf-8"))
            rows.append(existing)
            print(f"[skip] seed={seed} already completed (use --force to rerun).")
            continue

        seed_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        cfg = Config.from_yaml(args.config)
        cfg.train.seed = seed
        cfg.logging.run_name = f"baseline-seed-{seed}"

        before_metrics, before_checkpoints = _snapshot_artifacts()
        start_time = time.time()
        history = train(cfg)
        after_metrics, after_checkpoints = _snapshot_artifacts()

        new_metrics = _find_new_or_recent(
            before=before_metrics,
            after=after_metrics,
            start_time=start_time,
            suffixes=[".jsonl"],
        )
        new_checkpoints = _find_new_or_recent(
            before=before_checkpoints,
            after=after_checkpoints,
            start_time=start_time,
            suffixes=[".pt"],
        )

        copied_metrics = _copy_if_exists(new_metrics, artifacts_dir)
        copied_checkpoints = _copy_if_exists(new_checkpoints, artifacts_dir)
        hist_summary = _summarize_history(history, threshold=args.grok_threshold)

        row: Dict[str, object] = {
            "seed": seed,
            "grok_threshold": args.grok_threshold,
            "best_test_acc": hist_summary["best_test_acc"],
            "last_test_acc": hist_summary["last_test_acc"],
            "grok_epoch": hist_summary["grok_epoch"],
            "num_eval_points": len(history),
            "metrics_files": copied_metrics,
            "checkpoint_files": copied_checkpoints,
        }
        seed_summary_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        best = row["best_test_acc"]
        best_text = "NA" if best is None else f"{float(best):.4f}"
        print(f"[done] seed={seed} best_test_acc={best_text} grok_epoch={row['grok_epoch']}")

    fieldnames = [
        "seed",
        "grok_threshold",
        "best_test_acc",
        "last_test_acc",
        "grok_epoch",
        "num_eval_points",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    with summary_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote sweep summary: {summary_csv}")
    print(f"Wrote sweep details: {summary_jsonl}")


if __name__ == "__main__":
    main()
