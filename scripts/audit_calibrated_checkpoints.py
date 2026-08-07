#!/usr/bin/env python3
"""Audit canonical transition checkpoints without modifying historical files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibrated_separability import DataSubset, evaluate_subset
from src.config import Config
from src.dataset import get_dataset
from src.fourier_ablation import _checkpoint_config_or_default
from src.model import get_model
from src.surgery import _load_checkpoint_payload, _load_model_state_dict


CELL_META = {
    "p97_1layer": {"p": 97, "layers": 1},
    "p113_2layer": {"p": 113, "layers": 2},
}
MILESTONES = ["plateau", "05", "20", "50", "80", "95", "99"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", nargs="+", choices=sorted(CELL_META), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--milestones", nargs="+", default=MILESTONES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_dir(setting: str, seed: int) -> Path:
    return REPO_ROOT / "results" / "strategy2" / setting / f"seed_{seed}" / "artifacts"


def resolve_checkpoint(setting: str, seed: int, milestone: str) -> Path:
    directory = artifact_dir(setting, seed)
    if milestone == "plateau":
        candidates = sorted(directory.glob("*_epoch_002000.pt"))
    else:
        candidates = sorted(directory.glob(f"*_testacc_{milestone}.pt"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one {setting} seed {seed} milestone {milestone} checkpoint; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def evaluate_checkpoint(path: Path, batch_size: int) -> Dict[str, Any]:
    payload = _load_checkpoint_payload(path)
    cfg = _checkpoint_config_or_default(Config(), payload)
    model = get_model(cfg, seed=cfg.train.seed)
    model.load_state_dict(_load_model_state_dict(path))
    model = model.to("cpu")
    model.cfg.device = "cpu"
    model.eval()
    dataset = get_dataset(cfg, data_seed=cfg.train.seed)
    train_tokens, train_labels = dataset.train_data()
    test_tokens, test_labels = dataset.test_data()
    train = DataSubset(
        train_tokens,
        train_labels,
        torch.arange(len(train_tokens)),
        "audit_train",
    )
    test = DataSubset(
        test_tokens,
        test_labels,
        torch.arange(len(test_tokens)),
        "audit_test",
    )
    train_metrics = evaluate_subset(model, train, batch_size=batch_size)
    test_metrics = evaluate_subset(model, test, batch_size=batch_size)
    return {
        "payload": payload,
        "config": cfg,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def periodic_inventory(settings: Sequence[str], seeds: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for setting in settings:
        for seed in seeds:
            for path in sorted(artifact_dir(setting, seed).glob("*_epoch_*.pt")):
                payload = _load_checkpoint_payload(path)
                stored = payload.get("metrics") or {}
                rows.append(
                    {
                        "setting": setting,
                        "seed": seed,
                        "path": str(path.relative_to(REPO_ROOT)),
                        "update": payload.get("epoch"),
                        "stored_train_accuracy": stored.get("train_acc"),
                        "stored_test_accuracy": stored.get("test_acc"),
                        "size_bytes": path.stat().st_size,
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    expected = [
        {"setting": setting, "seed": seed, "milestone": milestone}
        for setting in args.settings
        for seed in args.seeds
        for milestone in args.milestones
    ]
    resolutions = []
    for item in expected:
        try:
            path = resolve_checkpoint(item["setting"], item["seed"], item["milestone"])
            resolutions.append({**item, "path": str(path.relative_to(REPO_ROOT))})
        except FileNotFoundError as exc:
            resolutions.append({**item, "path": None, "error": str(exc)})
    print(json.dumps({"expected": resolutions}, indent=2))
    if args.dry_run:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "checkpoint_audit_manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError("Audit manifest exists; use --resume or a new output directory.")
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {
            "version": 1,
            "expected_jobs": expected,
            "completed_jobs": [],
            "failed_jobs": [],
        }
    )
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    existing_csv = output_dir / "checkpoint_manifest.csv"
    if existing_csv.exists():
        with existing_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows_by_id[row["job_id"]] = row

    for item in expected:
        job_id = f"{item['setting']}-seed{item['seed']}-{item['milestone']}"
        if job_id in manifest["completed_jobs"] and args.resume:
            continue
        try:
            path = resolve_checkpoint(item["setting"], item["seed"], item["milestone"])
            result = evaluate_checkpoint(path, args.eval_batch_size)
            payload = result["payload"]
            cfg = result["config"]
            stored = payload.get("metrics") or {}
            train_metrics = result["train_metrics"]
            test_metrics = result["test_metrics"]
            row = {
                "job_id": job_id,
                "setting": item["setting"],
                "p": cfg.model.p,
                "layers": cfg.model.n_layers,
                "seed": item["seed"],
                "run_id": path.name.rsplit("_", 2)[0],
                "trajectory_kind": "historical_canonical",
                "milestone_label": item["milestone"],
                "path": str(path.relative_to(REPO_ROOT)),
                "update": payload.get("epoch"),
                "checkpoint_type": payload.get("checkpoint_type"),
                "checkpoint_threshold": payload.get("checkpoint_threshold"),
                "stored_train_accuracy": stored.get("train_acc"),
                "stored_test_accuracy": stored.get("test_acc"),
                "actual_train_accuracy": train_metrics.accuracy,
                "actual_test_accuracy": test_metrics.accuracy,
                "actual_train_cross_entropy": train_metrics.cross_entropy,
                "actual_test_cross_entropy": test_metrics.cross_entropy,
                "strict_plateau": (
                    item["milestone"] == "plateau"
                    and train_metrics.accuracy >= 0.99
                    and test_metrics.accuracy < 0.10
                ),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            rows_by_id[job_id] = row
            if job_id not in manifest["completed_jobs"]:
                manifest["completed_jobs"].append(job_id)
            manifest["failed_jobs"] = [
                failure
                for failure in manifest["failed_jobs"]
                if failure.get("job_id") != job_id
            ]
            print(
                f"[complete] {job_id} update={row['update']} "
                f"train={row['actual_train_accuracy']:.6f} "
                f"test={row['actual_test_accuracy']:.6f}",
                flush=True,
            )
        except Exception as exc:
            manifest["failed_jobs"].append(
                {
                    "job_id": job_id,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[failed] {job_id}: {exc}", file=sys.stderr, flush=True)
        write_csv(existing_csv, list(rows_by_id.values()))
        atomic_json(manifest_path, manifest)

    inventory = periodic_inventory(args.settings, args.seeds)
    write_csv(output_dir / "periodic_checkpoint_inventory.csv", inventory)
    missing = [
        {**item, "reason": "checkpoint_not_found"}
        for item in resolutions
        if item.get("path") is None
    ]
    if missing:
        write_csv(output_dir / "missing_checkpoints.csv", missing)
    else:
        (output_dir / "missing_checkpoints.csv").write_text(
            "setting,seed,milestone,reason\n", encoding="utf-8"
        )
    atomic_json(
        output_dir / "checkpoint_manifest.json",
        {
            "rows": list(rows_by_id.values()),
            "periodic_checkpoint_count": len(inventory),
            "missing": missing,
        },
    )
    if manifest["failed_jobs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
