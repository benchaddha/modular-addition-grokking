#!/usr/bin/env python3
"""Compare conditional Fourier-tail retention with literal selected-span retention."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cached_preunembed import build_cached_preunembed_model
from src.calibrated_separability import DataSubset, evaluate_subset, sha256_json, tensor_checksum
from src.config import Config
from src.dataset import get_dataset
from src.fourier_ablation import (
    _checkpoint_config_or_default,
    _orthonormalize_rows,
    _residual_directions_for_site,
    _site_hook_name,
    build_fourier_basis,
    score_frequencies,
)
from src.model import get_model
from src.surgery import _load_checkpoint_payload, _load_model_state_dict


DEFAULT_MILESTONES = ["plateau", "20", "50", "80", "95", "99"]
DEFAULT_TOP_K = [1, 2, 3, 5, 10, -1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--settings", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--milestones", nargs="+", default=DEFAULT_MILESTONES)
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=DEFAULT_TOP_K,
        help="Frequency counts; -1 means all p-dependent frequencies.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def load_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    with Path(args.checkpoint_manifest).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    jobs = [
        row
        for row in rows
        if row["setting"] in args.settings
        and int(row["seed"]) in args.seeds
        and row["milestone_label"] in args.milestones
    ]
    expected = len(args.settings) * len(args.seeds) * len(args.milestones)
    if len(jobs) != expected:
        raise ValueError(f"Resolved {len(jobs)} checkpoints but expected {expected}.")
    for row in jobs:
        for key in ("seed", "p", "layers", "update"):
            row[key] = int(float(row[key]))
        for key in ("actual_train_accuracy", "actual_test_accuracy"):
            row[key] = float(row[key])
    return jobs


def load_source_model(path: Path) -> tuple[torch.nn.Module, Config]:
    payload = _load_checkpoint_payload(path)
    cfg = _checkpoint_config_or_default(Config(), payload)
    model = get_model(cfg, seed=cfg.train.seed)
    model.load_state_dict(_load_model_state_dict(path))
    model = model.to("cpu")
    model.cfg.device = "cpu"
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, cfg


def whole_subset(tokens: torch.Tensor, labels: torch.Tensor, role: str) -> DataSubset:
    return DataSubset(tokens, labels, torch.arange(len(tokens)), role)


def make_hook(
    selected_rows: torch.Tensor,
    full_rows: torch.Tensor,
    *,
    variant: str,
    mean: torch.Tensor,
):
    def hook(value: torch.Tensor, _hook: Any) -> torch.Tensor:
        selected = value[:, [0], :]
        basis = selected_rows.to(value)
        full = full_rows.to(value)
        selected_reconstruction = (selected @ basis.T) @ basis
        patched = value.clone()
        if variant == "conditional_fourier_tail":
            full_reconstruction = (selected @ full.T) @ full
            patched[:, [0], :] = selected - full_reconstruction + selected_reconstruction
        elif variant == "literal_selected_span":
            patched[:, [0], :] = selected_reconstruction
        elif variant == "mean_preserving_selected_span":
            center = mean.to(value).reshape(1, 1, -1)
            centered = selected - center
            patched[:, [0], :] = center + (centered @ basis.T) @ basis
        else:
            raise ValueError(variant)
        return patched

    return hook


def flatten_metric(
    *,
    job: Dict[str, Any],
    split: str,
    variant: str,
    top_k: int,
    frequencies: Sequence[int],
    metric: Any,
    frequency_basis_checksum: str,
    cache_diagnostics: Dict[str, Any],
    master_seed: int,
) -> Dict[str, Any]:
    return {
        "job_id": f"{job['setting']}-seed{job['seed']}-{job['milestone_label']}",
        "setting": job["setting"],
        "p": job["p"],
        "layers": job["layers"],
        "seed": job["seed"],
        "milestone_label": job["milestone_label"],
        "checkpoint_update": job["update"],
        "checkpoint_actual_train_accuracy": job["actual_train_accuracy"],
        "checkpoint_actual_test_accuracy": job["actual_test_accuracy"],
        "checkpoint_path": job["path"],
        "checkpoint_sha256": job["sha256"],
        "site": "pre_unembed",
        "split": split,
        "variant": variant,
        "top_k": top_k,
        "frequencies": json.dumps(list(frequencies)),
        "frequency_basis_checksum": frequency_basis_checksum,
        "cache_representation_checksum": cache_diagnostics["representation_checksum"],
        "cache_max_abs_logit_error": cache_diagnostics["max_abs_logit_error"],
        "master_seed": master_seed,
        "n_examples": metric.n_examples,
        "accuracy": metric.accuracy,
        "cross_entropy": metric.cross_entropy,
        "confidence": metric.confidence,
        "logit_norm": metric.logit_norm,
    }


def run_job(args: argparse.Namespace, job: Dict[str, Any], output_dir: Path) -> None:
    job_id = f"{job['setting']}-seed{job['seed']}-{job['milestone_label']}"
    job_path = output_dir / "jobs" / job_id / "summary.json"
    if args.resume and job_path.exists():
        print(f"[resume] {job_id}", flush=True)
        return
    started = time.monotonic()
    source, cfg = load_source_model(Path(job["path"]))
    dataset = get_dataset(cfg, data_seed=cfg.train.seed)
    train_tokens, train_labels = dataset.train_data()
    test_tokens, test_labels = dataset.test_data()
    all_tokens = dataset.tokens
    hook_name = _site_hook_name(source, "pre_unembed")
    cached, cache_diagnostics = build_cached_preunembed_model(
        source,
        all_tokens,
        p=cfg.model.p,
        hook_name=hook_name,
        position=cfg.model.n_ctx - 1,
        batch_size=args.eval_batch_size,
    )
    train = whole_subset(train_tokens, train_labels, "train")
    test = whole_subset(test_tokens, test_labels, "test")
    clean = {
        "train": evaluate_subset(cached, train, batch_size=args.eval_batch_size),
        "test": evaluate_subset(cached, test, batch_size=args.eval_batch_size),
    }
    rows: List[Dict[str, Any]] = []
    for split, metric in clean.items():
        rows.append(
            flatten_metric(
                job=job,
                split=split,
                variant="clean",
                top_k=0,
                frequencies=[],
                metric=metric,
                frequency_basis_checksum="",
                cache_diagnostics=cache_diagnostics,
                master_seed=args.seed,
            )
        )
    token_fourier = build_fourier_basis(cfg.model.p)
    residual = _residual_directions_for_site(
        source, cfg.model.p, token_fourier, "pre_unembed"
    )
    ranked = [int(row["frequency"]) for row in score_frequencies(source, cfg.model.p)]
    full_rows = _orthonormalize_rows(torch.cat([residual[f] for f in ranked], dim=0)).float()
    activation_mean = cached.representations.mean(dim=0).float()
    for requested_k in args.top_k:
        k = len(ranked) if requested_k == -1 else requested_k
        if k <= 0 or k > len(ranked):
            raise ValueError(f"Invalid top-k={requested_k} for p={cfg.model.p}.")
        frequencies = ranked[:k]
        selected_rows = _orthonormalize_rows(
            torch.cat([residual[f] for f in frequencies], dim=0)
        ).float()
        basis_checksum = tensor_checksum(selected_rows)
        for variant in (
            "conditional_fourier_tail",
            "literal_selected_span",
            "mean_preserving_selected_span",
        ):
            hook = make_hook(
                selected_rows,
                full_rows,
                variant=variant,
                mean=activation_mean,
            )
            for split, subset in (("train", train), ("test", test)):
                metric = evaluate_subset(
                    cached,
                    subset,
                    batch_size=args.eval_batch_size,
                    hook_name=hook_name,
                    hook_fn=hook,
                )
                rows.append(
                    flatten_metric(
                        job=job,
                        split=split,
                        variant=variant,
                        top_k=k,
                        frequencies=frequencies,
                        metric=metric,
                        frequency_basis_checksum=basis_checksum,
                        cache_diagnostics=cache_diagnostics,
                        master_seed=args.seed,
                    )
                )
    atomic_json(
        job_path,
        {
            "status": "completed",
            "job": job,
            "cache_diagnostics": cache_diagnostics,
            "frequency_ranking": ranked,
            "full_fourier_basis_checksum": tensor_checksum(full_rows),
            "rows": rows,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"[complete] {job_id} runtime={time.monotonic()-started:.2f}s", flush=True)


def aggregate(
    output_dir: Path,
    jobs: Sequence[Dict[str, Any]],
    procedure_checksum: str,
) -> None:
    rows: List[Dict[str, Any]] = []
    completed: List[str] = []
    for path in sorted((output_dir / "jobs").glob("*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
        completed.append(path.parent.name)
    target = output_dir / "fourier_diagnostic.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    expected = [
        f"{job['setting']}-seed{job['seed']}-{job['milestone_label']}"
        for job in jobs
    ]
    completed_set = set(completed)
    unresolved = [job_id for job_id in expected if job_id not in completed_set]
    atomic_json(
        output_dir / "manifest.json",
        {
            "status": "completed" if len(completed) == len(expected) else "incomplete",
            "expected_jobs": expected,
            "completed_jobs": completed,
            "completed_count": len(completed),
            "failed_jobs": unresolved,
            "rows": len(rows),
            "procedure_checksum": procedure_checksum,
        },
    )


def main() -> None:
    args = parse_args()
    jobs = load_jobs(args)
    procedure = {
        "version": 1,
        "checkpoint_manifest": args.checkpoint_manifest,
        "jobs": [
            {key: job[key] for key in ("setting", "seed", "milestone_label", "path", "sha256")}
            for job in jobs
        ],
        "top_k": args.top_k,
        "variants": [
            "conditional_fourier_tail",
            "literal_selected_span",
            "mean_preserving_selected_span",
        ],
        "site": "pre_unembed",
        "eval_batch_size": args.eval_batch_size,
        "master_seed": args.seed,
    }
    print(json.dumps({"procedure": procedure, "checkpoint_jobs": len(jobs)}, indent=2))
    if args.dry_run:
        return
    output_dir = Path(args.output_dir)
    procedure_path = output_dir / "procedure.json"
    if procedure_path.exists() and not args.resume:
        raise FileExistsError("Procedure exists; use --resume or another output directory.")
    if procedure_path.exists():
        existing = json.loads(procedure_path.read_text(encoding="utf-8"))
        if sha256_json(existing) != sha256_json(procedure):
            raise ValueError("Resume procedure does not match the existing procedure.")
    else:
        atomic_json(procedure_path, procedure)
    failures = output_dir / "logs" / "failures.jsonl"
    failed = 0
    for job in jobs:
        try:
            run_job(args, job, output_dir)
        except Exception as exc:
            failed += 1
            failures.parent.mkdir(parents=True, exist_ok=True)
            with failures.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "job": job,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    + "\n"
                )
            print(f"[failed] {job['setting']}-seed{job['seed']}-{job['milestone_label']}: {exc}", flush=True)
    aggregate(output_dir, jobs, sha256_json(procedure))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
