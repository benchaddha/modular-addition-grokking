#!/usr/bin/env python3
"""Run manifest-driven synthetic separability calibration."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibrated_separability import (
    AccessAudit,
    Candidate,
    DataSubset,
    FinalEvaluationGuard,
    MetricSet,
    PairMetrics,
    add_validation_metrics,
    basis_checksum,
    deterministic_three_way_split,
    evaluate_basis_pair,
    evaluate_selected_on_final,
    optimize_projection_run,
    projection_overlap,
    select_validation_candidate,
    sha256_json,
    stable_seed,
    tensor_checksum,
)
from src.synthetic_separability import SYNTHETIC_HOOK_NAME, build_synthetic_model


DEFAULT_P = [5, 11, 17, 31]
DEFAULT_THETA = [90.0, 75.0, 60.0, 45.0, 30.0, 15.0]
DEFAULT_LAMBDAS = [0.0, 0.3, 1.0, 3.0, 10.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["development", "evaluation"], required=True)
    parser.add_argument("--p", nargs="+", type=int, default=DEFAULT_P)
    parser.add_argument("--theta", nargs="+", type=float, default=DEFAULT_THETA)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=None,
        help="Explicit common ranks. Default is {q//2,q,min(2q,64)} per p.",
    )
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--objective", choices=["ce", "capped_ce"], default="ce")
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--s-general", type=float, default=1.0)
    parser.add_argument("--s-memory", type=float, default=1.0)
    parser.add_argument("--readout-scale", type=float, default=8.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lock-manifest", default=None)
    parser.add_argument("--write-lock", default=None)
    parser.add_argument("--skip-negative-control", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def ranks_for_p(args: argparse.Namespace, p: int) -> List[int]:
    q = p - 1
    ranks = args.ranks if args.ranks is not None else [max(1, q // 2), q, min(2 * q, 64)]
    return sorted(set(rank for rank in ranks if 0 < rank <= args.d_model))


def procedure_payload(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "version": 1,
        "p": sorted(args.p),
        "theta": sorted(args.theta, reverse=True),
        "rank_rule": "explicit" if args.ranks is not None else "{q//2,q,min(2q,64)}",
        "explicit_ranks": args.ranks,
        "ranks_by_p": {str(p): ranks_for_p(args, p) for p in sorted(args.p)},
        "lambdas": args.lambdas,
        "restarts": args.restarts,
        "steps": args.steps,
        "lr": args.lr,
        "record_every": args.record_every,
        "d_model": args.d_model,
        "noise_std": args.noise_std,
        "s_general": args.s_general,
        "s_memory": args.s_memory,
        "readout_scale": args.readout_scale,
        "negative_control": not args.skip_negative_control,
        "master_seed": args.seed,
        "optimizer": "Adam",
        "objective": args.objective,
        "parameterization": "differentiated reduced QR",
        "selection_rule": (
            "validation max selective gap among >=90% clean-train-validation "
            "retention; ties test damage, train damage, earlier step"
        ),
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def metric_from_dict(payload: Dict[str, Any]) -> MetricSet:
    return MetricSet(**payload)


def pair_from_dict(payload: Dict[str, Any]) -> PairMetrics:
    copy = dict(payload)
    for key in ("clean_train", "clean_test", "intervened_train", "intervened_test"):
        copy[key] = metric_from_dict(copy[key])
    return PairMetrics(**copy)


def candidate_from_saved(metadata: Dict[str, Any], basis: torch.Tensor) -> Candidate:
    return Candidate(
        candidate_id=metadata["candidate_id"],
        step=int(metadata["step"]),
        rank=int(metadata["rank"]),
        lam=float(metadata["lambda"]),
        restart=int(metadata["restart"]),
        basis=basis,
        fit_metrics=pair_from_dict(metadata["fit_metrics"]),
        validation_metrics=pair_from_dict(metadata["validation_metrics"]),
        method=metadata.get("method", "learned"),
    )


def capacity_select(candidates: Sequence[Candidate]) -> Optional[Candidate]:
    if not candidates:
        return None
    # Capacity asks whether both splits can be damaged. Select without a train
    # retention constraint by maximin validation damage, then total damage.
    return sorted(
        candidates,
        key=lambda candidate: (
            -min(
                candidate.validation_metrics.train_damage,
                candidate.validation_metrics.test_damage,
            ),
            -(
                candidate.validation_metrics.train_damage
                + candidate.validation_metrics.test_damage
            ),
            candidate.rank,
            candidate.restart,
            candidate.candidate_id,
        ),
    )[0]


def make_subsets(metadata: Dict[str, Any], master_seed: int) -> Dict[str, Any]:
    tokens = metadata["tokens"]
    labels = metadata["labels"]
    train_indices = metadata["train_indices"]
    test_indices = metadata["test_indices"]
    train_tokens, train_labels = tokens[train_indices], labels[train_indices]
    test_tokens, test_labels = tokens[test_indices], labels[test_indices]
    train_split = deterministic_three_way_split(
        train_tokens,
        train_labels,
        seed=master_seed,
        namespace="synthetic_train",
    )
    test_split = deterministic_three_way_split(
        test_tokens,
        test_labels,
        seed=master_seed,
        namespace="synthetic_test",
    )
    return {
        "train": train_split,
        "test": test_split,
        "checksum": sha256_json(
            {
                "train": train_split.split_checksum,
                "test": test_split.split_checksum,
            }
        ),
    }


def final_row(
    *,
    job: Dict[str, Any],
    rank: Optional[int],
    selected: Optional[Candidate],
    final_metrics: Optional[PairMetrics],
    oracle_metrics: PairMetrics,
    a_basis: torch.Tensor,
    split_checksum: str,
    globally_selected: bool,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        **job,
        "rank": rank,
        "feasible_validation_candidate": selected is not None,
        "globally_selected": globally_selected,
        "split_checksum": split_checksum,
        "oracle_operational_pass": oracle_metrics.operational_pass,
        "oracle_train_damage": oracle_metrics.train_damage,
        "oracle_test_damage": oracle_metrics.test_damage,
        "oracle_selective_gap": oracle_metrics.selective_gap,
    }
    if selected is None or final_metrics is None:
        row.update(
            {
                "selected_candidate_id": None,
                "selected_lambda": None,
                "selected_restart": None,
                "selected_step": None,
                "basis_checksum": None,
                "final_operational_pass": False,
                "final_train_damage": None,
                "final_test_damage": None,
                "final_selective_gap": None,
                "final_relative_train_retention": None,
                "final_relative_test_damage": None,
                "final_recovered_ceiling_fraction": None,
                "oracle_gap_fraction_recovered": None,
                "truth_projection_overlap": None,
                "learned_projection_overlap": None,
                "principal_angles_degrees": None,
                "fit_metrics": None,
                "validation_metrics": None,
                "final_metrics": None,
            }
        )
        return row
    overlap = projection_overlap(selected.basis, a_basis)
    row.update(
        {
            "selected_candidate_id": selected.candidate_id,
            "selected_lambda": selected.lam,
            "selected_restart": selected.restart,
            "selected_step": selected.step,
            "basis_checksum": selected.checksum,
            "final_operational_pass": final_metrics.operational_pass,
            "final_train_damage": final_metrics.train_damage,
            "final_test_damage": final_metrics.test_damage,
            "final_selective_gap": final_metrics.selective_gap,
            "final_relative_train_retention": final_metrics.relative_train_retention,
            "final_relative_test_damage": final_metrics.relative_test_damage,
            "final_recovered_ceiling_fraction": final_metrics.recovered_ceiling_fraction,
            "oracle_gap_fraction_recovered": (
                final_metrics.selective_gap / oracle_metrics.selective_gap
                if oracle_metrics.selective_gap != 0
                else None
            ),
            "truth_projection_overlap": overlap["truth_projection_overlap"],
            "learned_projection_overlap": overlap["learned_projection_overlap"],
            "principal_angles_degrees": overlap["principal_angles_degrees"],
            "fit_metrics": selected.fit_metrics.to_dict(),
            "validation_metrics": selected.validation_metrics.to_dict(),
            "final_metrics": final_metrics.to_dict(),
        }
    )
    return row


def run_job(args: argparse.Namespace, job: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    started = time.monotonic()
    job_id = job["job_id"]
    job_dir = output_dir / "jobs" / job_id
    summary_path = job_dir / "summary.json"
    if summary_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"Output exists for {job_id}; use --resume or a new output directory."
            )
        return json.loads(summary_path.read_text(encoding="utf-8"))
    job_dir.mkdir(parents=True, exist_ok=True)

    construction_seed = stable_seed(
        args.seed,
        args.mode,
        job["control"],
        job["p"],
        job["theta_degrees"],
        job["seed"],
        "construction",
    )
    model, metadata = build_synthetic_model(
        p=job["p"],
        theta_degrees=job["theta_degrees"],
        data_seed=job["seed"],
        construction_seed=construction_seed,
        d_model=args.d_model,
        s_general=args.s_general,
        s_memory=args.s_memory,
        readout_scale=args.readout_scale,
        noise_std=args.noise_std,
        negative_control=job["control"] == "negative",
    )
    subsets = make_subsets(metadata, args.seed)
    train_split = subsets["train"]
    test_split = subsets["test"]
    p = int(job["p"])
    positions = [0]

    oracle_metrics = evaluate_basis_pair(
        model,
        train_split.final,
        test_split.final,
        hook_name=SYNTHETIC_HOOK_NAME,
        positions=positions,
        basis=metadata["A"],
        batch_size=args.eval_batch_size,
        p=p,
    )
    oracle_row = {
        **job,
        "split_checksum": subsets["checksum"],
        "basis_checksum": basis_checksum(metadata["A"]),
        "clean_train_accuracy": oracle_metrics.clean_train.accuracy,
        "clean_test_accuracy": oracle_metrics.clean_test.accuracy,
        "oracle_train_accuracy": oracle_metrics.intervened_train.accuracy,
        "oracle_test_accuracy": oracle_metrics.intervened_test.accuracy,
        "oracle_train_damage": oracle_metrics.train_damage,
        "oracle_test_damage": oracle_metrics.test_damage,
        "oracle_selective_gap": oracle_metrics.selective_gap,
        "oracle_operational_pass": oracle_metrics.operational_pass,
        "oracle_metrics": oracle_metrics.to_dict(),
    }

    selected_by_rank: Dict[int, List[Candidate]] = {
        rank: [] for rank in ranks_for_p(args, p)
    }
    capacity_candidates: List[Candidate] = []
    run_records: List[Dict[str, Any]] = []
    for rank in ranks_for_p(args, p):
        for lam in args.lambdas:
            for restart in range(args.restarts):
                optimizer_seed = stable_seed(
                    args.seed,
                    args.mode,
                    job_id,
                    rank,
                    lam,
                    restart,
                    "optimizer",
                )
                audit = AccessAudit()
                result = optimize_projection_run(
                    model,
                    train_split.fit,
                    test_split.fit,
                    train_split.fit,
                    test_split.fit,
                    train_validation=train_split.validation,
                    test_validation=test_split.validation,
                    hook_name=SYNTHETIC_HOOK_NAME,
                    positions=positions,
                    rank=rank,
                    lam=lam,
                    restart=restart,
                    steps=args.steps,
                    lr=args.lr,
                    seed=optimizer_seed,
                    batch_size=args.eval_batch_size,
                    p=p,
                    record_every=args.record_every,
                    objective=args.objective,
                    access_audit=audit,
                )
                record = {
                    "rank": rank,
                    "lambda": lam,
                    "restart": restart,
                    "optimizer_seed": optimizer_seed,
                    "feasible": result.selected is not None,
                    "selected": (
                        result.selected.metadata() if result.selected is not None else None
                    ),
                    "candidates": [candidate.metadata() for candidate in result.candidates],
                    "access_audit": {
                        "optimizer_roles": result.access_audit.optimizer_roles,
                        "validation_roles": result.access_audit.validation_roles,
                        "final_roles": result.access_audit.final_roles,
                    },
                }
                if result.selected is not None:
                    selected_by_rank[rank].append(result.selected)

                if lam == 0.0:
                    capacity = Candidate(
                        candidate_id=f"capacity-r{rank}-restart{restart}-step{args.steps-1}",
                        step=args.steps - 1,
                        rank=rank,
                        lam=lam,
                        restart=restart,
                        basis=result.final_basis,
                        fit_metrics=result.final_fit_metrics,
                        method="capacity",
                    )
                    add_validation_metrics(
                        model,
                        [capacity],
                        train_split.validation,
                        test_split.validation,
                        hook_name=SYNTHETIC_HOOK_NAME,
                        positions=positions,
                        batch_size=args.eval_batch_size,
                        p=p,
                    )
                    capacity_candidates.append(capacity)
                    record["capacity_final"] = capacity.metadata()
                run_records.append(record)

    rank_winners: Dict[int, Optional[Candidate]] = {}
    final_cache: Dict[str, PairMetrics] = {}
    learned_rows: List[Dict[str, Any]] = []
    for rank, candidates in selected_by_rank.items():
        winner = select_validation_candidate(candidates)
        rank_winners[rank] = winner
        final_metrics: Optional[PairMetrics] = None
        if winner is not None:
            audit = AccessAudit()
            final_metrics = evaluate_selected_on_final(
                model,
                winner,
                train_split.final,
                test_split.final,
                hook_name=SYNTHETIC_HOOK_NAME,
                positions=positions,
                batch_size=args.eval_batch_size,
                p=p,
                guard=FinalEvaluationGuard(),
                access_audit=audit,
            )
            final_cache[winner.candidate_id] = final_metrics
        learned_rows.append(
            final_row(
                job=job,
                rank=rank,
                selected=winner,
                final_metrics=final_metrics,
                oracle_metrics=oracle_metrics,
                a_basis=metadata["A"],
                split_checksum=subsets["checksum"],
                globally_selected=False,
            )
        )

    global_winner = select_validation_candidate(
        [winner for winner in rank_winners.values() if winner is not None]
    )
    if global_winner is not None:
        for row in learned_rows:
            if row["selected_candidate_id"] == global_winner.candidate_id:
                row["globally_selected"] = True

    capacity_winner = capacity_select(capacity_candidates)
    capacity_metrics: Optional[PairMetrics] = None
    if capacity_winner is not None:
        capacity_metrics = evaluate_selected_on_final(
            model,
            capacity_winner,
            train_split.final,
            test_split.final,
            hook_name=SYNTHETIC_HOOK_NAME,
            positions=positions,
            batch_size=args.eval_batch_size,
            p=p,
            guard=FinalEvaluationGuard(),
        )
    control_row = {
        **job,
        "method": "unconstrained_capacity",
        "split_checksum": subsets["checksum"],
        "selected": capacity_winner.metadata() if capacity_winner is not None else None,
        "final_metrics": capacity_metrics.to_dict() if capacity_metrics is not None else None,
        "substantially_damages_train": (
            capacity_metrics.relative_train_retention <= 0.5
            if capacity_metrics is not None
            else False
        ),
        "substantially_damages_test": (
            capacity_metrics.relative_test_damage >= 0.5
            if capacity_metrics is not None
            else False
        ),
    }
    control_row["substantially_damages_both"] = (
        control_row["substantially_damages_train"]
        and control_row["substantially_damages_test"]
    )

    summary = {
        "job": job,
        "construction_seed": construction_seed,
        "split_checksum": subsets["checksum"],
        "train_split_checksum": train_split.split_checksum,
        "test_split_checksum": test_split.split_checksum,
        "A_checksum": tensor_checksum(metadata["A"]),
        "B_perp_checksum": tensor_checksum(metadata["B_perp"]),
        "B_theta_checksum": tensor_checksum(metadata["B_theta"]),
        "oracle": oracle_row,
        "learned": learned_rows,
        "capacity_control": control_row,
        "run_records": run_records,
        "runtime_seconds": time.monotonic() - started,
    }
    atomic_json(summary_path, summary)
    return summary


def flatten_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: (
            json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list))
            else value
        )
        for key, value in row.items()
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_for_csv(row))
    temporary.replace(path)


def aggregate(output_dir: Path) -> None:
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "jobs").glob("*/summary.json"))
    ]
    write_csv(output_dir / "synthetic_oracle_results.csv", [s["oracle"] for s in summaries])
    write_csv(
        output_dir / "synthetic_learned_results.csv",
        [row for summary in summaries for row in summary["learned"]],
    )
    write_csv(
        output_dir / "control_results.csv",
        [summary["capacity_control"] for summary in summaries],
    )


def expected_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for p in args.p:
        for theta in args.theta:
            for seed in args.seeds:
                jobs.append(
                    {
                        "job_id": f"positive_p{p}_theta{theta:g}_seed{seed}",
                        "mode": args.mode,
                        "control": "positive",
                        "p": p,
                        "theta_degrees": theta,
                        "seed": seed,
                        "true_rank": p - 1,
                    }
                )
        if not args.skip_negative_control:
            for seed in args.seeds:
                jobs.append(
                    {
                        "job_id": f"negative_p{p}_seed{seed}",
                        "mode": args.mode,
                        "control": "negative",
                        "p": p,
                        "theta_degrees": 90.0,
                        "seed": seed,
                        "true_rank": p - 1,
                    }
                )
    return jobs


def validate_lock(args: argparse.Namespace, procedure: Dict[str, Any]) -> None:
    if args.mode == "evaluation":
        if args.lock_manifest is None:
            raise ValueError("Evaluation mode requires --lock-manifest.")
        path = Path(args.lock_manifest)
        if not path.exists():
            raise FileNotFoundError(f"Procedure lock not found: {path}")
        locked = json.loads(path.read_text(encoding="utf-8"))
        if locked["procedure_checksum"] != sha256_json(procedure):
            raise ValueError("Evaluation procedure differs from the frozen lock manifest.")


def main() -> None:
    args = parse_args()
    procedure = procedure_payload(args)
    validate_lock(args, procedure)
    jobs = expected_jobs(args)
    print(json.dumps({"procedure": procedure, "jobs": jobs}, indent=2))
    if args.dry_run:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["procedure_checksum"] != sha256_json(procedure):
            raise ValueError("Existing output manifest has a different procedure.")
        if not args.resume:
            raise FileExistsError("Manifest exists; use --resume or a new output directory.")
    else:
        manifest = {
            "version": 1,
            "mode": args.mode,
            "procedure": procedure,
            "procedure_checksum": sha256_json(procedure),
            "expected_jobs": jobs,
            "completed_jobs": [],
            "failed_jobs": [],
        }
        atomic_json(manifest_path, manifest)

    for job in jobs:
        if job["job_id"] in manifest["completed_jobs"] and args.resume:
            continue
        try:
            summary = run_job(args, job, output_dir)
            if job["job_id"] not in manifest["completed_jobs"]:
                manifest["completed_jobs"].append(job["job_id"])
            manifest["failed_jobs"] = [
                failure
                for failure in manifest["failed_jobs"]
                if failure.get("job_id") != job["job_id"]
            ]
            print(
                f"[complete] {job['job_id']} runtime={summary['runtime_seconds']:.2f}s",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "job_id": job["job_id"],
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            manifest["failed_jobs"].append(failure)
            failure_dir = output_dir / "failures"
            failure_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(failure_dir / f"{job['job_id']}.json", failure)
            print(f"[failed] {job['job_id']}: {exc}", file=sys.stderr, flush=True)
        atomic_json(manifest_path, manifest)
        aggregate(output_dir)

    if args.write_lock is not None:
        if args.mode != "development":
            raise ValueError("--write-lock is only valid in development mode.")
        lock_path = Path(args.write_lock)
        atomic_json(
            lock_path,
            {
                "version": 1,
                "created_from": str(manifest_path),
                "procedure": procedure,
                "procedure_checksum": sha256_json(procedure),
            },
        )
    if manifest["failed_jobs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
