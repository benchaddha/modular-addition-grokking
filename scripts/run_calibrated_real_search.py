#!/usr/bin/env python3
"""Run validated, manifest-driven separability searches on canonical checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cached_preunembed import build_cached_preunembed_model
from src.calibrated_separability import (
    AccessAudit,
    Candidate,
    DataSubset,
    FinalEvaluationGuard,
    MetricSet,
    PairMetrics,
    add_validation_metrics,
    capped_fit_view,
    deterministic_three_way_split,
    evaluate_basis_pair,
    evaluate_selected_on_final,
    optimize_projection_run,
    select_validation_candidate,
    sha256_json,
    stable_right_singular_basis,
    stable_seed,
)
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


DEFAULT_RANKS = [1, 2, 4, 8, 16, 32]
DEFAULT_LAMBDAS = [0.0, 0.3, 1.0, 3.0, 10.0]
DEFAULT_MILESTONES = ["plateau", "05", "20", "50", "80", "95", "99"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--settings", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--milestones", nargs="+", default=DEFAULT_MILESTONES)
    parser.add_argument("--site", default="pre_unembed", choices=["pre_unembed"])
    parser.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    parser.add_argument("--stress-rank", type=int, default=64)
    parser.add_argument("--stress-milestones", nargs="+", default=["20", "50", "80"])
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--objective", choices=["ce", "capped_ce"], default="capped_ce")
    parser.add_argument("--max-fit-examples", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--random-draws", type=int, default=5)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
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


def load_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        row["p"] = int(row["p"])
        row["layers"] = int(row["layers"])
        row["update"] = int(float(row["update"]))
        row["actual_train_accuracy"] = float(row["actual_train_accuracy"])
        row["actual_test_accuracy"] = float(row["actual_test_accuracy"])
    return rows


def procedure_payload(args: argparse.Namespace, jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "version": 1,
        "checkpoint_manifest": str(Path(args.checkpoint_manifest)),
        "checkpoint_jobs": [
            {
                key: job[key]
                for key in ("setting", "seed", "milestone_label", "path", "sha256")
            }
            for job in jobs
        ],
        "site": args.site,
        "ranks": args.ranks,
        "stress_rank": args.stress_rank,
        "stress_milestones": args.stress_milestones,
        "lambdas": args.lambdas,
        "restarts": args.restarts,
        "steps": args.steps,
        "lr": args.lr,
        "record_every": args.record_every,
        "objective": args.objective,
        "max_fit_examples": args.max_fit_examples,
        "eval_batch_size": args.eval_batch_size,
        "random_draws": args.random_draws,
        "master_seed": args.seed,
        "split": "deterministic label-stratified exact 50/25/25",
        "selection": "validation selective gap with prespecified feasibility/ties",
        "final": "one evaluation of each selected method comparison",
        "cached_prefix": "verified exact pre_unembed residual cache",
    }


def load_source_model(checkpoint_path: Path) -> Tuple[torch.nn.Module, Config, Dict[str, Any]]:
    payload = _load_checkpoint_payload(checkpoint_path)
    cfg = _checkpoint_config_or_default(Config(), payload)
    model = get_model(cfg, seed=cfg.train.seed)
    model.load_state_dict(_load_model_state_dict(checkpoint_path))
    model = model.to("cpu")
    model.cfg.device = "cpu"
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, cfg, payload


def make_splits(cfg: Config, args: argparse.Namespace, job: Dict[str, Any]) -> Dict[str, Any]:
    dataset = get_dataset(cfg, data_seed=cfg.train.seed)
    train_tokens, train_labels = dataset.train_data()
    test_tokens, test_labels = dataset.test_data()
    namespace = f"{job['setting']}_seed{job['seed']}"
    train = deterministic_three_way_split(
        train_tokens,
        train_labels,
        seed=args.seed,
        namespace=f"{namespace}_train",
    )
    test = deterministic_three_way_split(
        test_tokens,
        test_labels,
        seed=args.seed,
        namespace=f"{namespace}_test",
    )
    train_opt = capped_fit_view(
        train.fit,
        max_examples=args.max_fit_examples,
        seed=stable_seed(args.seed, namespace, "train-cap"),
    )
    test_opt = capped_fit_view(
        test.fit,
        max_examples=args.max_fit_examples,
        seed=stable_seed(args.seed, namespace, "test-cap"),
    )
    return {
        "dataset": dataset,
        "train": train,
        "test": test,
        "train_opt": train_opt,
        "test_opt": test_opt,
        "checksum": sha256_json(
            {
                "train": train.split_checksum,
                "test": test.split_checksum,
                "train_opt_indices": train_opt.indices.tolist(),
                "test_opt_indices": test_opt.indices.tolist(),
            }
        ),
    }


def fourier_topk_basis(
    model: torch.nn.Module,
    *,
    p: int,
    rank: int,
) -> torch.Tensor:
    fourier = build_fourier_basis(p)
    directions = _residual_directions_for_site(model, p, fourier, "pre_unembed")
    ordered = [directions[int(row["frequency"])] for row in score_frequencies(model, p)]
    blocks: List[torch.Tensor] = []
    count = 0
    for block in ordered:
        blocks.append(block)
        count += int(block.shape[0])
        if count >= rank:
            break
    rows = _orthonormalize_rows(torch.cat(blocks, dim=0))[:rank]
    if int(rows.shape[0]) != rank:
        raise RuntimeError(f"Fourier basis only supplied {rows.shape[0]} directions for rank {rank}.")
    return rows.T.contiguous().float()


def random_basis(d_model: int, rank: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(
        torch.randn(d_model, rank, generator=generator), mode="reduced"
    ).Q.float()


def cached_activations(model: torch.nn.Module, subset: DataSubset) -> torch.Tensor:
    ids = subset.tokens[:, 0].long() * model.p + subset.tokens[:, 1].long()
    return model.representations[ids].detach().cpu().float()


def gradient_matrix(
    model: torch.nn.Module,
    subset: DataSubset,
    *,
    hook_name: str,
    batch_size: int,
) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for start in range(0, len(subset), batch_size):
        end = min(start + batch_size, len(subset))
        captured: Dict[str, torch.Tensor] = {}

        def capture(value: torch.Tensor, hook: Any) -> torch.Tensor:
            value.requires_grad_(True)
            value.retain_grad()
            captured["value"] = value
            return value

        model.zero_grad(set_to_none=True)
        with model.hooks(fwd_hooks=[(hook_name, capture)]):
            logits = model(subset.tokens[start:end])[:, -1, :]
            loss = torch.nn.functional.cross_entropy(
                logits, subset.labels[start:end], reduction="sum"
            )
            loss.backward()
        rows.append(captured["value"].grad[:, 0, :].detach().cpu().float())
    return torch.cat(rows, dim=0)


def evaluate_fixed_candidate(
    model: torch.nn.Module,
    basis: torch.Tensor,
    *,
    candidate_id: str,
    method: str,
    rank: int,
    repeat: int,
    train_fit: DataSubset,
    test_fit: DataSubset,
    train_validation: DataSubset,
    test_validation: DataSubset,
    hook_name: str,
    batch_size: int,
    p: int,
) -> Candidate:
    fit = evaluate_basis_pair(
        model,
        train_fit,
        test_fit,
        hook_name=hook_name,
        positions=[0],
        basis=basis,
        batch_size=batch_size,
        p=p,
    )
    candidate = Candidate(
        candidate_id=candidate_id,
        step=0,
        rank=rank,
        lam=0.0,
        restart=repeat,
        basis=basis,
        fit_metrics=fit,
        method=method,
    )
    add_validation_metrics(
        model,
        [candidate],
        train_validation,
        test_validation,
        hook_name=hook_name,
        positions=[0],
        batch_size=batch_size,
        p=p,
    )
    return candidate


def selected_row(
    *,
    job: Dict[str, Any],
    comparison: str,
    method: str,
    candidate: Optional[Candidate],
    final_metrics: Optional[PairMetrics],
    split_checksum: str,
    cache_diagnostics: Dict[str, Any],
    master_seed: int,
) -> Dict[str, Any]:
    base = {
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
        "comparison": comparison,
        "method": method,
        "split_checksum": split_checksum,
        "master_seed": master_seed,
        "cache_representation_checksum": cache_diagnostics["representation_checksum"],
        "cache_max_abs_logit_error": cache_diagnostics["max_abs_logit_error"],
        "status": "selected" if candidate is not None else "no_feasible_candidate",
    }
    if candidate is None or final_metrics is None:
        return {
            **base,
            "selected_candidate_id": None,
            "rank": None,
            "lambda": None,
            "restart": None,
            "step": None,
            "basis_checksum": None,
        }
    return {
        **base,
        "selected_candidate_id": candidate.candidate_id,
        "rank": candidate.rank,
        "lambda": candidate.lam,
        "restart": candidate.restart,
        "step": candidate.step,
        "basis_checksum": candidate.checksum,
        "fit_train_accuracy": candidate.fit_metrics.intervened_train.accuracy,
        "fit_test_accuracy": candidate.fit_metrics.intervened_test.accuracy,
        "fit_selective_gap": candidate.fit_metrics.selective_gap,
        "validation_clean_train_accuracy": candidate.validation_metrics.clean_train.accuracy,
        "validation_clean_test_accuracy": candidate.validation_metrics.clean_test.accuracy,
        "validation_train_accuracy": candidate.validation_metrics.intervened_train.accuracy,
        "validation_test_accuracy": candidate.validation_metrics.intervened_test.accuracy,
        "validation_train_damage": candidate.validation_metrics.train_damage,
        "validation_test_damage": candidate.validation_metrics.test_damage,
        "validation_selective_gap": candidate.validation_metrics.selective_gap,
        "final_clean_train_accuracy": final_metrics.clean_train.accuracy,
        "final_clean_test_accuracy": final_metrics.clean_test.accuracy,
        "final_train_accuracy": final_metrics.intervened_train.accuracy,
        "final_test_accuracy": final_metrics.intervened_test.accuracy,
        "final_train_damage": final_metrics.train_damage,
        "final_test_damage": final_metrics.test_damage,
        "final_selective_gap": final_metrics.selective_gap,
        "final_relative_train_retention": final_metrics.relative_train_retention,
        "final_relative_test_damage": final_metrics.relative_test_damage,
        "final_attainable_ceiling": final_metrics.attainable_ceiling,
        "final_recovered_ceiling_fraction": final_metrics.recovered_ceiling_fraction,
        "criterion_pass_80": (
            final_metrics.relative_train_retention >= 0.80
            and final_metrics.intervened_test.accuracy <= 2.0 / int(job["p"])
        ),
        "criterion_pass_90": final_metrics.operational_pass,
        "criterion_pass_95": (
            final_metrics.relative_train_retention >= 0.95
            and final_metrics.intervened_test.accuracy <= 2.0 / int(job["p"])
        ),
        "final_cross_entropy_train": final_metrics.intervened_train.cross_entropy,
        "final_cross_entropy_test": final_metrics.intervened_test.cross_entropy,
        "final_confidence_train": final_metrics.intervened_train.confidence,
        "final_confidence_test": final_metrics.intervened_test.confidence,
        "final_logit_norm_train": final_metrics.intervened_train.logit_norm,
        "final_logit_norm_test": final_metrics.intervened_test.logit_norm,
    }


def run_learned_configs(
    args: argparse.Namespace,
    job: Dict[str, Any],
    job_dir: Path,
    model: torch.nn.Module,
    splits: Dict[str, Any],
    hook_name: str,
) -> Tuple[List[Candidate], List[Candidate], List[Dict[str, Any]]]:
    selected: List[Candidate] = []
    capacities: List[Candidate] = []
    records: List[Dict[str, Any]] = []
    ranks = list(args.ranks)
    if job["milestone_label"] in args.stress_milestones:
        ranks.append(args.stress_rank)
    for rank in ranks:
        for lam in args.lambdas:
            for restart in range(args.restarts):
                stem = f"learned_r{rank}_lam{lam:g}_restart{restart}"
                metadata_path = job_dir / "runs" / f"{stem}.json"
                basis_path = job_dir / "runs" / f"{stem}.pt"
                if metadata_path.exists() and basis_path.exists() and args.resume:
                    record = json.loads(metadata_path.read_text(encoding="utf-8"))
                    tensors = torch.load(basis_path, map_location="cpu", weights_only=True)
                    if record["selected"] is not None:
                        selected.append(candidate_from_saved(record["selected"], tensors["selected_basis"]))
                    if record.get("capacity") is not None:
                        capacities.append(candidate_from_saved(record["capacity"], tensors["capacity_basis"]))
                    records.append(record)
                    continue
                optimizer_seed = stable_seed(
                    args.seed,
                    job["sha256"],
                    rank,
                    lam,
                    restart,
                    args.objective,
                )
                audit = AccessAudit()
                started = time.monotonic()
                result = optimize_projection_run(
                    model,
                    splits["train_opt"],
                    splits["test_opt"],
                    splits["train"].fit,
                    splits["test"].fit,
                    train_validation=splits["train"].validation,
                    test_validation=splits["test"].validation,
                    hook_name=hook_name,
                    positions=[0],
                    rank=rank,
                    lam=lam,
                    restart=restart,
                    steps=args.steps,
                    lr=args.lr,
                    seed=optimizer_seed,
                    batch_size=args.eval_batch_size,
                    p=int(job["p"]),
                    record_every=args.record_every,
                    objective=args.objective,
                    access_audit=audit,
                )
                capacity = None
                if lam == 0.0:
                    capacity = Candidate(
                        candidate_id=f"capacity-r{rank}-restart{restart}-step{args.steps-1}",
                        step=args.steps - 1,
                        rank=rank,
                        lam=lam,
                        restart=restart,
                        basis=result.final_basis,
                        fit_metrics=result.final_fit_metrics,
                        method="unconstrained_capacity",
                    )
                    add_validation_metrics(
                        model,
                        [capacity],
                        splits["train"].validation,
                        splits["test"].validation,
                        hook_name=hook_name,
                        positions=[0],
                        batch_size=args.eval_batch_size,
                        p=int(job["p"]),
                    )
                    capacities.append(capacity)
                if result.selected is not None:
                    selected.append(result.selected)
                record = {
                    "rank": rank,
                    "lambda": lam,
                    "restart": restart,
                    "optimizer_seed": optimizer_seed,
                    "runtime_seconds": time.monotonic() - started,
                    "selected": result.selected.metadata() if result.selected is not None else None,
                    "capacity": capacity.metadata() if capacity is not None else None,
                    "candidate_records": [candidate.metadata() for candidate in result.candidates],
                    "access_audit": {
                        "optimizer_roles": result.access_audit.optimizer_roles,
                        "validation_roles": result.access_audit.validation_roles,
                        "final_roles": result.access_audit.final_roles,
                    },
                }
                tensors: Dict[str, torch.Tensor] = {}
                if result.selected is not None:
                    tensors["selected_basis"] = result.selected.basis
                if capacity is not None:
                    tensors["capacity_basis"] = capacity.basis
                atomic_torch_save(basis_path, tensors)
                atomic_json(metadata_path, record)
                records.append(record)
                print(
                    f"  [run] r={rank} lambda={lam:g} restart={restart} "
                    f"feasible={result.selected is not None} runtime={record['runtime_seconds']:.2f}s",
                    flush=True,
                )
    return selected, capacities, records


def run_controls(
    args: argparse.Namespace,
    job: Dict[str, Any],
    model: torch.nn.Module,
    source_model: torch.nn.Module,
    splits: Dict[str, Any],
    hook_name: str,
) -> Tuple[Dict[str, List[Candidate]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    methods: Dict[str, List[Candidate]] = defaultdict(list)
    diagnostics: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    ranks = list(args.ranks)
    if job["milestone_label"] in args.stress_milestones:
        ranks.append(args.stress_rank)

    combined_fit = DataSubset(
        torch.cat([splits["train_opt"].tokens, splits["test_opt"].tokens]),
        torch.cat([splits["train_opt"].labels, splits["test_opt"].labels]),
        torch.arange(len(splits["train_opt"]) + len(splits["test_opt"])),
        "combined_fit_opt",
    )
    activations = cached_activations(model, combined_fit)
    centered = activations - activations.mean(dim=0, keepdim=True)
    gradient = None
    try:
        gradient = gradient_matrix(
            model,
            splits["test_opt"],
            hook_name=hook_name,
            batch_size=args.eval_batch_size,
        )
    except Exception as exc:
        failures.append(
            {"method": "gradient", "rank": None, "error": repr(exc), "traceback": traceback.format_exc()}
        )

    for rank in ranks:
        basis_specs: List[Tuple[str, int, Callable[[], Tuple[torch.Tensor, Dict[str, Any]]]]] = []
        basis_specs.append(
            (
                "fourier",
                0,
                lambda rank=rank: (
                    fourier_topk_basis(source_model, p=int(job["p"]), rank=rank),
                    {"source": "weight-ranked Fourier directions"},
                ),
            )
        )
        for repeat in range(args.random_draws):
            random_seed = stable_seed(
                args.seed, job["sha256"], "random", rank, repeat
            )
            basis_specs.append(
                (
                    "random",
                    repeat,
                    lambda rank=rank, random_seed=random_seed: (
                        random_basis(int(model.cfg.d_model), rank, random_seed),
                        {"random_seed": random_seed},
                    ),
                )
            )
        basis_specs.append(
            (
                "pca",
                0,
                lambda rank=rank: stable_right_singular_basis(centered, rank),
            )
        )
        if gradient is not None:
            basis_specs.append(
                (
                    "gradient",
                    0,
                    lambda rank=rank: stable_right_singular_basis(gradient, rank),
                )
            )

        for method, repeat, builder in basis_specs:
            try:
                basis, diag = builder()
                candidate = evaluate_fixed_candidate(
                    model,
                    basis,
                    candidate_id=f"{method}-r{rank}-repeat{repeat}",
                    method=method,
                    rank=rank,
                    repeat=repeat,
                    train_fit=splits["train"].fit,
                    test_fit=splits["test"].fit,
                    train_validation=splits["train"].validation,
                    test_validation=splits["test"].validation,
                    hook_name=hook_name,
                    batch_size=args.eval_batch_size,
                    p=int(job["p"]),
                )
                methods[method].append(candidate)
                diagnostics.append(
                    {
                        "method": method,
                        "rank": rank,
                        "repeat": repeat,
                        "basis_checksum": candidate.checksum,
                        **diag,
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "method": method,
                        "rank": rank,
                        "repeat": repeat,
                        "matrix_shape": (
                            list(centered.shape)
                            if method == "pca"
                            else list(gradient.shape) if method == "gradient" and gradient is not None else None
                        ),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    return methods, diagnostics, failures


def run_checkpoint(args: argparse.Namespace, job: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    job_id = f"{job['setting']}-seed{job['seed']}-{job['milestone_label']}"
    job_dir = output_dir / "jobs" / job_id
    summary_path = job_dir / "summary.json"
    if summary_path.exists():
        if not args.resume:
            raise FileExistsError(f"{job_id} exists; use --resume or a new output directory.")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    job_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    checkpoint_path = REPO_ROOT / job["path"]
    source_model, cfg, payload = load_source_model(checkpoint_path)
    splits = make_splits(cfg, args, job)
    all_tokens = splits["dataset"].tokens
    hook_name = _site_hook_name(source_model, "pre_unembed")
    cached_model, cache_diagnostics = build_cached_preunembed_model(
        source_model,
        all_tokens,
        p=int(job["p"]),
        hook_name=hook_name,
        position=2,
        batch_size=args.eval_batch_size,
    )
    selected, capacities, run_records = run_learned_configs(
        args, job, job_dir, cached_model, splits, hook_name
    )

    comparison_candidates: List[Tuple[str, str, Optional[Candidate]]] = []
    standard = [candidate for candidate in selected if candidate.rank in args.ranks]
    comparison_candidates.append(
        ("standard", "learned", select_validation_candidate(standard))
    )
    rank32 = [candidate for candidate in selected if candidate.rank == 32]
    comparison_candidates.append(
        ("rank32_only", "learned", select_validation_candidate(rank32))
    )
    if job["milestone_label"] in args.stress_milestones:
        rank64 = [candidate for candidate in selected if candidate.rank == args.stress_rank]
        comparison_candidates.append(
            ("rank64_stress", "learned", select_validation_candidate(rank64))
        )

    standard_capacity = [candidate for candidate in capacities if candidate.rank in args.ranks]
    comparison_candidates.append(
        ("standard", "unconstrained_capacity", capacity_select(standard_capacity))
    )
    if job["milestone_label"] in args.stress_milestones:
        stress_capacity = [candidate for candidate in capacities if candidate.rank == args.stress_rank]
        comparison_candidates.append(
            ("rank64_stress", "unconstrained_capacity", capacity_select(stress_capacity))
        )

    controls: Dict[str, List[Candidate]] = {}
    control_diagnostics: List[Dict[str, Any]] = []
    control_failures: List[Dict[str, Any]] = []
    if not args.skip_controls:
        controls, control_diagnostics, control_failures = run_controls(
            args, job, cached_model, source_model, splits, hook_name
        )
        for method, candidates in controls.items():
            standard_controls = [candidate for candidate in candidates if candidate.rank in args.ranks]
            comparison_candidates.append(
                ("standard", method, select_validation_candidate(standard_controls))
            )
            rank32_controls = [candidate for candidate in candidates if candidate.rank == 32]
            comparison_candidates.append(
                ("rank32_only", method, select_validation_candidate(rank32_controls))
            )
            if job["milestone_label"] in args.stress_milestones:
                stress_controls = [
                    candidate for candidate in candidates if candidate.rank == args.stress_rank
                ]
                comparison_candidates.append(
                    ("rank64_stress", method, select_validation_candidate(stress_controls))
                )

    final_cache: Dict[str, PairMetrics] = {}
    selected_rows: List[Dict[str, Any]] = []
    for comparison, method, candidate in comparison_candidates:
        final_metrics = None
        if candidate is not None:
            if candidate.candidate_id not in final_cache:
                final_cache[candidate.candidate_id] = evaluate_selected_on_final(
                    cached_model,
                    candidate,
                    splits["train"].final,
                    splits["test"].final,
                    hook_name=hook_name,
                    positions=[0],
                    batch_size=args.eval_batch_size,
                    p=int(job["p"]),
                    guard=FinalEvaluationGuard(),
                )
            final_metrics = final_cache[candidate.candidate_id]
        selected_rows.append(
            selected_row(
                job=job,
                comparison=comparison,
                method=method,
                candidate=candidate,
                final_metrics=final_metrics,
                split_checksum=splits["checksum"],
                cache_diagnostics=cache_diagnostics,
                master_seed=args.seed,
            )
        )

    for failure in control_failures:
        selected_rows.append(
            {
                "job_id": job_id,
                "setting": job["setting"],
                "p": job["p"],
                "layers": job["layers"],
                "seed": job["seed"],
                "milestone_label": job["milestone_label"],
                "checkpoint_update": job["update"],
                "comparison": "control_failure",
                "method": failure["method"],
                "rank": failure.get("rank"),
                "status": "failed",
                "failure": failure,
                "split_checksum": splits["checksum"],
            }
        )

    summary = {
        "job": job,
        "split_checksum": splits["checksum"],
        "split_counts": {
            "train_fit": len(splits["train"].fit),
            "train_fit_optimization": len(splits["train_opt"]),
            "train_validation": len(splits["train"].validation),
            "train_final": len(splits["train"].final),
            "test_fit": len(splits["test"].fit),
            "test_fit_optimization": len(splits["test_opt"]),
            "test_validation": len(splits["test"].validation),
            "test_final": len(splits["test"].final),
        },
        "cache_diagnostics": cache_diagnostics,
        "run_records": run_records,
        "control_diagnostics": control_diagnostics,
        "control_failures": control_failures,
        "selected_rows": selected_rows,
        "runtime_seconds": time.monotonic() - started,
    }
    atomic_json(summary_path, summary)
    return summary


def flatten(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten(row))
    temporary.replace(path)


def aggregate(output_dir: Path) -> None:
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "jobs").glob("*/summary.json"))
    ]
    rows = [row for summary in summaries for row in summary["selected_rows"]]
    learned = [
        row
        for row in rows
        if row.get("method") == "learned" and row.get("status") != "failed"
    ]
    controls = [
        row
        for row in rows
        if row.get("method") not in {"learned", "unconstrained_capacity"}
    ]
    capacity = [row for row in rows if row.get("method") == "unconstrained_capacity"]
    write_csv(output_dir / "selected_final_candidates.csv", learned)
    write_csv(output_dir / "control_results.csv", controls)
    write_csv(output_dir / "capacity_control_results.csv", capacity)
    stress = [row for row in rows if row.get("comparison") == "rank64_stress"]
    write_csv(output_dir / "high_rank_stress_results.csv", stress)


def main() -> None:
    args = parse_args()
    rows = load_manifest_rows(Path(args.checkpoint_manifest))
    jobs = [
        row
        for row in rows
        if row["setting"] in args.settings
        and row["seed"] in args.seeds
        and row["milestone_label"] in args.milestones
    ]
    jobs.sort(key=lambda row: (args.settings.index(row["setting"]), row["seed"], args.milestones.index(row["milestone_label"])))
    procedure = procedure_payload(args, jobs)
    print(
        json.dumps(
            {
                "procedure": procedure,
                "checkpoint_jobs": len(jobs),
                "learned_runs": sum(
                    (len(args.ranks) + (1 if job["milestone_label"] in args.stress_milestones else 0))
                    * len(args.lambdas)
                    * args.restarts
                    for job in jobs
                ),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["procedure_checksum"] != sha256_json(procedure):
            raise ValueError("Existing real-search manifest uses a different procedure.")
        if not args.resume:
            raise FileExistsError("Manifest exists; use --resume or a new output directory.")
    else:
        manifest = {
            "version": 1,
            "procedure": procedure,
            "procedure_checksum": sha256_json(procedure),
            "expected_jobs": [
                f"{job['setting']}-seed{job['seed']}-{job['milestone_label']}"
                for job in jobs
            ],
            "completed_jobs": [],
            "failed_jobs": [],
        }
        atomic_json(manifest_path, manifest)
    for job in jobs:
        job_id = f"{job['setting']}-seed{job['seed']}-{job['milestone_label']}"
        if job_id in manifest["completed_jobs"] and args.resume:
            continue
        try:
            summary = run_checkpoint(args, job, output_dir)
            if job_id not in manifest["completed_jobs"]:
                manifest["completed_jobs"].append(job_id)
            manifest["failed_jobs"] = [
                failure for failure in manifest["failed_jobs"] if failure["job_id"] != job_id
            ]
            print(f"[complete] {job_id} runtime={summary['runtime_seconds']:.2f}s", flush=True)
        except Exception as exc:
            failure = {
                "job_id": job_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            manifest["failed_jobs"].append(failure)
            atomic_json(output_dir / "failures" / f"{job_id}.json", failure)
            print(f"[failed] {job_id}: {exc}", file=sys.stderr, flush=True)
        atomic_json(manifest_path, manifest)
        aggregate(output_dir)
    if manifest["failed_jobs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
