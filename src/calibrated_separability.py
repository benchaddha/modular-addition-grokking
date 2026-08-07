"""Shared primitives for calibrated train/test separability experiments.

This module is intentionally separate from the historical adversarial search.
It enforces a fit/validation/final protocol in which final data cannot enter
optimization or candidate selection.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


def stable_seed(*parts: Any) -> int:
    material = ":".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31)


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_checksum(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def basis_checksum(basis: torch.Tensor) -> str:
    return tensor_checksum(basis)


@dataclass
class DataSubset:
    """A named view of examples with original-split indices attached."""

    tokens: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor
    role: str

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def with_labels(self, labels: torch.Tensor) -> "DataSubset":
        if labels.shape != self.labels.shape:
            raise ValueError("Replacement labels must have the same shape.")
        return DataSubset(self.tokens, labels, self.indices, self.role)


@dataclass
class ThreeWaySplit:
    fit: DataSubset
    validation: DataSubset
    final: DataSubset
    split_seed: int
    split_checksum: str
    source_size: int

    def assert_valid(self) -> None:
        groups = [
            set(int(x) for x in self.fit.indices.tolist()),
            set(int(x) for x in self.validation.indices.tolist()),
            set(int(x) for x in self.final.indices.tolist()),
        ]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise AssertionError("Three-way split is not disjoint.")
        union = groups[0] | groups[1] | groups[2]
        if union != set(range(self.source_size)):
            raise AssertionError("Three-way split is not exhaustive.")


def deterministic_three_way_split(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    namespace: str,
    stratify: bool = True,
) -> ThreeWaySplit:
    """Split one original task split into deterministic 50%/25%/25% views."""

    if int(tokens.shape[0]) != int(labels.shape[0]):
        raise ValueError("tokens and labels must have the same first dimension.")
    n_examples = int(tokens.shape[0])
    if n_examples < 4:
        raise ValueError("At least four examples are needed for a three-way split.")
    n_fit = n_examples // 2
    n_validation = n_examples // 4
    split_seed = stable_seed(
        seed, namespace, n_examples, "fit-validation-final-stratified-v2"
    )

    def allocate(class_counts: List[int], target: int, salt: str) -> List[int]:
        total = sum(class_counts)
        raw = [count * target / total for count in class_counts]
        result = [min(count, int(math.floor(value))) for count, value in zip(class_counts, raw)]
        remaining = target - sum(result)
        order = sorted(
            range(len(class_counts)),
            key=lambda index: (
                -(raw[index] - math.floor(raw[index])),
                stable_seed(split_seed, salt, index),
            ),
        )
        while remaining > 0:
            progressed = False
            for index in order:
                if result[index] < class_counts[index]:
                    result[index] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                raise AssertionError("Unable to allocate stratified split counts.")
        return result

    if stratify:
        classes = sorted(int(value) for value in torch.unique(labels).tolist())
        class_indices: List[torch.Tensor] = []
        for class_value in classes:
            indices = torch.nonzero(labels == class_value, as_tuple=False).flatten()
            generator = torch.Generator().manual_seed(
                stable_seed(split_seed, "class", class_value)
            )
            class_indices.append(indices[torch.randperm(len(indices), generator=generator)])
        class_counts = [len(indices) for indices in class_indices]
        fit_counts = allocate(class_counts, n_fit, "fit")
        remaining_counts = [count - fit for count, fit in zip(class_counts, fit_counts)]
        validation_counts = allocate(remaining_counts, n_validation, "validation")
        grouped: Dict[str, List[torch.Tensor]] = {"fit": [], "validation": [], "final": []}
        for indices, fit_count, validation_count in zip(
            class_indices, fit_counts, validation_counts
        ):
            grouped["fit"].append(indices[:fit_count])
            grouped["validation"].append(
                indices[fit_count : fit_count + validation_count]
            )
            grouped["final"].append(indices[fit_count + validation_count :])
        assignments = {}
        for role, pieces in grouped.items():
            combined = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long)
            generator = torch.Generator().manual_seed(stable_seed(split_seed, role, "shuffle"))
            assignments[role] = combined[
                torch.randperm(len(combined), generator=generator)
            ]
    else:
        generator = torch.Generator().manual_seed(split_seed)
        permutation = torch.randperm(n_examples, generator=generator)
        assignments = {
            "fit": permutation[:n_fit],
            "validation": permutation[n_fit : n_fit + n_validation],
            "final": permutation[n_fit + n_validation :],
        }
    checksum = sha256_json(
        {
            "version": 2,
            "namespace": namespace,
            "stratified": stratify,
            "seed": seed,
            "derived_seed": split_seed,
            "source_size": n_examples,
            "fit": assignments["fit"].tolist(),
            "validation": assignments["validation"].tolist(),
            "final": assignments["final"].tolist(),
        }
    )
    result = ThreeWaySplit(
        fit=DataSubset(
            tokens[assignments["fit"]],
            labels[assignments["fit"]],
            assignments["fit"],
            f"{namespace}_fit",
        ),
        validation=DataSubset(
            tokens[assignments["validation"]],
            labels[assignments["validation"]],
            assignments["validation"],
            f"{namespace}_validation",
        ),
        final=DataSubset(
            tokens[assignments["final"]],
            labels[assignments["final"]],
            assignments["final"],
            f"{namespace}_final",
        ),
        split_seed=split_seed,
        split_checksum=checksum,
        source_size=n_examples,
    )
    result.assert_valid()
    return result


def capped_fit_view(subset: DataSubset, *, max_examples: int, seed: int) -> DataSubset:
    """Return a deterministic optimization view without changing split membership."""

    if not subset.role.endswith("_fit"):
        raise ValueError(f"Only fit data may be capped for optimization: {subset.role}")
    if max_examples <= 0 or len(subset) <= max_examples:
        indices = torch.arange(len(subset), dtype=torch.long)
    else:
        generator = torch.Generator().manual_seed(
            stable_seed(seed, subset.role, len(subset), max_examples, "fit-cap-v1")
        )
        indices = torch.randperm(len(subset), generator=generator)[:max_examples]
    return DataSubset(
        subset.tokens[indices],
        subset.labels[indices],
        subset.indices[indices],
        f"{subset.role}_opt",
    )


@dataclass(frozen=True)
class MetricSet:
    n_examples: int
    accuracy: float
    cross_entropy: float
    confidence: float
    logit_norm: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairMetrics:
    clean_train: MetricSet
    clean_test: MetricSet
    intervened_train: MetricSet
    intervened_test: MetricSet
    train_damage: float
    test_damage: float
    selective_gap: float
    relative_train_retention: float
    relative_test_damage: float
    attainable_ceiling: float
    recovered_ceiling_fraction: float
    operational_pass: bool

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


def project_out_hook(basis: torch.Tensor, positions: Sequence[int]) -> Callable:
    target_positions = list(positions)

    def hook_fn(value: torch.Tensor, hook: Any) -> torch.Tensor:
        u = basis.to(device=value.device, dtype=value.dtype)
        patched = value.clone()
        selected = patched[:, target_positions, :]
        patched[:, target_positions, :] = selected - (selected @ u) @ u.T
        return patched

    return hook_fn


@torch.inference_mode()
def evaluate_subset(
    model: torch.nn.Module,
    subset: DataSubset,
    *,
    batch_size: int,
    hook_name: Optional[str] = None,
    hook_fn: Optional[Callable] = None,
) -> MetricSet:
    if len(subset) == 0:
        raise ValueError(f"Cannot evaluate empty subset {subset.role}.")
    model.eval()
    correct = 0
    ce_sum = 0.0
    confidence_sum = 0.0
    logit_norm_sum = 0.0
    hooks = [(hook_name, hook_fn)] if hook_name is not None and hook_fn is not None else []
    with model.hooks(fwd_hooks=hooks):
        for start in range(0, len(subset), batch_size):
            end = min(start + batch_size, len(subset))
            tokens = subset.tokens[start:end].to("cpu")
            labels = subset.labels[start:end].to("cpu")
            logits = model(tokens)[:, -1, :]
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            ce_sum += float(
                torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            )
            probabilities = torch.softmax(logits, dim=-1)
            confidence_sum += float(probabilities.max(dim=-1).values.sum().item())
            logit_norm_sum += float(torch.linalg.vector_norm(logits, dim=-1).sum().item())
    n_examples = len(subset)
    return MetricSet(
        n_examples=n_examples,
        accuracy=correct / n_examples,
        cross_entropy=ce_sum / n_examples,
        confidence=confidence_sum / n_examples,
        logit_norm=logit_norm_sum / n_examples,
    )


def pair_metrics(
    clean_train: MetricSet,
    clean_test: MetricSet,
    intervened_train: MetricSet,
    intervened_test: MetricSet,
    *,
    p: int,
    retention_threshold: float = 0.90,
) -> PairMetrics:
    train_damage = clean_train.accuracy - intervened_train.accuracy
    test_damage = clean_test.accuracy - intervened_test.accuracy
    train_retention = (
        intervened_train.accuracy / clean_train.accuracy
        if clean_train.accuracy > 0
        else float("nan")
    )
    relative_test_damage = (
        test_damage / clean_test.accuracy if clean_test.accuracy > 0 else float("nan")
    )
    ceiling = max(clean_test.accuracy - 2.0 / p, 0.0)
    recovered_fraction = test_damage / ceiling if ceiling > 0 else float("nan")
    return PairMetrics(
        clean_train=clean_train,
        clean_test=clean_test,
        intervened_train=intervened_train,
        intervened_test=intervened_test,
        train_damage=train_damage,
        test_damage=test_damage,
        selective_gap=test_damage - train_damage,
        relative_train_retention=train_retention,
        relative_test_damage=relative_test_damage,
        attainable_ceiling=ceiling,
        recovered_ceiling_fraction=recovered_fraction,
        operational_pass=(
            intervened_train.accuracy >= retention_threshold * clean_train.accuracy
            and intervened_test.accuracy <= 2.0 / p
        ),
    )


def evaluate_basis_pair(
    model: torch.nn.Module,
    train_subset: DataSubset,
    test_subset: DataSubset,
    *,
    hook_name: str,
    positions: Sequence[int],
    basis: Optional[torch.Tensor],
    batch_size: int,
    clean_train: Optional[MetricSet] = None,
    clean_test: Optional[MetricSet] = None,
    p: int,
    retention_threshold: float = 0.90,
) -> PairMetrics:
    if clean_train is None:
        clean_train = evaluate_subset(model, train_subset, batch_size=batch_size)
    if clean_test is None:
        clean_test = evaluate_subset(model, test_subset, batch_size=batch_size)
    hook_fn = None if basis is None else project_out_hook(basis, positions)
    intervened_train = evaluate_subset(
        model,
        train_subset,
        batch_size=batch_size,
        hook_name=hook_name if hook_fn is not None else None,
        hook_fn=hook_fn,
    )
    intervened_test = evaluate_subset(
        model,
        test_subset,
        batch_size=batch_size,
        hook_name=hook_name if hook_fn is not None else None,
        hook_fn=hook_fn,
    )
    return pair_metrics(
        clean_train,
        clean_test,
        intervened_train,
        intervened_test,
        p=p,
        retention_threshold=retention_threshold,
    )


@dataclass
class AccessAudit:
    optimizer_roles: List[str] = field(default_factory=list)
    validation_roles: List[str] = field(default_factory=list)
    final_roles: List[str] = field(default_factory=list)

    def record_optimizer(self, *subsets: DataSubset) -> None:
        for subset in subsets:
            if "_fit" not in subset.role or "_final" in subset.role:
                raise ValueError(f"Optimizer received non-fit data: {subset.role}")
            self.optimizer_roles.append(subset.role)

    def record_validation(self, *subsets: DataSubset) -> None:
        for subset in subsets:
            if not subset.role.endswith("_validation"):
                raise ValueError(f"Selection received non-validation data: {subset.role}")
            self.validation_roles.append(subset.role)

    def record_final(self, *subsets: DataSubset) -> None:
        for subset in subsets:
            if not subset.role.endswith("_final"):
                raise ValueError(f"Final evaluator received non-final data: {subset.role}")
            self.final_roles.append(subset.role)


@dataclass
class Candidate:
    candidate_id: str
    step: int
    rank: int
    lam: float
    restart: int
    basis: torch.Tensor
    fit_metrics: PairMetrics
    validation_metrics: Optional[PairMetrics] = None
    method: str = "learned"

    @property
    def checksum(self) -> str:
        return basis_checksum(self.basis)

    def metadata(self, include_validation: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "step": self.step,
            "rank": self.rank,
            "lambda": self.lam,
            "restart": self.restart,
            "method": self.method,
            "basis_checksum": self.checksum,
            "fit_metrics": self.fit_metrics.to_dict(),
        }
        if include_validation and self.validation_metrics is not None:
            payload["validation_metrics"] = self.validation_metrics.to_dict()
        return payload


@dataclass
class SearchRunResult:
    candidates: List[Candidate]
    selected: Optional[Candidate]
    final_basis: torch.Tensor
    final_fit_metrics: PairMetrics
    optimizer_seed: int
    access_audit: AccessAudit


def _paired_cross_entropy_losses(
    model: torch.nn.Module,
    train_fit: DataSubset,
    test_fit: DataSubset,
    *,
    hook_name: str,
    basis: torch.Tensor,
    positions: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_train = len(train_fit)
    tokens = torch.cat([train_fit.tokens, test_fit.tokens], dim=0).to("cpu")
    labels = torch.cat([train_fit.labels, test_fit.labels], dim=0).to("cpu")
    with model.hooks(fwd_hooks=[(hook_name, project_out_hook(basis, positions))]):
        logits = model(tokens)[:, -1, :]
    return (
        torch.nn.functional.cross_entropy(
            logits[:n_train], labels[:n_train], reduction="none"
        ),
        torch.nn.functional.cross_entropy(
            logits[n_train:], labels[n_train:], reduction="none"
        ),
    )


def candidate_steps(steps: int, record_every: int = 10) -> List[int]:
    if steps <= 0 or record_every <= 0:
        raise ValueError("steps and record_every must be positive.")
    result = list(range(0, steps, record_every))
    if steps - 1 not in result:
        result.append(steps - 1)
    return result


def select_validation_candidate(
    candidates: Iterable[Candidate],
    *,
    retention_threshold: float = 0.90,
) -> Optional[Candidate]:
    """Apply the prespecified validation rule and deterministic tie breaks."""

    feasible: List[Candidate] = []
    for candidate in candidates:
        metrics = candidate.validation_metrics
        if metrics is None:
            raise ValueError(f"Candidate lacks validation metrics: {candidate.candidate_id}")
        if (
            metrics.intervened_train.accuracy
            >= retention_threshold * metrics.clean_train.accuracy
        ):
            feasible.append(candidate)
    if not feasible:
        return None
    feasible.sort(
        key=lambda candidate: (
            -candidate.validation_metrics.selective_gap,  # type: ignore[union-attr]
            -candidate.validation_metrics.test_damage,  # type: ignore[union-attr]
            candidate.validation_metrics.train_damage,  # type: ignore[union-attr]
            candidate.step,
            candidate.candidate_id,
        )
    )
    return feasible[0]


def add_validation_metrics(
    model: torch.nn.Module,
    candidates: Sequence[Candidate],
    train_validation: DataSubset,
    test_validation: DataSubset,
    *,
    hook_name: str,
    positions: Sequence[int],
    batch_size: int,
    p: int,
    retention_threshold: float = 0.90,
    access_audit: Optional[AccessAudit] = None,
) -> None:
    if access_audit is not None:
        access_audit.record_validation(train_validation, test_validation)
    clean_train = evaluate_subset(model, train_validation, batch_size=batch_size)
    clean_test = evaluate_subset(model, test_validation, batch_size=batch_size)
    for candidate in candidates:
        candidate.validation_metrics = evaluate_basis_pair(
            model,
            train_validation,
            test_validation,
            hook_name=hook_name,
            positions=positions,
            basis=candidate.basis,
            batch_size=batch_size,
            clean_train=clean_train,
            clean_test=clean_test,
            p=p,
            retention_threshold=retention_threshold,
        )


def optimize_projection_run(
    model: torch.nn.Module,
    train_fit_optimization: DataSubset,
    test_fit_optimization: DataSubset,
    train_fit_reporting: DataSubset,
    test_fit_reporting: DataSubset,
    *,
    train_validation: DataSubset,
    test_validation: DataSubset,
    hook_name: str,
    positions: Sequence[int],
    rank: int,
    lam: float,
    restart: int,
    steps: int,
    lr: float,
    seed: int,
    batch_size: int,
    p: int,
    record_every: int = 10,
    retention_threshold: float = 0.90,
    objective: str = "ce",
    access_audit: Optional[AccessAudit] = None,
) -> SearchRunResult:
    """Fit one rank/lambda/restart run and select its winner on validation."""

    audit = access_audit if access_audit is not None else AccessAudit()
    audit.record_optimizer(
        train_fit_optimization,
        test_fit_optimization,
        train_fit_reporting,
        test_fit_reporting,
    )
    if rank <= 0 or rank > int(model.cfg.d_model):
        raise ValueError(f"Invalid rank {rank} for d_model={model.cfg.d_model}.")
    if objective not in {"ce", "capped_ce"}:
        raise ValueError(f"Unsupported objective: {objective}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    clean_fit_train = evaluate_subset(model, train_fit_reporting, batch_size=batch_size)
    clean_fit_test = evaluate_subset(model, test_fit_reporting, batch_size=batch_size)
    clean_opt_train = evaluate_subset(model, train_fit_optimization, batch_size=batch_size)

    generator = torch.Generator().manual_seed(seed)
    v = torch.randn(
        int(model.cfg.d_model), rank, generator=generator, requires_grad=True
    )
    optimizer = torch.optim.Adam([v], lr=lr)
    record_at = set(candidate_steps(steps, record_every))
    candidates: List[Candidate] = []

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        basis = torch.linalg.qr(v, mode="reduced").Q
        train_ce_losses, test_ce_losses = _paired_cross_entropy_losses(
            model,
            train_fit_optimization,
            test_fit_optimization,
            hook_name=hook_name,
            basis=basis,
            positions=positions,
        )
        train_ce = train_ce_losses.mean()
        if objective == "ce":
            test_score = test_ce_losses.mean()
        else:
            # The historical mean CE objective can concentrate damage on a
            # small subset of fit examples. Capping each example at uniform
            # prediction loss makes the attack continue searching after one
            # example/class has already been reduced to chance.
            cap = math.log(p)
            test_score = torch.minimum(
                test_ce_losses,
                torch.full_like(test_ce_losses, cap),
            ).mean()
        loss = -test_score + lam * torch.relu(
            train_ce
            - torch.tensor(clean_opt_train.cross_entropy, dtype=train_ce.dtype)
        )
        loss.backward()
        optimizer.step()

        if step in record_at:
            with torch.no_grad():
                recorded_basis = torch.linalg.qr(v, mode="reduced").Q.detach().clone()
            fit_metrics = evaluate_basis_pair(
                model,
                train_fit_reporting,
                test_fit_reporting,
                hook_name=hook_name,
                positions=positions,
                basis=recorded_basis,
                batch_size=batch_size,
                clean_train=clean_fit_train,
                clean_test=clean_fit_test,
                p=p,
                retention_threshold=retention_threshold,
            )
            candidates.append(
                Candidate(
                    candidate_id=(
                        f"learned-r{rank}-lam{lam:g}-restart{restart}-step{step}"
                    ),
                    step=step,
                    rank=rank,
                    lam=lam,
                    restart=restart,
                    basis=recorded_basis,
                    fit_metrics=fit_metrics,
                )
            )

    with torch.no_grad():
        final_basis = torch.linalg.qr(v, mode="reduced").Q.detach().clone()
    final_fit_metrics = evaluate_basis_pair(
        model,
        train_fit_reporting,
        test_fit_reporting,
        hook_name=hook_name,
        positions=positions,
        basis=final_basis,
        batch_size=batch_size,
        clean_train=clean_fit_train,
        clean_test=clean_fit_test,
        p=p,
        retention_threshold=retention_threshold,
    )
    add_validation_metrics(
        model,
        candidates,
        train_validation,
        test_validation,
        hook_name=hook_name,
        positions=positions,
        batch_size=batch_size,
        p=p,
        retention_threshold=retention_threshold,
        access_audit=audit,
    )
    selected = select_validation_candidate(
        candidates, retention_threshold=retention_threshold
    )
    return SearchRunResult(
        candidates=candidates,
        selected=selected,
        final_basis=final_basis,
        final_fit_metrics=final_fit_metrics,
        optimizer_seed=seed,
        access_audit=audit,
    )


@dataclass
class FinalEvaluationGuard:
    evaluated_candidate_id: Optional[str] = None
    calls: int = 0

    def claim(self, candidate_id: str) -> None:
        if self.calls != 0:
            raise RuntimeError(
                "Final intervention data may be evaluated exactly once per comparison."
            )
        self.evaluated_candidate_id = candidate_id
        self.calls = 1


def evaluate_selected_on_final(
    model: torch.nn.Module,
    selected: Candidate,
    train_final: DataSubset,
    test_final: DataSubset,
    *,
    hook_name: str,
    positions: Sequence[int],
    batch_size: int,
    p: int,
    retention_threshold: float = 0.90,
    guard: Optional[FinalEvaluationGuard] = None,
    access_audit: Optional[AccessAudit] = None,
) -> PairMetrics:
    final_guard = guard if guard is not None else FinalEvaluationGuard()
    final_guard.claim(selected.candidate_id)
    if access_audit is not None:
        access_audit.record_final(train_final, test_final)
    return evaluate_basis_pair(
        model,
        train_final,
        test_final,
        hook_name=hook_name,
        positions=positions,
        basis=selected.basis,
        batch_size=batch_size,
        p=p,
        retention_threshold=retention_threshold,
    )


def projection_overlap(learned: torch.Tensor, truth: torch.Tensor) -> Dict[str, Any]:
    """Return singular-value/principal-angle overlap between two column bases."""

    learned_q = torch.linalg.qr(learned.double(), mode="reduced").Q
    truth_q = torch.linalg.qr(truth.double(), mode="reduced").Q
    singular_values = torch.linalg.svdvals(learned_q.T @ truth_q).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.acos(singular_values))
    truth_fraction = float((singular_values.square().sum() / truth_q.shape[1]).item())
    learned_fraction = float((singular_values.square().sum() / learned_q.shape[1]).item())
    return {
        "singular_values": [float(value) for value in singular_values.tolist()],
        "principal_angles_degrees": [float(value) for value in angles.tolist()],
        "truth_projection_overlap": truth_fraction,
        "learned_projection_overlap": learned_fraction,
    }


def stable_right_singular_basis(
    matrix: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute a deterministic SVD basis with a documented CPU-float64 retry."""

    diagnostics: Dict[str, Any] = {
        "shape": list(matrix.shape),
        "input_dtype": str(matrix.dtype),
        "fallback_used": False,
    }
    try:
        singular_values = torch.linalg.svdvals(matrix.float().cpu())
        _, _, vh = torch.linalg.svd(matrix.float().cpu(), full_matrices=False)
    except RuntimeError as exc:
        diagnostics["fallback_used"] = True
        diagnostics["primary_error"] = repr(exc)
        singular_values = torch.linalg.svdvals(matrix.double().cpu())
        _, _, vh = torch.linalg.svd(matrix.double().cpu(), full_matrices=False)
    finite = singular_values[torch.isfinite(singular_values)]
    diagnostics["largest_singular_value"] = (
        float(finite.max().item()) if finite.numel() else None
    )
    diagnostics["smallest_singular_value"] = (
        float(finite.min().item()) if finite.numel() else None
    )
    diagnostics["condition_estimate"] = (
        float((finite.max() / finite.min()).item())
        if finite.numel() and float(finite.min().item()) > 0
        else None
    )
    return vh[:rank].T.contiguous().float(), diagnostics
