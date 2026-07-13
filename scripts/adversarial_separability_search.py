import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config
from src.dataset import get_dataset
from src.fourier_ablation import (
    _checkpoint_config_or_default,
    _orthonormalize_rows,
    _residual_directions_for_site,
    _site_hook_name,
    _site_positions,
    build_fourier_basis,
    score_frequencies,
)
from src.model import get_model
from src.surgery import _load_checkpoint_payload, _load_model_state_dict


DEFAULT_RANKS = [1, 2, 4, 8, 16, 32]
DEFAULT_LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
DEFAULT_OUTPUT = "results/metrics/adversarial_separability.csv"

CSV_FIELDS = [
    "checkpoint_path",
    "checkpoint_epoch",
    "checkpoint_threshold",
    "checkpoint_type",
    "site",
    "method",
    "rank",
    "lam",
    "repeat",
    "n_train_fit",
    "n_test_fit",
    "clean_train_acc_heldout",
    "clean_test_acc_heldout",
    "int_train_acc_heldout",
    "int_test_acc_heldout",
    "int_train_acc_fit",
    "int_test_acc_fit",
    "train_drop",
    "test_drop",
    "selective_gap",
    "separability_pass",
    "scoped_separability_pass",
    "final_basis_train_acc_heldout",
    "final_basis_test_acc_heldout",
    "final_ce_train_fit",
    "final_ce_test_fit",
    "opt_steps",
    "opt_lr",
]


def _stable_seed(*parts: Any) -> int:
    material = ":".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31)


def _load_model_on_cpu(checkpoint_path: Path) -> Tuple[torch.nn.Module, Dict[str, Any], Config]:
    checkpoint_payload = _load_checkpoint_payload(checkpoint_path)
    base_cfg = Config()
    checkpoint_cfg = _checkpoint_config_or_default(base_cfg, checkpoint_payload)
    checkpoint_cfg.validate()

    model = get_model(checkpoint_cfg, seed=checkpoint_cfg.train.seed)
    model.load_state_dict(_load_model_state_dict(checkpoint_path))
    model = model.to("cpu")
    model.cfg.device = "cpu"
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, checkpoint_payload, checkpoint_cfg


def _split_half(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    generator: torch.Generator,
    max_fit_examples: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    n = int(tokens.shape[0])
    perm = torch.randperm(n, generator=generator)
    half = n // 2
    fit_idx = perm[:half]
    heldout_idx = perm[half:]
    if fit_idx.shape[0] > max_fit_examples:
        keep = torch.randperm(fit_idx.shape[0], generator=generator)[:max_fit_examples]
        fit_idx = fit_idx[keep]
    fit = (tokens[fit_idx], labels[fit_idx])
    heldout = (tokens[heldout_idx], labels[heldout_idx])
    return fit, heldout


def _project_out_hook(basis: torch.Tensor, positions: Sequence[int]) -> Callable:
    target_positions = list(positions)

    def _hook(value: torch.Tensor, hook: Any) -> torch.Tensor:
        u = basis.to(device=value.device, dtype=value.dtype)
        patched = value.clone()
        selected = patched[:, target_positions, :]
        coeffs = selected @ u
        patched[:, target_positions, :] = selected - coeffs @ u.T
        return patched

    return _hook


@torch.inference_mode()
def _accuracy(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    hook_name: Optional[str] = None,
    hook_fn: Optional[Callable] = None,
) -> float:
    model.eval()
    total = int(tokens.shape[0])
    if total == 0:
        raise ValueError("Cannot evaluate an empty split.")
    correct = 0
    hooks = [(hook_name, hook_fn)] if hook_name and hook_fn else []
    with model.hooks(fwd_hooks=hooks):
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_tokens = tokens[start:end].to("cpu")
            batch_labels = labels[start:end].to("cpu")
            logits = model(batch_tokens)[:, -1, :]
            correct += int((logits.argmax(dim=-1) == batch_labels).sum().item())
    return correct / total


def _cross_entropy_with_basis(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    hook_name: str,
    basis: Optional[torch.Tensor],
    positions: Sequence[int],
) -> torch.Tensor:
    hooks = []
    if basis is not None:
        hooks = [(hook_name, _project_out_hook(basis, positions))]
    with model.hooks(fwd_hooks=hooks):
        logits = model(tokens.to("cpu"))[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, labels.to("cpu"))


def _paired_cross_entropy_with_basis(
    model: torch.nn.Module,
    train_fit: Tuple[torch.Tensor, torch.Tensor],
    test_fit: Tuple[torch.Tensor, torch.Tensor],
    hook_name: str,
    basis: torch.Tensor,
    positions: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_train = int(train_fit[0].shape[0])
    tokens = torch.cat([train_fit[0], test_fit[0]], dim=0)
    hooks = [(hook_name, _project_out_hook(basis, positions))]
    with model.hooks(fwd_hooks=hooks):
        logits = model(tokens.to("cpu"))[:, -1, :]
    ce_train = torch.nn.functional.cross_entropy(
        logits[:n_train], train_fit[1].to("cpu")
    )
    ce_test = torch.nn.functional.cross_entropy(
        logits[n_train:], test_fit[1].to("cpu")
    )
    return ce_train, ce_test


def _learn_adversarial_basis(
    model: torch.nn.Module,
    hook_name: str,
    positions: Sequence[int],
    train_fit: Tuple[torch.Tensor, torch.Tensor],
    test_fit: Tuple[torch.Tensor, torch.Tensor],
    rank: int,
    lam: float,
    steps: int,
    lr: float,
    seed: int,
    eval_batch_size: int,
    train_acc_floor: float,
    clean_train_fit_ce: float,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Maximize test-fit CE subject to a hinge penalty on train-fit CE excess.

    loss = -CE_test + lam * relu(CE_train - clean_train_fit_ce)

    Returns (best gated basis, final basis, best-basis train/test fit CE). The
    gated basis is the one with maximal test-fit CE among evaluation points
    whose train-fit accuracy stayed above train_acc_floor; if no point
    qualifies, the final basis is returned for both.
    """
    d_model = int(model.cfg.d_model)
    generator = torch.Generator().manual_seed(seed)
    v = torch.randn(d_model, rank, generator=generator, requires_grad=True)
    optimizer = torch.optim.Adam([v], lr=lr)

    train_tokens, train_labels = train_fit

    best_basis: Optional[torch.Tensor] = None
    best_test_ce = -float("inf")
    saw_valid = False

    for step in range(steps):
        optimizer.zero_grad()
        u = torch.linalg.qr(v, mode="reduced").Q
        ce_train, ce_test = _paired_cross_entropy_with_basis(
            model, train_fit, test_fit, hook_name, u, positions
        )
        loss = -ce_test + lam * torch.relu(ce_train - clean_train_fit_ce)
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == steps - 1:
            with torch.no_grad():
                u_eval = torch.linalg.qr(v, mode="reduced").Q
                hook_fn = _project_out_hook(u_eval, positions)
                train_acc = _accuracy(
                    model, train_tokens, train_labels, eval_batch_size, hook_name, hook_fn
                )
                test_ce_eval = _cross_entropy_with_basis(
                    model, test_fit[0], test_fit[1], hook_name, u_eval, positions
                ).item()
                if train_acc >= train_acc_floor:
                    saw_valid = True
                    if test_ce_eval > best_test_ce:
                        best_test_ce = test_ce_eval
                        best_basis = u_eval.detach().clone()

    with torch.no_grad():
        final_u = torch.linalg.qr(v, mode="reduced").Q.detach()
    if not saw_valid or best_basis is None:
        best_basis = final_u

    with torch.no_grad():
        best_ce_train, best_ce_test = _paired_cross_entropy_with_basis(
            model, train_fit, test_fit, hook_name, best_basis, positions
        )
    return best_basis, final_u, float(best_ce_train.item()), float(best_ce_test.item())


def _fourier_topk_basis(
    model: torch.nn.Module,
    p: int,
    site: str,
    rank: int,
) -> torch.Tensor:
    fourier_basis = build_fourier_basis(p)
    score_rows = score_frequencies(model=model, p=p)
    directions = _residual_directions_for_site(
        model=model, p=p, fourier_basis=fourier_basis, site=site
    )
    ordered = [directions[int(row["frequency"])] for row in score_rows]
    stacked_rows: List[torch.Tensor] = []
    total = 0
    for block in ordered:
        stacked_rows.append(block)
        total += int(block.shape[0])
        if total >= rank:
            break
    concatenated = _orthonormalize_rows(torch.cat(stacked_rows, dim=0))
    basis_rows = concatenated[:rank]
    return basis_rows.T.contiguous().float()


def _random_basis(d_model: int, rank: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    v = torch.randn(d_model, rank, generator=generator)
    return torch.linalg.qr(v, mode="reduced").Q.float()


def _collect_site_activations(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    hook_name: str,
    positions: Sequence[int],
    batch_size: int,
) -> torch.Tensor:
    target_positions = list(positions)
    collected: List[torch.Tensor] = []

    def _cache_hook(value: torch.Tensor, hook: Any) -> torch.Tensor:
        collected.append(value[:, target_positions, :].reshape(-1, value.shape[-1]).detach().clone())
        return value

    total = int(tokens.shape[0])
    with torch.inference_mode():
        with model.hooks(fwd_hooks=[(hook_name, _cache_hook)]):
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                model(tokens[start:end].to("cpu"))
    return torch.cat(collected, dim=0).float()


def _pca_basis(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    hook_name: str,
    positions: Sequence[int],
    rank: int,
    batch_size: int,
) -> torch.Tensor:
    activations = _collect_site_activations(model, tokens, hook_name, positions, batch_size)
    centered = activations - activations.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[:rank].T.contiguous().float()


def _gradient_basis(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    hook_name: str,
    positions: Sequence[int],
    rank: int,
    batch_size: int,
) -> torch.Tensor:
    target_positions = list(positions)
    grad_rows: List[torch.Tensor] = []
    total = int(tokens.shape[0])
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_tokens = tokens[start:end].to("cpu")
        batch_labels = labels[start:end].to("cpu")
        captured: Dict[str, torch.Tensor] = {}

        def _capture_hook(value: torch.Tensor, hook: Any) -> torch.Tensor:
            value.requires_grad_(True)
            value.retain_grad()
            captured["act"] = value
            return value

        model.zero_grad(set_to_none=True)
        with model.hooks(fwd_hooks=[(hook_name, _capture_hook)]):
            logits = model(batch_tokens)[:, -1, :]
            loss = torch.nn.functional.cross_entropy(
                logits, batch_labels, reduction="sum"
            )
            loss.backward()
        act = captured["act"]
        grad = act.grad[:, target_positions, :].reshape(-1, act.shape[-1]).detach().clone()
        grad_rows.append(grad)
    grad_matrix = torch.cat(grad_rows, dim=0).float()
    _, _, vh = torch.linalg.svd(grad_matrix, full_matrices=False)
    return vh[:rank].T.contiguous().float()


def _metrics_row(
    *,
    checkpoint_path: Path,
    checkpoint_payload: Dict[str, Any],
    site: str,
    method: str,
    rank: int,
    lam: Optional[float],
    repeat: int,
    n_train_fit: int,
    n_test_fit: int,
    clean_train_heldout: float,
    clean_test_heldout: float,
    int_train_heldout: float,
    int_test_heldout: float,
    int_train_fit: float,
    int_test_fit: float,
    final_ce_train_fit: Optional[float],
    final_ce_test_fit: Optional[float],
    final_basis_train_acc_heldout: Optional[float] = None,
    final_basis_test_acc_heldout: Optional[float] = None,
    steps: Optional[int],
    lr: Optional[float],
    p: int,
) -> Dict[str, Any]:
    train_drop = clean_train_heldout - int_train_heldout
    test_drop = clean_test_heldout - int_test_heldout
    separability_pass = (int_train_heldout >= 0.90) and (int_test_heldout <= 2.0 / p)
    scoped_separability_pass = (
        int_train_heldout >= 0.9 * clean_train_heldout
        and int_test_heldout <= 2.0 / p
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "checkpoint_threshold": checkpoint_payload.get("checkpoint_threshold"),
        "checkpoint_type": checkpoint_payload.get("checkpoint_type"),
        "site": site,
        "method": method,
        "rank": rank,
        "lam": "" if lam is None else lam,
        "repeat": repeat,
        "n_train_fit": n_train_fit,
        "n_test_fit": n_test_fit,
        "clean_train_acc_heldout": clean_train_heldout,
        "clean_test_acc_heldout": clean_test_heldout,
        "int_train_acc_heldout": int_train_heldout,
        "int_test_acc_heldout": int_test_heldout,
        "int_train_acc_fit": int_train_fit,
        "int_test_acc_fit": int_test_fit,
        "train_drop": train_drop,
        "test_drop": test_drop,
        "selective_gap": test_drop - train_drop,
        "separability_pass": separability_pass,
        "scoped_separability_pass": scoped_separability_pass,
        "final_basis_train_acc_heldout": (
            "" if final_basis_train_acc_heldout is None else final_basis_train_acc_heldout
        ),
        "final_basis_test_acc_heldout": (
            "" if final_basis_test_acc_heldout is None else final_basis_test_acc_heldout
        ),
        "final_ce_train_fit": "" if final_ce_train_fit is None else final_ce_train_fit,
        "final_ce_test_fit": "" if final_ce_test_fit is None else final_ce_test_fit,
        "opt_steps": "" if steps is None else steps,
        "opt_lr": "" if lr is None else lr,
    }


def _evaluate_basis(
    model: torch.nn.Module,
    hook_name: str,
    positions: Sequence[int],
    basis: torch.Tensor,
    train_heldout: Tuple[torch.Tensor, torch.Tensor],
    test_heldout: Tuple[torch.Tensor, torch.Tensor],
    train_fit: Tuple[torch.Tensor, torch.Tensor],
    test_fit: Tuple[torch.Tensor, torch.Tensor],
    batch_size: int,
) -> Dict[str, float]:
    hook_fn = _project_out_hook(basis, positions)
    return {
        "int_train_heldout": _accuracy(
            model, train_heldout[0], train_heldout[1], batch_size, hook_name, hook_fn
        ),
        "int_test_heldout": _accuracy(
            model, test_heldout[0], test_heldout[1], batch_size, hook_name, hook_fn
        ),
        "int_train_fit": _accuracy(
            model, train_fit[0], train_fit[1], batch_size, hook_name, hook_fn
        ),
        "int_test_fit": _accuracy(
            model, test_fit[0], test_fit[1], batch_size, hook_name, hook_fn
        ),
    }


def process_checkpoint(
    checkpoint_path: Path,
    sites: Sequence[str],
    ranks: Sequence[int],
    lambdas: Sequence[float],
    steps: int,
    lr: float,
    base_seed: int,
    max_fit_examples: int,
    split_seed: int,
    skip_baselines: bool,
    eval_batch_size: int,
    restarts: int,
    writer: csv.DictWriter,
    handle: Any,
) -> None:
    model, checkpoint_payload, checkpoint_cfg = _load_model_on_cpu(checkpoint_path)
    p = int(checkpoint_cfg.model.p)
    d_model = int(model.cfg.d_model)

    dataset = get_dataset(checkpoint_cfg, data_seed=checkpoint_cfg.train.seed)
    train_tokens, train_labels = dataset.train_data()
    test_tokens, test_labels = dataset.test_data()

    generator = torch.Generator().manual_seed(split_seed)
    train_fit, train_heldout = _split_half(
        train_tokens, train_labels, generator, max_fit_examples
    )
    test_fit, test_heldout = _split_half(
        test_tokens, test_labels, generator, max_fit_examples
    )

    n_train_fit = int(train_fit[0].shape[0])
    n_test_fit = int(test_fit[0].shape[0])

    for site in sites:
        hook_name = _site_hook_name(model=model, site=site)
        positions = _site_positions(model=model, site=site)

        clean_train_heldout = _accuracy(
            model, train_heldout[0], train_heldout[1], eval_batch_size
        )
        clean_test_heldout = _accuracy(
            model, test_heldout[0], test_heldout[1], eval_batch_size
        )
        print(
            f"[{checkpoint_path.name}] site={site} clean heldout "
            f"train={clean_train_heldout:.4f} test={clean_test_heldout:.4f}"
        )

        clean_train_fit_acc = _accuracy(
            model, train_fit[0], train_fit[1], eval_batch_size
        )
        train_acc_floor = 0.9 * clean_train_fit_acc
        with torch.no_grad():
            clean_train_fit_ce = _cross_entropy_with_basis(
                model, train_fit[0], train_fit[1], hook_name, None, positions
            ).item()

        for rank in ranks:
            for lam in lambdas:
              for restart in range(restarts):
                seed = _stable_seed(
                    base_seed, str(checkpoint_path), rank, lam, site, restart
                )
                basis, final_basis, ce_train, ce_test = _learn_adversarial_basis(
                    model=model,
                    hook_name=hook_name,
                    positions=positions,
                    train_fit=train_fit,
                    test_fit=test_fit,
                    rank=rank,
                    lam=lam,
                    steps=steps,
                    lr=lr,
                    seed=seed,
                    eval_batch_size=eval_batch_size,
                    train_acc_floor=train_acc_floor,
                    clean_train_fit_ce=clean_train_fit_ce,
                )
                evals = _evaluate_basis(
                    model, hook_name, positions, basis,
                    train_heldout, test_heldout, train_fit, test_fit, eval_batch_size,
                )
                final_hook = _project_out_hook(final_basis, positions)
                final_train_heldout = _accuracy(
                    model, train_heldout[0], train_heldout[1], eval_batch_size,
                    hook_name, final_hook,
                )
                final_test_heldout = _accuracy(
                    model, test_heldout[0], test_heldout[1], eval_batch_size,
                    hook_name, final_hook,
                )
                row = _metrics_row(
                    checkpoint_path=checkpoint_path,
                    checkpoint_payload=checkpoint_payload,
                    site=site,
                    method="adversarial",
                    rank=rank,
                    lam=lam,
                    repeat=restart,
                    n_train_fit=n_train_fit,
                    n_test_fit=n_test_fit,
                    clean_train_heldout=clean_train_heldout,
                    clean_test_heldout=clean_test_heldout,
                    int_train_heldout=evals["int_train_heldout"],
                    int_test_heldout=evals["int_test_heldout"],
                    int_train_fit=evals["int_train_fit"],
                    int_test_fit=evals["int_test_fit"],
                    final_ce_train_fit=ce_train,
                    final_ce_test_fit=ce_test,
                    final_basis_train_acc_heldout=final_train_heldout,
                    final_basis_test_acc_heldout=final_test_heldout,
                    steps=steps,
                    lr=lr,
                    p=p,
                )
                writer.writerow(row)
                handle.flush()
                print(
                    f"  adversarial r={rank} lam={lam} -> heldout "
                    f"train={evals['int_train_heldout']:.4f} "
                    f"test={evals['int_test_heldout']:.4f} "
                    f"selgap={row['selective_gap']:.4f} "
                    f"pass={row['separability_pass']}"
                )

            if skip_baselines:
                continue

            baseline_bases: List[Tuple[str, int, torch.Tensor]] = []
            baseline_bases.append(
                ("fourier_topk", 0, _fourier_topk_basis(model, p, site, rank))
            )
            for repeat in range(5):
                rseed = _stable_seed(base_seed, str(checkpoint_path), rank, "random", repeat, site)
                baseline_bases.append(("random", repeat, _random_basis(d_model, rank, rseed)))
            pca_tokens = torch.cat([train_fit[0], test_fit[0]], dim=0)
            baseline_bases.append(
                ("pca", 0, _pca_basis(model, pca_tokens, hook_name, positions, rank, eval_batch_size))
            )
            baseline_bases.append(
                (
                    "gradient",
                    0,
                    _gradient_basis(
                        model, test_fit[0], test_fit[1], hook_name, positions, rank, eval_batch_size
                    ),
                )
            )

            for method, repeat, basis in baseline_bases:
                evals = _evaluate_basis(
                    model, hook_name, positions, basis,
                    train_heldout, test_heldout, train_fit, test_fit, eval_batch_size,
                )
                row = _metrics_row(
                    checkpoint_path=checkpoint_path,
                    checkpoint_payload=checkpoint_payload,
                    site=site,
                    method=method,
                    rank=rank,
                    lam=None,
                    repeat=repeat,
                    n_train_fit=n_train_fit,
                    n_test_fit=n_test_fit,
                    clean_train_heldout=clean_train_heldout,
                    clean_test_heldout=clean_test_heldout,
                    int_train_heldout=evals["int_train_heldout"],
                    int_test_heldout=evals["int_test_heldout"],
                    int_train_fit=evals["int_train_fit"],
                    int_test_fit=evals["int_test_fit"],
                    final_ce_train_fit=None,
                    final_ce_test_fit=None,
                    steps=None,
                    lr=None,
                    p=p,
                )
                writer.writerow(row)
                handle.flush()
                print(
                    f"  {method} r={rank} rep={repeat} -> heldout "
                    f"train={evals['int_train_heldout']:.4f} "
                    f"test={evals['int_test_heldout']:.4f}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--sites", nargs="+", default=["pre_unembed"])
    parser.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--max-fit-examples", type=int, default=2048)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(True)

    ranks = args.ranks
    lambdas = args.lambdas
    steps = args.steps
    restarts = args.restarts
    if args.smoke:
        ranks = [4]
        lambdas = [0.0, 1.0]
        steps = 50
        restarts = 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()

    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
            handle.flush()

        for checkpoint in args.checkpoints:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            print(f"Processing {checkpoint_path}")
            process_checkpoint(
                checkpoint_path=checkpoint_path,
                sites=args.sites,
                ranks=ranks,
                lambdas=lambdas,
                steps=steps,
                lr=args.lr,
                base_seed=args.base_seed,
                max_fit_examples=args.max_fit_examples,
                split_seed=args.base_seed,
                skip_baselines=args.skip_baselines,
                eval_batch_size=args.eval_batch_size,
                restarts=restarts,
                writer=writer,
                handle=handle,
            )


if __name__ == "__main__":
    main()
