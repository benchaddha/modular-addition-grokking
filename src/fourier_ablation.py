import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from .config import Config
from .dataset import get_dataset
from .model import get_model
from .surgery import _load_checkpoint_payload, _load_model_state_dict


def _orthonormalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("Expected a rank-2 matrix to orthonormalize.")
    if matrix.shape[0] == 0:
        return matrix

    q_matrix, _ = torch.linalg.qr(matrix.T, mode="reduced")
    return q_matrix.T.contiguous()


def build_fourier_basis(p: int) -> Dict[int, torch.Tensor]:
    """Build an explicit real Fourier basis over Z_p for frequencies 1..(p-1)//2."""
    if p <= 2:
        raise ValueError("p must be > 2 for a non-trivial real Fourier basis.")

    values = torch.arange(p, dtype=torch.float64)
    basis: Dict[int, torch.Tensor] = {}
    scale = math.sqrt(2.0 / p)
    for frequency in range(1, (p - 1) // 2 + 1):
        angle = 2.0 * math.pi * frequency * values / p
        cos_vec = scale * torch.cos(angle)
        sin_vec = scale * torch.sin(angle)
        basis[frequency] = _orthonormalize_rows(torch.stack([cos_vec, sin_vec], dim=0))
    return basis


def _number_token_embedding_matrix(model: torch.nn.Module, p: int) -> torch.Tensor:
    return model.W_E.detach().float().cpu()[:p, :]


def _number_token_unembedding_matrix(model: torch.nn.Module, p: int) -> torch.Tensor:
    return model.W_U.detach().float().cpu()[:, :p].T.contiguous()


def _project_frequency_subspace(
    token_space_basis: torch.Tensor,
    token_matrix: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    projected = token_space_basis @ token_matrix.to(dtype=token_space_basis.dtype)
    return {
        "projected": projected,
        "orthonormal": _orthonormalize_rows(projected),
    }


def score_frequencies(model: torch.nn.Module, p: int) -> List[Dict[str, Any]]:
    """Return per-frequency embed/unembed importance scores."""
    fourier_basis = build_fourier_basis(p)
    embed_matrix = _number_token_embedding_matrix(model=model, p=p)
    unembed_matrix = _number_token_unembedding_matrix(model=model, p=p)

    rows: List[Dict[str, Any]] = []
    for frequency, token_basis in fourier_basis.items():
        embed_projection = _project_frequency_subspace(token_basis, embed_matrix)
        unembed_projection = _project_frequency_subspace(token_basis, unembed_matrix)

        embed_score = float(torch.linalg.matrix_norm(embed_projection["projected"]).item())
        unembed_score = float(
            torch.linalg.matrix_norm(unembed_projection["projected"]).item()
        )
        rows.append(
            {
                "frequency": frequency,
                "embed_score": embed_score,
                "unembed_score": unembed_score,
                "combined_score": embed_score * unembed_score,
            }
        )

    rows.sort(key=lambda row: row["combined_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _site_positions(model: torch.nn.Module, site: str) -> List[int]:
    if site == "post_embed":
        return [0, 1]
    if site == "pre_unembed":
        return [int(model.cfg.n_ctx) - 1]
    raise ValueError(f"Unsupported Fourier ablation site: {site}")


def _site_hook_name(model: torch.nn.Module, site: str) -> str:
    if site == "post_embed":
        return "hook_embed"
    if site == "pre_unembed":
        return f"blocks.{int(model.cfg.n_layers) - 1}.hook_resid_post"
    raise ValueError(f"Unsupported Fourier ablation site: {site}")


def _residual_directions_for_site(
    model: torch.nn.Module,
    p: int,
    fourier_basis: Dict[int, torch.Tensor],
    site: str,
) -> Dict[int, torch.Tensor]:
    if site == "post_embed":
        token_matrix = _number_token_embedding_matrix(model=model, p=p)
    elif site == "pre_unembed":
        token_matrix = _number_token_unembedding_matrix(model=model, p=p)
    else:
        raise ValueError(f"Unsupported Fourier ablation site: {site}")

    directions: Dict[int, torch.Tensor] = {}
    for frequency, token_basis in fourier_basis.items():
        directions[frequency] = _project_frequency_subspace(
            token_basis,
            token_matrix,
        )["orthonormal"]
    return directions


def make_freq_ablation_hook(
    freqs_to_ablate: List[int],
    fourier_basis: Dict[int, torch.Tensor],
    model: torch.nn.Module,
    site: str,
    intervention_mode: str = "ablate_selected",
    positions: Optional[Sequence[int]] = None,
) -> Callable:
    residual_directions = _residual_directions_for_site(
        model=model,
        p=int(model.cfg.d_vocab) - 1,
        fourier_basis=fourier_basis,
        site=site,
    )
    selected_directions = [residual_directions[freq] for freq in freqs_to_ablate]
    if not selected_directions:
        raise ValueError("freqs_to_ablate must be non-empty.")

    direction_basis = _orthonormalize_rows(torch.cat(selected_directions, dim=0))
    full_fourier_basis = _orthonormalize_rows(
        torch.cat(list(residual_directions.values()), dim=0)
    )
    target_positions = list(_site_positions(model, site) if positions is None else positions)

    def _ablation_hook(value: torch.Tensor, hook: Any) -> torch.Tensor:
        basis = direction_basis.to(device=value.device, dtype=value.dtype)
        full_basis = full_fourier_basis.to(device=value.device, dtype=value.dtype)
        patched = value.clone()
        selected = patched[:, target_positions, :]
        coeffs = torch.einsum("bpd,fd->bpf", selected, basis)
        selected_reconstruction = torch.einsum("bpf,fd->bpd", coeffs, basis)
        if intervention_mode == "ablate_selected":
            patched[:, target_positions, :] = selected - selected_reconstruction
            return patched
        if intervention_mode == "keep_only_selected":
            full_coeffs = torch.einsum("bpd,fd->bpf", selected, full_basis)
            full_reconstruction = torch.einsum("bpf,fd->bpd", full_coeffs, full_basis)
            patched[:, target_positions, :] = selected - full_reconstruction + selected_reconstruction
            return patched
        raise ValueError(f"Unsupported intervention_mode: {intervention_mode}")
        return patched

    return _ablation_hook


@torch.inference_mode()
def _evaluate_with_optional_hook(
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
            batch_tokens = tokens[start:end].to(model.cfg.device)
            batch_labels = labels[start:end].to(model.cfg.device)
            logits = model(batch_tokens)[:, -1, :]
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == batch_labels).sum().item())
    return correct / total


def _checkpoint_config_or_default(
    cfg: Config,
    checkpoint_payload: Dict[str, Any],
) -> Config:
    checkpoint_cfg = checkpoint_payload.get("cfg")
    if isinstance(checkpoint_cfg, dict):
        merged_cfg = cfg.to_dict()
        for key in checkpoint_cfg:
            merged_cfg[key] = checkpoint_cfg[key]
        merged_cfg["fourier_ablation"] = cfg.to_dict()["fourier_ablation"]
        return Config.from_dict(merged_cfg)
    return cfg


def _stable_random_seed(
    *,
    base_seed: int,
    checkpoint_path: str,
    k: int,
    repeat_index: int,
) -> int:
    material = f"{base_seed}:{checkpoint_path}:{k}:{repeat_index}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31)


def _frequency_sets_from_config(
    cfg: Config,
    score_rows: Sequence[Dict[str, Any]],
    checkpoint_path: str,
) -> List[Dict[str, Any]]:
    max_frequency = (cfg.model.p - 1) // 2
    mode = cfg.fourier_ablation.sweep_mode
    ranked_frequencies = [int(row["frequency"]) for row in score_rows]

    if mode == "all_singles":
        return [
            {
                "label": f"freq_{frequency:02d}",
                "frequencies": [frequency],
                "selection_mode": mode,
                "selection_family": "single",
                "repeat_index": 0,
            }
            for frequency in range(1, max_frequency + 1)
        ]

    if mode == "top_k":
        return [
            {
                "label": f"top_{k}",
                "frequencies": ranked_frequencies[:k],
                "selection_mode": mode,
                "selection_family": "top",
                "repeat_index": 0,
            }
            for k in cfg.fourier_ablation.top_k_values
        ]

    if mode == "multi_frequency":
        frequency_sets: List[Dict[str, Any]] = []
        for k in cfg.fourier_ablation.top_k_values:
            frequency_sets.append(
                {
                    "label": f"top_{k}",
                    "frequencies": ranked_frequencies[:k],
                    "selection_mode": mode,
                    "selection_family": "top",
                    "repeat_index": 0,
                }
            )

        for k in cfg.fourier_ablation.bottom_k_values:
            frequency_sets.append(
                {
                    "label": f"bottom_{k}",
                    "frequencies": ranked_frequencies[-k:],
                    "selection_mode": mode,
                    "selection_family": "bottom",
                    "repeat_index": 0,
                }
            )

        all_frequencies = torch.tensor(ranked_frequencies, dtype=torch.long)
        for k in cfg.fourier_ablation.random_k_values:
            for repeat_index in range(1, cfg.fourier_ablation.random_control_repeats + 1):
                generator = torch.Generator()
                generator.manual_seed(
                    _stable_random_seed(
                        base_seed=cfg.fourier_ablation.seed,
                        checkpoint_path=checkpoint_path,
                        k=k,
                        repeat_index=repeat_index,
                    )
                )
                perm = torch.randperm(int(all_frequencies.shape[0]), generator=generator)
                random_frequencies = all_frequencies[perm[:k]].tolist()
                frequency_sets.append(
                    {
                        "label": f"random_{k}_rep{repeat_index}",
                        "frequencies": sorted(int(freq) for freq in random_frequencies),
                        "selection_mode": mode,
                        "selection_family": "random",
                        "repeat_index": repeat_index,
                    }
                )

        return frequency_sets

    if mode == "custom":
        return [
            {
                "label": "custom_" + "_".join(str(freq) for freq in freq_set),
                "frequencies": list(freq_set),
                "selection_mode": mode,
                "selection_family": "custom",
                "repeat_index": 0,
            }
            for freq_set in cfg.fourier_ablation.custom_frequency_sets
        ]

    raise ValueError(f"Unsupported Fourier ablation sweep mode: {mode}")


def run_fourier_ablation_sweep(cfg: Config) -> Dict[str, List[Dict[str, Any]]]:
    cfg.validate()
    cfg.validate_fourier_ablation()

    all_score_rows: List[Dict[str, Any]] = []
    all_ablation_rows: List[Dict[str, Any]] = []

    for checkpoint in cfg.fourier_ablation.checkpoint_paths:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint_payload = _load_checkpoint_payload(checkpoint_path)
        checkpoint_cfg = _checkpoint_config_or_default(cfg, checkpoint_payload)
        checkpoint_cfg.validate()

        model = get_model(checkpoint_cfg, seed=checkpoint_cfg.train.seed)
        model.load_state_dict(_load_model_state_dict(checkpoint_path))
        model.eval()

        score_rows = score_frequencies(model=model, p=checkpoint_cfg.model.p)
        for row in score_rows:
            row["checkpoint_path"] = str(checkpoint_path)
            row["checkpoint_epoch"] = checkpoint_payload.get("epoch")
            row["checkpoint_type"] = checkpoint_payload.get("checkpoint_type")
            row["checkpoint_threshold"] = checkpoint_payload.get("checkpoint_threshold")
        all_score_rows.extend(score_rows)

        dataset = get_dataset(checkpoint_cfg, data_seed=checkpoint_cfg.train.seed)
        train_tokens, train_labels = dataset.train_data()
        test_tokens, test_labels = dataset.test_data()
        baseline_train_acc = _evaluate_with_optional_hook(
            model=model,
            tokens=train_tokens,
            labels=train_labels,
            batch_size=cfg.fourier_ablation.eval_batch_size,
        )
        baseline_test_acc = _evaluate_with_optional_hook(
            model=model,
            tokens=test_tokens,
            labels=test_labels,
            batch_size=cfg.fourier_ablation.eval_batch_size,
        )

        fourier_basis = build_fourier_basis(checkpoint_cfg.model.p)
        frequency_sets = _frequency_sets_from_config(
            cfg=checkpoint_cfg,
            score_rows=score_rows,
            checkpoint_path=str(checkpoint_path),
        )
        chance_threshold = (
            checkpoint_cfg.fourier_ablation.causal_test_chance_multiplier
            / checkpoint_cfg.model.p
        )

        for site in checkpoint_cfg.fourier_ablation.sites:
            all_ablation_rows.append(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_epoch": checkpoint_payload.get("epoch"),
                    "checkpoint_type": checkpoint_payload.get("checkpoint_type"),
                    "checkpoint_threshold": checkpoint_payload.get("checkpoint_threshold"),
                    "site": site,
                    "intervention_mode": checkpoint_cfg.fourier_ablation.intervention_mode,
                    "selection_mode": "baseline",
                    "selection_family": "baseline",
                    "frequency_set_label": "baseline",
                    "frequencies": [],
                    "num_frequencies": 0,
                    "repeat_index": 0,
                    "train_acc": baseline_train_acc,
                    "test_acc": baseline_test_acc,
                    "baseline_train_acc": baseline_train_acc,
                    "baseline_test_acc": baseline_test_acc,
                    "train_drop": 0.0,
                    "test_drop": 0.0,
                    "selective_gap": 0.0,
                    "strong_h2_pass": False,
                }
            )

            hook_name = _site_hook_name(model=model, site=site)
            for frequency_set in frequency_sets:
                hook_fn = make_freq_ablation_hook(
                    freqs_to_ablate=frequency_set["frequencies"],
                    fourier_basis=fourier_basis,
                    model=model,
                    site=site,
                    intervention_mode=checkpoint_cfg.fourier_ablation.intervention_mode,
                )
                train_acc = _evaluate_with_optional_hook(
                    model=model,
                    tokens=train_tokens,
                    labels=train_labels,
                    batch_size=checkpoint_cfg.fourier_ablation.eval_batch_size,
                    hook_name=hook_name,
                    hook_fn=hook_fn,
                )
                test_acc = _evaluate_with_optional_hook(
                    model=model,
                    tokens=test_tokens,
                    labels=test_labels,
                    batch_size=checkpoint_cfg.fourier_ablation.eval_batch_size,
                    hook_name=hook_name,
                    hook_fn=hook_fn,
                )
                train_drop = baseline_train_acc - train_acc
                test_drop = baseline_test_acc - test_acc
                all_ablation_rows.append(
                    {
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_epoch": checkpoint_payload.get("epoch"),
                        "checkpoint_type": checkpoint_payload.get("checkpoint_type"),
                        "checkpoint_threshold": checkpoint_payload.get("checkpoint_threshold"),
                        "site": site,
                        "intervention_mode": checkpoint_cfg.fourier_ablation.intervention_mode,
                        "selection_mode": frequency_set["selection_mode"],
                        "selection_family": frequency_set["selection_family"],
                        "frequency_set_label": frequency_set["label"],
                        "frequencies": frequency_set["frequencies"],
                        "num_frequencies": len(frequency_set["frequencies"]),
                        "repeat_index": frequency_set["repeat_index"],
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                        "baseline_train_acc": baseline_train_acc,
                        "baseline_test_acc": baseline_test_acc,
                        "train_drop": train_drop,
                        "test_drop": test_drop,
                        "selective_gap": test_drop - train_drop,
                        "strong_h2_pass": (
                            test_acc <= chance_threshold
                            and train_acc >= checkpoint_cfg.fourier_ablation.causal_train_floor
                        ),
                    }
                )

    return {
        "score_rows": all_score_rows,
        "ablation_rows": all_ablation_rows,
    }
