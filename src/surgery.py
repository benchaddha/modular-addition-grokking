import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import Config
from .dataset import get_dataset
from .model import get_model


def _load_model_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")


def _select_probe_data(
    cfg: Config,
    probe_split: str,
    probe_max_examples: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = get_dataset(cfg, data_seed=seed)
    if probe_split == "train":
        tokens, labels = dataset.train_data()
    elif probe_split == "test":
        tokens, labels = dataset.test_data()
    else:
        raise ValueError(f"Unsupported probe split: {probe_split}")

    if probe_max_examples > 0 and probe_max_examples < tokens.shape[0]:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(tokens.shape[0], generator=generator)[:probe_max_examples]
        tokens = tokens[indices]
        labels = labels[indices]
    return tokens, labels


def rank_heads_by_correct_logit_attribution(
    cfg: Config,
    checkpoint_path: Path,
    probe_split: Optional[str] = None,
    probe_max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    cfg.validate()
    cfg.validate_surgery()

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    split = probe_split or cfg.surgery.probe_split
    max_examples = (
        cfg.surgery.probe_max_examples
        if probe_max_examples is None
        else probe_max_examples
    )

    model = get_model(cfg, seed=cfg.surgery.seed)
    model.load_state_dict(_load_model_state_dict(checkpoint_path))
    model.eval()

    tokens, labels = _select_probe_data(
        cfg=cfg,
        probe_split=split,
        probe_max_examples=max_examples,
        seed=cfg.surgery.seed,
    )

    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    signed_sum = torch.zeros((n_layers, n_heads), dtype=torch.float64)
    abs_sum = torch.zeros((n_layers, n_heads), dtype=torch.float64)
    num_examples = int(tokens.shape[0])
    if num_examples == 0:
        raise ValueError("Probe split produced zero examples.")

    batch_size = cfg.surgery.eval_batch_size
    for start in range(0, num_examples, batch_size):
        end = min(start + batch_size, num_examples)
        batch_tokens = tokens[start:end].to(model.cfg.device)
        batch_labels = labels[start:end].to(model.cfg.device)

        model.zero_grad(set_to_none=True)
        cache = model.add_caching_hooks(
            names_filter=lambda name: name.endswith("attn.hook_z"),
            incl_bwd=True,
            device="cpu",
        )
        logits = model(batch_tokens)
        correct_logits = logits[:, -1, :].gather(1, batch_labels.unsqueeze(-1)).squeeze(-1)
        correct_logits.mean().backward()
        model.reset_hooks()

        for layer in range(n_layers):
            key = f"blocks.{layer}.attn.hook_z"
            grad_key = f"{key}_grad"
            if key not in cache or grad_key not in cache:
                raise KeyError(f"Missing cache key(s): {key}, {grad_key}")

            # Attribution at final position for each head in this layer.
            z = cache[key][:, -1, :, :]
            z_grad = cache[grad_key][:, -1, :, :]
            contributions = (z * z_grad).sum(dim=-1)  # [batch, head]
            signed_sum[layer] += contributions.sum(dim=0, dtype=torch.float64)
            abs_sum[layer] += contributions.abs().sum(dim=0, dtype=torch.float64)

    signed_mean = signed_sum / num_examples
    abs_mean = abs_sum / num_examples

    rows: List[Dict[str, Any]] = []
    for layer in range(n_layers):
        for head in range(n_heads):
            rows.append(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "probe_split": split,
                    "probe_examples": num_examples,
                    "layer": layer,
                    "head": head,
                    "score": float(signed_mean[layer, head].item()),
                    "abs_score": float(abs_mean[layer, head].item()),
                }
            )

    rows.sort(key=lambda row: row["abs_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def run_surgery_head_ranking(
    cfg: Config,
    output_path: Path = Path("results/metrics/surgery_head_scores.jsonl"),
) -> List[Dict[str, Any]]:
    cfg.validate()
    cfg.validate_surgery()

    all_rows: List[Dict[str, Any]] = []
    for checkpoint in cfg.surgery.checkpoint_paths:
        checkpoint_rows = rank_heads_by_correct_logit_attribution(
            cfg=cfg,
            checkpoint_path=Path(checkpoint),
        )
        all_rows.extend(checkpoint_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")
    return all_rows
