import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .config import Config
from .dataset import get_dataset
from .model import get_model


def _load_checkpoint_payload(checkpoint_path: Path) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")


def _load_model_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    payload = _load_checkpoint_payload(checkpoint_path)
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
    grad_signed_sum = torch.zeros((n_layers, n_heads), dtype=torch.float64)
    grad_abs_sum = torch.zeros((n_layers, n_heads), dtype=torch.float64)
    dla_signed_sum = torch.zeros((n_layers, n_heads), dtype=torch.float64)
    dla_abs_sum = torch.zeros((n_layers, n_heads), dtype=torch.float64)
    num_examples = int(tokens.shape[0])
    if num_examples == 0:
        raise ValueError("Probe split produced zero examples.")

    # CPU copies for deterministic DLA projection with cached activations.
    w_u_cpu = model.W_U.detach().float().cpu()  # [d_model, d_vocab_out]
    w_o_by_layer_cpu = [
        model.blocks[layer].attn.W_O.detach().float().cpu() for layer in range(n_layers)
    ]  # each [head, d_head, d_model]

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

        batch_labels_cpu = batch_labels.detach().cpu()
        # [batch, d_model] each row is W_U[:, target_token]
        target_unembed = w_u_cpu[:, batch_labels_cpu].T.contiguous()

        for layer in range(n_layers):
            key = f"blocks.{layer}.attn.hook_z"
            grad_key = f"{key}_grad"
            if key not in cache or grad_key not in cache:
                raise KeyError(f"Missing cache key(s): {key}, {grad_key}")

            # Attribution at final position for each head in this layer.
            z = cache[key][:, -1, :, :]
            z_grad = cache[grad_key][:, -1, :, :]
            contributions = (z * z_grad).sum(dim=-1)  # [batch, head]
            grad_signed_sum[layer] += contributions.sum(dim=0, dtype=torch.float64)
            grad_abs_sum[layer] += contributions.abs().sum(dim=0, dtype=torch.float64)

            # DLA: z -> head output via W_O -> projection onto correct-token unembed.
            # z: [batch, head, d_head], W_O: [head, d_head, d_model]
            head_out = torch.einsum("bhd,hdm->bhm", z.float(), w_o_by_layer_cpu[layer])
            dla_contrib = torch.einsum("bhm,bm->bh", head_out, target_unembed)
            dla_signed_sum[layer] += dla_contrib.sum(dim=0, dtype=torch.float64)
            dla_abs_sum[layer] += dla_contrib.abs().sum(dim=0, dtype=torch.float64)

    grad_signed_mean = grad_signed_sum / num_examples
    grad_abs_mean = grad_abs_sum / num_examples
    dla_signed_mean = dla_signed_sum / num_examples
    dla_abs_mean = dla_abs_sum / num_examples

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
                    # Back-compat aliases kept as gradient-based values.
                    "score": float(grad_signed_mean[layer, head].item()),
                    "abs_score": float(grad_abs_mean[layer, head].item()),
                    "grad_score": float(grad_signed_mean[layer, head].item()),
                    "grad_abs_score": float(grad_abs_mean[layer, head].item()),
                    "dla_score": float(dla_signed_mean[layer, head].item()),
                    "dla_abs_score": float(dla_abs_mean[layer, head].item()),
                }
            )

    ranking_metric = cfg.surgery.ranking_metric
    rows.sort(key=lambda row: row[ranking_metric], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["ranking_metric"] = ranking_metric
        row["ranking_score"] = row[ranking_metric]
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


def _heads_to_string_list(heads: Sequence[Tuple[int, int]]) -> List[str]:
    return [f"L{layer}H{head}" for layer, head in heads]


def _subset_label(heads: Sequence[Tuple[int, int]]) -> str:
    if not heads:
        return "none"
    return "+".join(_heads_to_string_list(heads))


def _all_heads(cfg: Config) -> List[Tuple[int, int]]:
    return [
        (layer, head)
        for layer in range(cfg.model.n_layers)
        for head in range(cfg.model.n_heads)
    ]


def _enumerate_head_subsets(
    all_heads: Sequence[Tuple[int, int]],
    max_heads: int = 10,
) -> List[List[Tuple[int, int]]]:
    total_heads = len(all_heads)
    if total_heads > max_heads:
        raise ValueError(
            f"Exhaustive subset ablation is disabled for total_heads={total_heads}; "
            f"max supported is {max_heads}."
        )

    subsets: List[List[Tuple[int, int]]] = []
    for mask in range(1 << total_heads):
        subset = [
            all_heads[index]
            for index in range(total_heads)
            if (mask >> index) & 1
        ]
        subsets.append(subset)
    return subsets


def _build_ablation_hooks(
    ablated_heads: Sequence[Tuple[int, int]],
) -> List[Tuple[str, Any]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for layer, head in ablated_heads:
        grouped[layer].append(head)

    hooks: List[Tuple[str, Any]] = []
    for layer, heads in grouped.items():
        unique_heads = sorted(set(heads))

        def _make_hook(head_indices: List[int]) -> Any:
            def _ablate_hook(value: torch.Tensor, hook: Any) -> torch.Tensor:
                patched = value.clone()
                patched[:, :, head_indices, :] = 0.0
                return patched

            return _ablate_hook

        hooks.append((f"blocks.{layer}.attn.hook_z", _make_hook(unique_heads)))
    return hooks


@torch.inference_mode()
def _evaluate_split_with_ablation(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    ablated_heads: Sequence[Tuple[int, int]],
) -> float:
    model.eval()
    total = int(tokens.shape[0])
    if total == 0:
        raise ValueError("Cannot evaluate empty split.")

    hooks = _build_ablation_hooks(ablated_heads) if ablated_heads else []
    correct = 0
    with model.hooks(fwd_hooks=hooks):
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_tokens = tokens[start:end].to(model.cfg.device)
            batch_labels = labels[start:end].to(model.cfg.device)
            logits = model(batch_tokens)[:, -1, :]
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == batch_labels).sum().item())

    return correct / total


def run_surgery_ablation_sweep(
    cfg: Config,
    output_path: Path = Path("results/metrics/surgery_ablations.jsonl"),
) -> List[Dict[str, Any]]:
    cfg.validate()
    cfg.validate_surgery()

    dataset = get_dataset(cfg, data_seed=cfg.surgery.seed)
    train_tokens, train_labels = dataset.train_data()
    test_tokens, test_labels = dataset.test_data()

    all_rows: List[Dict[str, Any]] = []
    for checkpoint_index, checkpoint in enumerate(cfg.surgery.checkpoint_paths):
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ranked_rows = rank_heads_by_correct_logit_attribution(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
        )
        ranked_heads = [(row["layer"], row["head"]) for row in ranked_rows]
        all_heads = _all_heads(cfg)

        model = get_model(cfg, seed=cfg.surgery.seed)
        model.load_state_dict(_load_model_state_dict(checkpoint_path))
        model.eval()

        baseline_train_acc = _evaluate_split_with_ablation(
            model=model,
            tokens=train_tokens,
            labels=train_labels,
            batch_size=cfg.surgery.eval_batch_size,
            ablated_heads=[],
        )
        baseline_test_acc = _evaluate_split_with_ablation(
            model=model,
            tokens=test_tokens,
            labels=test_labels,
            batch_size=cfg.surgery.eval_batch_size,
            ablated_heads=[],
        )
        all_rows.append(
            {
                "checkpoint_path": str(checkpoint_path),
                "condition": "baseline",
                "k": 0,
                "random_repeat": None,
                "num_ablated_heads": 0,
                "ablated_heads": [],
                "train_acc": baseline_train_acc,
                "test_acc": baseline_test_acc,
                "baseline_train_acc": baseline_train_acc,
                "baseline_test_acc": baseline_test_acc,
            }
        )

        for k in cfg.surgery.top_k:
            if k > len(ranked_heads):
                raise ValueError(
                    f"Requested top_k={k} exceeds available heads={len(ranked_heads)}"
                )
            top_k_heads = ranked_heads[:k]
            topk_train_acc = _evaluate_split_with_ablation(
                model=model,
                tokens=train_tokens,
                labels=train_labels,
                batch_size=cfg.surgery.eval_batch_size,
                ablated_heads=top_k_heads,
            )
            topk_test_acc = _evaluate_split_with_ablation(
                model=model,
                tokens=test_tokens,
                labels=test_labels,
                batch_size=cfg.surgery.eval_batch_size,
                ablated_heads=top_k_heads,
            )
            all_rows.append(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "condition": "top_k",
                    "k": k,
                    "random_repeat": None,
                    "num_ablated_heads": len(top_k_heads),
                    "ablated_heads": _heads_to_string_list(top_k_heads),
                    "train_acc": topk_train_acc,
                    "test_acc": topk_test_acc,
                    "baseline_train_acc": baseline_train_acc,
                    "baseline_test_acc": baseline_test_acc,
                }
            )

            for repeat in range(cfg.surgery.random_control_repeats):
                generator = torch.Generator().manual_seed(
                    cfg.surgery.seed + checkpoint_index * 10_000 + k * 100 + repeat
                )
                perm = torch.randperm(len(all_heads), generator=generator)
                sampled_indices = perm[:k].tolist()
                sampled_heads = [all_heads[idx] for idx in sampled_indices]

                random_train_acc = _evaluate_split_with_ablation(
                    model=model,
                    tokens=train_tokens,
                    labels=train_labels,
                    batch_size=cfg.surgery.eval_batch_size,
                    ablated_heads=sampled_heads,
                )
                random_test_acc = _evaluate_split_with_ablation(
                    model=model,
                    tokens=test_tokens,
                    labels=test_labels,
                    batch_size=cfg.surgery.eval_batch_size,
                    ablated_heads=sampled_heads,
                )
                all_rows.append(
                    {
                        "checkpoint_path": str(checkpoint_path),
                        "condition": "random_k",
                        "k": k,
                        "random_repeat": repeat,
                        "num_ablated_heads": len(sampled_heads),
                        "ablated_heads": _heads_to_string_list(sampled_heads),
                        "train_acc": random_train_acc,
                        "test_acc": random_test_acc,
                        "baseline_train_acc": baseline_train_acc,
                        "baseline_test_acc": baseline_test_acc,
                    }
                )

        all_head_train_acc = _evaluate_split_with_ablation(
            model=model,
            tokens=train_tokens,
            labels=train_labels,
            batch_size=cfg.surgery.eval_batch_size,
            ablated_heads=all_heads,
        )
        all_head_test_acc = _evaluate_split_with_ablation(
            model=model,
            tokens=test_tokens,
            labels=test_labels,
            batch_size=cfg.surgery.eval_batch_size,
            ablated_heads=all_heads,
        )
        all_rows.append(
            {
                "checkpoint_path": str(checkpoint_path),
                "condition": "all_heads",
                "k": len(all_heads),
                "random_repeat": None,
                "num_ablated_heads": len(all_heads),
                "ablated_heads": _heads_to_string_list(all_heads),
                "train_acc": all_head_train_acc,
                "test_acc": all_head_test_acc,
                "baseline_train_acc": baseline_train_acc,
                "baseline_test_acc": baseline_test_acc,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")
    return all_rows


def run_exhaustive_subset_ablation(
    cfg: Config,
    output_path: Path = Path("results/metrics/surgery_exhaustive_subsets.jsonl"),
) -> List[Dict[str, Any]]:
    cfg.validate()
    cfg.validate_surgery()

    dataset = get_dataset(cfg, data_seed=cfg.surgery.seed)
    train_tokens, train_labels = dataset.train_data()
    test_tokens, test_labels = dataset.test_data()
    all_heads = _all_heads(cfg)
    subsets = _enumerate_head_subsets(all_heads=all_heads)

    all_rows: List[Dict[str, Any]] = []
    for checkpoint in cfg.surgery.checkpoint_paths:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint_payload = _load_checkpoint_payload(checkpoint_path)
        checkpoint_epoch = checkpoint_payload.get("epoch")
        checkpoint_type = checkpoint_payload.get("checkpoint_type")
        checkpoint_threshold = checkpoint_payload.get("checkpoint_threshold")

        model = get_model(cfg, seed=cfg.surgery.seed)
        model.load_state_dict(_load_model_state_dict(checkpoint_path))
        model.eval()

        baseline_train_acc = _evaluate_split_with_ablation(
            model=model,
            tokens=train_tokens,
            labels=train_labels,
            batch_size=cfg.surgery.eval_batch_size,
            ablated_heads=[],
        )
        baseline_test_acc = _evaluate_split_with_ablation(
            model=model,
            tokens=test_tokens,
            labels=test_labels,
            batch_size=cfg.surgery.eval_batch_size,
            ablated_heads=[],
        )

        for subset_index, subset_heads in enumerate(subsets):
            train_acc = _evaluate_split_with_ablation(
                model=model,
                tokens=train_tokens,
                labels=train_labels,
                batch_size=cfg.surgery.eval_batch_size,
                ablated_heads=subset_heads,
            )
            test_acc = _evaluate_split_with_ablation(
                model=model,
                tokens=test_tokens,
                labels=test_labels,
                batch_size=cfg.surgery.eval_batch_size,
                ablated_heads=subset_heads,
            )
            train_drop = baseline_train_acc - train_acc
            test_drop = baseline_test_acc - test_acc
            all_rows.append(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_epoch": checkpoint_epoch,
                    "checkpoint_type": checkpoint_type,
                    "checkpoint_threshold": checkpoint_threshold,
                    "subset_index": subset_index,
                    "subset_label": _subset_label(subset_heads),
                    "num_ablated_heads": len(subset_heads),
                    "ablated_heads": _heads_to_string_list(subset_heads),
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "baseline_train_acc": baseline_train_acc,
                    "baseline_test_acc": baseline_test_acc,
                    "train_drop": train_drop,
                    "test_drop": test_drop,
                    "selective_gap": test_drop - train_drop,
                    "strong_h2_pass": (
                        test_acc
                        <= (cfg.surgery.causal_test_chance_multiplier / cfg.model.p)
                        and train_acc >= cfg.surgery.causal_train_floor
                    ),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")
    return all_rows
