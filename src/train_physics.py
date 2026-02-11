import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from .config import Config
from .dataset import ModularAdditionDataset
from .metrics import evaluate_accuracy
from .model import get_model


def clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in state_dict.items()}


def hash_batch_schedule(batch_schedule: torch.Tensor, rows: int = 64) -> str:
    import hashlib

    sample = batch_schedule[:rows].detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(sample).hexdigest()[:16]


def _sample_noise_like(
    param: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    noise_cpu = torch.randn(
        param.shape,
        generator=generator,
        dtype=param.dtype,
        device="cpu",
    )
    return noise_cpu.to(device=param.device)


def _apply_sgld_noise(
    model: torch.nn.Module,
    lr: float,
    temperature: float,
    noise_generator: torch.Generator,
) -> None:
    if temperature <= 0.0:
        return

    scale = math.sqrt(2.0 * lr * temperature)
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            param.add_(scale * _sample_noise_like(param, noise_generator))


def train_physics_run(
    cfg: Config,
    dataset: ModularAdditionDataset,
    initial_state_dict: Mapping[str, torch.Tensor],
    batch_schedule: torch.Tensor,
    noise_generator: torch.Generator,
    temperature: float,
    seed: int,
    max_epochs: int,
    eval_every: int,
    thresholds: Sequence[float],
) -> Dict[str, Any]:
    if batch_schedule.ndim != 2:
        raise ValueError("batch_schedule must have shape [max_epochs, batch_size].")
    if batch_schedule.shape[0] < max_epochs:
        raise ValueError("batch_schedule does not have enough epochs.")
    if batch_schedule.shape[1] != cfg.train.batch_size:
        raise ValueError("batch_schedule batch dimension must match cfg.train.batch_size.")

    model = get_model(cfg, seed=seed)
    model.load_state_dict(clone_state_dict(initial_state_dict))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    train_tokens, train_labels = dataset.train_data()
    test_tokens, test_labels = dataset.test_data()
    thresholds_sorted = sorted(float(value) for value in thresholds)
    grok_epochs: Dict[str, Optional[int]] = {
        f"{value:.2f}": None for value in thresholds_sorted
    }

    history: List[Dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        sample_idx = batch_schedule[epoch]
        batch_tokens = train_tokens[sample_idx].to(model.cfg.device)
        batch_labels = train_labels[sample_idx].to(model.cfg.device)

        logits = model(batch_tokens)[:, -1, :]
        loss = loss_fn(logits, batch_labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _apply_sgld_noise(
            model=model,
            lr=cfg.optim.lr,
            temperature=temperature,
            noise_generator=noise_generator,
        )

        should_eval = (epoch % eval_every == 0) or (epoch == max_epochs - 1)
        if not should_eval:
            continue

        train_acc = evaluate_accuracy(model, train_tokens, train_labels)
        test_acc = evaluate_accuracy(model, test_tokens, test_labels)
        row = {
            "epoch": float(epoch),
            "train_loss": float(loss.item()),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
        }
        history.append(row)

        for threshold in thresholds_sorted:
            threshold_key = f"{threshold:.2f}"
            if grok_epochs[threshold_key] is None and test_acc >= threshold:
                grok_epochs[threshold_key] = epoch

        if all(value is not None for value in grok_epochs.values()):
            break

    return {
        "seed": seed,
        "temperature": temperature,
        "history": history,
        "grok_epochs": grok_epochs,
        "batch_schedule_hash": hash_batch_schedule(batch_schedule),
        "eval_every": eval_every,
        "max_epochs": max_epochs,
    }
