import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from .config import Config
from .dataset import ModularAdditionDataset
from .metrics import evaluate_accuracy
from .model import get_model
from .train import _checkpoint_payload, _checkpoint_suffix_for_threshold


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


def train_physics_checkpointed_run(
    cfg: Config,
    dataset: ModularAdditionDataset,
    initial_state_dict: Mapping[str, torch.Tensor],
    batch_schedule: torch.Tensor,
    noise_generator: torch.Generator,
    temperature: float,
    seed: int,
    max_epochs: int,
    eval_every: int,
    run_id: str,
    checkpoints_dir: Path,
    metrics_path: Path,
    stop_on_all_thresholds: bool = False,
) -> Dict[str, Any]:
    if batch_schedule.ndim != 2:
        raise ValueError("batch_schedule must have shape [max_epochs, batch_size].")
    if batch_schedule.shape[0] < max_epochs:
        raise ValueError("batch_schedule does not have enough epochs.")
    if batch_schedule.shape[1] != cfg.train.batch_size:
        raise ValueError("batch_schedule batch dimension must match cfg.train.batch_size.")

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

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

    history: List[Dict[str, float]] = []
    best_test_acc = float("-inf")
    saved_milestones = set()
    final_epoch = max_epochs - 1

    for epoch in range(max_epochs):
        final_epoch = epoch
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
            "temperature": float(temperature),
            "seed": float(seed),
        }
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        for threshold in cfg.train.checkpoint_milestones:
            if threshold in saved_milestones or test_acc < threshold:
                continue
            saved_milestones.add(threshold)
            payload = _checkpoint_payload(
                cfg=cfg,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics=row,
                checkpoint_type="milestone",
                checkpoint_threshold=threshold,
            )
            payload["physics_temperature"] = float(temperature)
            payload["physics_seed"] = int(seed)
            payload["run_id"] = run_id
            torch.save(
                payload,
                checkpoints_dir / f"{run_id}_{_checkpoint_suffix_for_threshold(threshold)}.pt",
            )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            payload = _checkpoint_payload(
                cfg=cfg,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics=row,
                checkpoint_type="best",
            )
            payload["physics_temperature"] = float(temperature)
            payload["physics_seed"] = int(seed)
            payload["run_id"] = run_id
            torch.save(payload, checkpoints_dir / f"{run_id}_best.pt")

        if stop_on_all_thresholds and len(saved_milestones) == len(cfg.train.checkpoint_milestones):
            break

    final_metrics = history[-1] if history else None
    payload = _checkpoint_payload(
        cfg=cfg,
        epoch=final_epoch,
        model=model,
        optimizer=optimizer,
        metrics=final_metrics,
        checkpoint_type="final",
    )
    payload["physics_temperature"] = float(temperature)
    payload["physics_seed"] = int(seed)
    payload["run_id"] = run_id
    torch.save(payload, checkpoints_dir / f"{run_id}_final.pt")

    return {
        "seed": seed,
        "temperature": temperature,
        "history": history,
        "batch_schedule_hash": hash_batch_schedule(batch_schedule),
        "eval_every": eval_every,
        "max_epochs": max_epochs,
        "best_test_acc": max((row["test_acc"] for row in history), default=None),
        "last_test_acc": history[-1]["test_acc"] if history else None,
        "final_epoch": final_epoch,
    }
