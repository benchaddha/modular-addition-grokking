import os
import random
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import wandb
from tqdm import tqdm

from .config import Config
from .dataset import get_dataset
from .fft_logging import append_fft_log
from .metrics import evaluate_accuracy
from .model import get_model


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg: Config) -> List[Dict[str, float]]:
    cfg.validate()
    set_global_seed(cfg.train.seed)

    dataset = get_dataset(cfg)
    model = get_model(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    train_tokens, train_labels = dataset.train_data()
    test_tokens, test_labels = dataset.test_data()

    run = wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.logging.run_name,
        config=cfg.to_dict(),
        mode=os.getenv("WANDB_MODE", "offline"),
    )
    run_id = run.id or f"seed-{cfg.train.seed}"

    checkpoints_dir = Path("results") / "checkpoints"
    metrics_dir = Path("results") / "metrics"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{run_id}.jsonl"
    fft_metrics_path = metrics_dir / f"{run_id}_fft.jsonl"

    history: List[Dict[str, float]] = []
    train_size = train_tokens.shape[0]
    best_test_acc = float("-inf")
    pbar = tqdm(range(cfg.train.epochs), desc="training")

    try:
        for epoch in pbar:
            model.train()

            sample_idx = torch.randint(0, train_size, (cfg.train.batch_size,))
            batch_tokens = train_tokens[sample_idx].to(model.cfg.device)
            batch_labels = train_labels[sample_idx].to(model.cfg.device)

            logits = model(batch_tokens)[:, -1, :]
            loss = loss_fn(logits, batch_labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            should_eval = (
                epoch % cfg.train.eval_every == 0
                or epoch == cfg.train.epochs - 1
            )
            if should_eval:
                train_acc = evaluate_accuracy(model, train_tokens, train_labels)
                test_acc = evaluate_accuracy(model, test_tokens, test_labels)
                row = {
                    "epoch": float(epoch),
                    "train_loss": float(loss.item()),
                    "train_acc": float(train_acc),
                    "test_acc": float(test_acc),
                }
                fft_summary = append_fft_log(
                    model=model,
                    tensor_path="blocks.0.attn.W_Q",
                    epoch=epoch,
                    output_path=fft_metrics_path,
                )
                row.update(fft_summary)
                history.append(row)
                wandb.log(row, step=epoch)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    torch.save(
                        {
                            "epoch": epoch,
                            "cfg": cfg.to_dict(),
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "metrics": row,
                        },
                        checkpoints_dir / f"{run_id}_best.pt",
                    )
                pbar.set_description(
                    f"epoch={epoch} train={train_acc:.3f} test={test_acc:.3f}"
                )
    finally:
        torch.save(
            {
                "epoch": cfg.train.epochs - 1,
                "cfg": cfg.to_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": history[-1] if history else None,
            },
            checkpoints_dir / f"{run_id}_final.pt",
        )
        run.finish()

    return history


if __name__ == "__main__":
    train(Config())
