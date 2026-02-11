import os
import random
from typing import Dict, List

import numpy as np
import torch
import wandb
from tqdm import tqdm

from .config import Config
from .dataset import get_dataset
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

    history: List[Dict[str, float]] = []
    train_size = train_tokens.shape[0]
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
                history.append(row)
                wandb.log(row, step=epoch)
                pbar.set_description(
                    f"epoch={epoch} train={train_acc:.3f} test={test_acc:.3f}"
                )
    finally:
        run.finish()

    return history


if __name__ == "__main__":
    train(Config())
