from typing import Dict, Iterable, Optional

import torch
from transformer_lens import HookedTransformer


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return (predictions == labels).float().mean().item()


@torch.inference_mode()
def evaluate_accuracy(
    model: HookedTransformer,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 2048,
) -> float:
    model.eval()
    num_examples = tokens.shape[0]
    correct = 0

    for start in range(0, num_examples, batch_size):
        end = min(start + batch_size, num_examples)
        batch_tokens = tokens[start:end].to(model.cfg.device)
        batch_labels = labels[start:end].to(model.cfg.device)
        logits = model(batch_tokens)[:, -1, :]
        predictions = logits.argmax(dim=-1)
        correct += (predictions == batch_labels).sum().item()

    return correct / num_examples


def find_grok_epoch(
    history: Iterable[Dict[str, float]],
    accuracy_key: str = "test_acc",
    epoch_key: str = "epoch",
    threshold: float = 0.99,
) -> Optional[int]:
    for row in history:
        if row.get(accuracy_key, 0.0) >= threshold:
            return int(row[epoch_key])
    return None
