from dataclasses import dataclass
from typing import Tuple

import torch

from .config import Config


@dataclass
class ModularAdditionDataset:
    tokens: torch.Tensor
    labels: torch.Tensor
    train_indices: torch.Tensor
    test_indices: torch.Tensor

    def train_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[self.train_indices], self.labels[self.train_indices]

    def test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[self.test_indices], self.labels[self.test_indices]


def build_modular_addition_tokens(p: int) -> Tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(p, dtype=torch.long)
    x_vals, y_vals = torch.meshgrid(values, values, indexing="ij")

    x_flat = x_vals.reshape(-1)
    y_flat = y_vals.reshape(-1)
    eq_token = torch.full_like(x_flat, fill_value=p)

    tokens = torch.stack([x_flat, y_flat, eq_token], dim=-1)
    labels = (x_flat + y_flat) % p
    return tokens, labels


def train_test_split_indices(
    num_examples: int,
    frac_train: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < frac_train < 1.0:
        raise ValueError("frac_train must be in the open interval (0, 1).")

    generator = torch.Generator()
    generator.manual_seed(seed)

    permutation = torch.randperm(num_examples, generator=generator)
    num_train = int(num_examples * frac_train)
    train_indices = permutation[:num_train]
    test_indices = permutation[num_train:]

    return train_indices, test_indices


def get_dataset(cfg: Config) -> ModularAdditionDataset:
    tokens, labels = build_modular_addition_tokens(cfg.model.p)
    train_indices, test_indices = train_test_split_indices(
        num_examples=tokens.shape[0],
        frac_train=cfg.data.frac_train,
        seed=cfg.train.seed,
    )
    return ModularAdditionDataset(
        tokens=tokens,
        labels=labels,
        train_indices=train_indices,
        test_indices=test_indices,
    )
