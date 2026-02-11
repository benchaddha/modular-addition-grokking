from dataclasses import dataclass
from typing import Optional, Tuple

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
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < frac_train < 1.0:
        raise ValueError("frac_train must be in the open interval (0, 1).")

    if generator is None:
        if seed is None:
            raise ValueError("Either seed or generator must be provided.")
        generator = torch.Generator()
        generator.manual_seed(seed)

    permutation = torch.randperm(num_examples, generator=generator)
    num_train = int(num_examples * frac_train)
    train_indices = permutation[:num_train]
    test_indices = permutation[num_train:]

    return train_indices, test_indices


def build_batch_schedule(
    train_size: int,
    batch_size: int,
    max_epochs: int,
    data_seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if train_size <= 0:
        raise ValueError("train_size must be > 0.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be > 0.")

    if generator is None:
        if data_seed is None:
            raise ValueError("Either data_seed or generator must be provided.")
        generator = torch.Generator()
        generator.manual_seed(data_seed)

    return torch.randint(
        low=0,
        high=train_size,
        size=(max_epochs, batch_size),
        generator=generator,
        dtype=torch.long,
    )


def get_dataset(
    cfg: Config,
    data_seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> ModularAdditionDataset:
    tokens, labels = build_modular_addition_tokens(cfg.model.p)
    split_seed = cfg.train.seed if data_seed is None else data_seed
    train_indices, test_indices = train_test_split_indices(
        num_examples=tokens.shape[0],
        frac_train=cfg.data.frac_train,
        seed=split_seed,
        generator=generator,
    )
    return ModularAdditionDataset(
        tokens=tokens,
        labels=labels,
        train_indices=train_indices,
        test_indices=test_indices,
    )
