"""Ground-truth residual-stream construction for separability calibration.

Let C be a p by (p-1) zero-mean simplex code with orthonormal columns. For
class y, c_y is row y of C. We sample orthogonal column bases A and B_perp and
set B_theta = cos(theta) A + sin(theta) B_perp. Positive-control residuals are

    h(x) = s_general A c_y
           - I[x in original train split] s_memory B_theta c_y + noise.

The unembedding is proportional to (A-B_theta) C^T. Consequently A c_y alone
and -B_theta c_y alone both give the correct class for every theta > 0. At 90
degrees, removing A erases held-out signal exactly while the orthogonal memory
branch preserves training predictions. A tiny deterministic wrong-class logit
breaks otherwise exact zero-logit ties without supplying correct behavior.

The model exposes a TransformerLens-like hook context so the exact calibrated
projection/QR/search implementation is exercised rather than bypassed.
"""

from __future__ import annotations

import contextlib
import math
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

from .dataset import build_modular_addition_tokens, train_test_split_indices


SYNTHETIC_HOOK_NAME = "synthetic.hook_residual"


def simplex_codes(p: int) -> torch.Tensor:
    """Return a p by (p-1) zero-mean simplex code with orthonormal columns."""

    if p < 2:
        raise ValueError("p must be at least two.")
    contrasts = torch.zeros(p, p - 1, dtype=torch.float64)
    for column in range(p - 1):
        contrasts[column, column] = 1.0
        contrasts[-1, column] = -1.0
    return torch.linalg.qr(contrasts, mode="reduced").Q.float()


class SyntheticResidualModel(torch.nn.Module):
    def __init__(
        self,
        representations: torch.Tensor,
        readout: torch.Tensor,
        tie_logits: torch.Tensor,
        *,
        p: int,
    ) -> None:
        super().__init__()
        self.register_buffer("representations", representations.float())
        self.register_buffer("readout", readout.float())
        self.register_buffer("tie_logits", tie_logits.float())
        self.cfg = SimpleNamespace(
            d_model=int(representations.shape[1]),
            d_vocab=p,
            d_vocab_out=p,
            n_ctx=3,
            device="cpu",
        )
        self.p = p
        self._active_hooks: List[Tuple[str, Callable]] = []

    @contextlib.contextmanager
    def hooks(self, *, fwd_hooks: List[Tuple[str, Callable]]) -> Iterator[None]:
        previous = self._active_hooks
        self._active_hooks = list(fwd_hooks)
        try:
            yield
        finally:
            self._active_hooks = previous

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.long().cpu()
        example_ids = tokens[:, 0] * self.p + tokens[:, 1]
        hidden = self.representations[example_ids].unsqueeze(1)
        for hook_name, hook_fn in self._active_hooks:
            if hook_name == SYNTHETIC_HOOK_NAME:
                hidden = hook_fn(hidden, None)
        logits = hidden @ self.readout
        logits = logits + self.tie_logits[example_ids].unsqueeze(1)
        return logits


def _random_orthogonal_bases(
    d_model: int,
    q: int,
    *,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if 2 * q > d_model:
        raise ValueError("Synthetic construction requires 2*(p-1) <= d_model.")
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(d_model, 2 * q, generator=generator, dtype=torch.float64)
    orthogonal = torch.linalg.qr(matrix, mode="reduced").Q.float()
    return orthogonal[:, :q], orthogonal[:, q : 2 * q]


def build_synthetic_model(
    *,
    p: int,
    theta_degrees: float,
    data_seed: int,
    construction_seed: int,
    d_model: int = 128,
    s_general: float = 1.0,
    s_memory: float = 1.0,
    readout_scale: float = 8.0,
    noise_std: float = 0.0,
    negative_control: bool = False,
) -> Tuple[SyntheticResidualModel, Dict[str, Any]]:
    tokens, labels = build_modular_addition_tokens(p)
    train_indices, test_indices = train_test_split_indices(
        int(tokens.shape[0]), 0.30, seed=data_seed
    )
    q = p - 1
    codes = simplex_codes(p)
    a_basis, b_perp = _random_orthogonal_bases(
        d_model, q, seed=construction_seed
    )
    theta_radians = math.radians(theta_degrees)
    b_theta = math.cos(theta_radians) * a_basis + math.sin(theta_radians) * b_perp

    label_codes = codes[labels]
    general = label_codes @ a_basis.T
    train_mask = torch.zeros(int(tokens.shape[0]), dtype=torch.bool)
    train_mask[train_indices] = True

    if negative_control:
        representations = s_general * general
        readout = readout_scale * (a_basis @ codes.T)
    else:
        memory = label_codes @ b_theta.T
        representations = s_general * general
        representations[train_mask] -= s_memory * memory[train_mask]
        readout = readout_scale * ((a_basis - b_theta) @ codes.T)

    if noise_std > 0:
        generator = torch.Generator().manual_seed(construction_seed + 1)
        representations = representations + noise_std * torch.randn(
            representations.shape, generator=generator
        )

    # If an intervention removes all signal, predict y+1 rather than relying
    # on argmax's class-zero tie behavior. The 1e-4 scale cannot overturn any
    # clean class margin in the prespecified construction.
    tie_logits = torch.zeros(int(tokens.shape[0]), p)
    tie_logits[torch.arange(int(tokens.shape[0])), (labels + 1) % p] = 1e-4

    model = SyntheticResidualModel(
        representations,
        readout,
        tie_logits,
        p=p,
    )
    metadata: Dict[str, Any] = {
        "p": p,
        "q": q,
        "theta_degrees": theta_degrees,
        "data_seed": data_seed,
        "construction_seed": construction_seed,
        "d_model": d_model,
        "s_general": s_general,
        "s_memory": s_memory,
        "readout_scale": readout_scale,
        "noise_std": noise_std,
        "negative_control": negative_control,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "tokens": tokens,
        "labels": labels,
        "class_codes": codes,
        "A": a_basis,
        "B_perp": b_perp,
        "B_theta": b_theta,
    }
    return model, metadata
