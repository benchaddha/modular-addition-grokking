"""Exact cached suffix model for repeated pre-unembedding interventions."""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Iterator, List, Sequence, Tuple

import torch

from .calibrated_separability import tensor_checksum


class CachedPreUnembedModel(torch.nn.Module):
    """Cache the expensive transformer prefix and reuse its exact output suffix.

    For `pre_unembed`, the intervention is at the final token's last-block
    `hook_resid_post`. With normalization disabled in the canonical models, the
    remaining computation is the model's final normalization module followed
    by unembedding. This wrapper calls those original modules, and verifies its
    clean logits against the full model before any search uses it.
    """

    def __init__(
        self,
        source_model: torch.nn.Module,
        representations: torch.Tensor,
        *,
        p: int,
        hook_name: str,
    ) -> None:
        super().__init__()
        self.source_model = source_model
        self.register_buffer("representations", representations.float())
        self.p = p
        self.hook_name = hook_name
        self.cfg = source_model.cfg
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
        for name, hook_fn in self._active_hooks:
            if name == self.hook_name:
                hidden = hook_fn(hidden, None)
        # TransformerLens omits ``ln_final`` entirely when normalization_type
        # is None (the pinned canonical configuration).
        if hasattr(self.source_model, "ln_final"):
            hidden = self.source_model.ln_final(hidden)
        return self.source_model.unembed(hidden)


@torch.inference_mode()
def build_cached_preunembed_model(
    model: torch.nn.Module,
    all_tokens: torch.Tensor,
    *,
    p: int,
    hook_name: str,
    position: int,
    batch_size: int,
    tolerance: float = 1e-5,
) -> Tuple[CachedPreUnembedModel, Dict[str, Any]]:
    model.eval()
    residual_chunks: List[torch.Tensor] = []
    full_logits_chunks: List[torch.Tensor] = []

    def capture(value: torch.Tensor, hook: Any) -> torch.Tensor:
        residual_chunks.append(value[:, position, :].detach().cpu().clone())
        return value

    with model.hooks(fwd_hooks=[(hook_name, capture)]):
        for start in range(0, len(all_tokens), batch_size):
            end = min(start + batch_size, len(all_tokens))
            logits = model(all_tokens[start:end].to("cpu"))[:, -1, :]
            full_logits_chunks.append(logits.detach().cpu())
    representations = torch.cat(residual_chunks, dim=0)
    full_logits = torch.cat(full_logits_chunks, dim=0)
    cached = CachedPreUnembedModel(
        model,
        representations,
        p=p,
        hook_name=hook_name,
    )
    cached_logits_chunks = []
    for start in range(0, len(all_tokens), batch_size):
        end = min(start + batch_size, len(all_tokens))
        cached_logits_chunks.append(cached(all_tokens[start:end])[:, -1, :].detach().cpu())
    cached_logits = torch.cat(cached_logits_chunks, dim=0)
    max_abs_error = float((full_logits - cached_logits).abs().max().item())
    predictions_equal = bool(
        torch.equal(full_logits.argmax(dim=-1), cached_logits.argmax(dim=-1))
    )
    if max_abs_error > tolerance or not predictions_equal:
        raise RuntimeError(
            f"Cached pre_unembed suffix verification failed: max_abs_error={max_abs_error}, "
            f"predictions_equal={predictions_equal}"
        )
    diagnostics = {
        "n_examples": len(all_tokens),
        "representation_shape": list(representations.shape),
        "representation_checksum": tensor_checksum(representations),
        "max_abs_logit_error": max_abs_error,
        "predictions_equal": predictions_equal,
        "tolerance": tolerance,
    }
    return cached, diagnostics
