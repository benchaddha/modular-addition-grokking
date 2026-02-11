import inspect
from typing import Optional

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from .config import Config


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model(cfg: Config, seed: Optional[int] = None) -> HookedTransformer:
    cfg.validate()
    model_seed = cfg.train.seed if seed is None else seed
    desired_kwargs = {
        "n_layers": cfg.model.n_layers,
        "d_model": cfg.model.d_model,
        "n_heads": cfg.model.n_heads,
        "d_head": cfg.model.d_head,
        "d_mlp": cfg.model.d_mlp,
        "n_ctx": cfg.model.n_ctx,
        "act_fn": cfg.model.act_fn,
        "normalization_type": cfg.model.normalization_type,
        "d_vocab": cfg.model.vocab_size,
        "d_vocab_out": cfg.model.vocab_size,
        "seed": model_seed,
        "device": get_device(),
    }
    valid_keys = set(inspect.signature(HookedTransformerConfig.__init__).parameters)
    filtered_kwargs = {k: v for k, v in desired_kwargs.items() if k in valid_keys}

    model_cfg = HookedTransformerConfig(
        **filtered_kwargs,
    )
    return HookedTransformer(model_cfg)
