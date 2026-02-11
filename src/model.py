import inspect

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from .config import Config


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model(cfg: Config) -> HookedTransformer:
    cfg.validate()
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
        "seed": cfg.train.seed,
        "device": get_device(),
    }
    valid_keys = set(inspect.signature(HookedTransformerConfig.__init__).parameters)
    filtered_kwargs = {k: v for k, v in desired_kwargs.items() if k in valid_keys}

    model_cfg = HookedTransformerConfig(
        **filtered_kwargs,
    )
    return HookedTransformer(model_cfg)
