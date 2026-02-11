from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelConfig:
    p: int = 113
    d_model: int = 128
    n_layers: int = 1
    n_heads: int = 4
    d_head: int = 32
    d_mlp: int = 512
    n_ctx: int = 3
    act_fn: str = "relu"
    normalization_type: Optional[str] = None

    @property
    def vocab_size(self) -> int:
        # Tokens are x, y, and "=" as the delimiter token id p.
        return self.p + 1


@dataclass
class DataConfig:
    frac_train: float = 0.3


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 1.0


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 10_000
    eval_every: int = 100


@dataclass
class LoggingConfig:
    wandb_project: str = "grokking-demo"
    run_name: Optional[str] = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "Config":
        cfg = cls()
        if data:
            _update_dataclass(cfg, data)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError("YAML config must deserialize to a mapping.")
        return cls.from_dict(payload)

    def validate(self) -> None:
        if not 0.0 < self.data.frac_train < 1.0:
            raise ValueError("data.frac_train must be in the open interval (0, 1).")
        if self.model.d_model != self.model.n_heads * self.model.d_head:
            raise ValueError("model.d_model must equal model.n_heads * model.d_head.")
        if self.train.batch_size <= 0:
            raise ValueError("train.batch_size must be > 0.")


def _update_dataclass(instance: Any, updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
