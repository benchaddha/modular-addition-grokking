from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
class PhysicsConfig:
    temperatures: List[float] = field(
        default_factory=lambda: [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    )
    seeds: List[int] = field(default_factory=lambda: list(range(10)))
    max_epochs: int = 20_000
    eval_every: int = 100
    grok_thresholds: List[float] = field(default_factory=lambda: [0.95, 0.99])
    noise_seed_offset: int = 10_000


@dataclass
class SurgeryConfig:
    checkpoint_paths: List[str] = field(
        default_factory=lambda: ["results/checkpoints/replace_with_checkpoint.pt"]
    )
    probe_split: str = "test"
    probe_max_examples: int = 0
    top_k: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    random_control_repeats: int = 3
    eval_batch_size: int = 2048
    min_baseline_train_acc: float = 0.95
    min_baseline_test_acc: float = 0.95
    causal_train_floor: float = 0.90
    causal_test_chance_multiplier: float = 2.0
    seed: int = 123


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    surgery: SurgeryConfig = field(default_factory=SurgeryConfig)

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
        if self.physics.max_epochs <= 0:
            raise ValueError("physics.max_epochs must be > 0.")
        if self.physics.eval_every <= 0:
            raise ValueError("physics.eval_every must be > 0.")
        if not self.physics.temperatures:
            raise ValueError("physics.temperatures must be non-empty.")
        if not self.physics.seeds:
            raise ValueError("physics.seeds must be non-empty.")
        for threshold in self.physics.grok_thresholds:
            if not 0.0 < threshold <= 1.0:
                raise ValueError("physics.grok_thresholds must be in the range (0, 1].")
        self.validate_surgery()

    def validate_surgery(self) -> None:
        if not self.surgery.checkpoint_paths:
            raise ValueError("surgery.checkpoint_paths must be non-empty.")
        if self.surgery.probe_split not in {"train", "test"}:
            raise ValueError("surgery.probe_split must be either 'train' or 'test'.")
        if self.surgery.probe_max_examples < 0:
            raise ValueError("surgery.probe_max_examples must be >= 0.")
        if self.surgery.eval_batch_size <= 0:
            raise ValueError("surgery.eval_batch_size must be > 0.")
        if self.surgery.random_control_repeats < 0:
            raise ValueError("surgery.random_control_repeats must be >= 0.")
        if not 0.0 < self.surgery.causal_train_floor <= 1.0:
            raise ValueError("surgery.causal_train_floor must be in the range (0, 1].")
        if self.surgery.causal_test_chance_multiplier <= 0.0:
            raise ValueError("surgery.causal_test_chance_multiplier must be > 0.")
        if not self.surgery.top_k:
            raise ValueError("surgery.top_k must be non-empty.")
        if any(k <= 0 for k in self.surgery.top_k):
            raise ValueError("surgery.top_k values must be > 0.")
        unique_sorted = sorted(set(self.surgery.top_k))
        if unique_sorted != self.surgery.top_k:
            raise ValueError("surgery.top_k must be unique and sorted ascending.")


def _update_dataclass(instance: Any, updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
