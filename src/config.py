from dataclasses import dataclass

@dataclass
class Config:
    p: int = 113          # The prime modulus
    d_model: int = 128    # Transformer width
    n_layers: int = 1     # Tiny model for interpretability
    n_heads: int = 4
    d_head: int = 32
    d_mlp: int = 512
    seed: int = 42
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1.0 # High weight decay is CRUCIAL for grokking
    epochs: int = 10000
    frac_train: float = 0.3   # Train on only 30% of data to force generalization