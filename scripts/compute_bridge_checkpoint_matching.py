"""
T3: Checkpoint-matching diagnostics for bridge cohort.

For each milestone checkpoint (testacc_80/90/95/99, best, final) of both
conditions (T=0 and T=1e-4) of seeds 4, 6, 7, 8, 9:
  - Load model from checkpoint
  - Compute test cross-entropy loss, train cross-entropy loss
  - Test accuracy, train accuracy
  - Global parameter L2 norm
  - Checkpoint epoch
Writes results/metrics/bridge_checkpoint_matching.csv
"""

import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.dataset import get_dataset
from src.model import get_model
from src.surgery import _load_checkpoint_payload

SEEDS_RUNS = [
    # (seed, run_dir, t0_artifacts, t1e4_artifacts)
    (4,  "run2", "results/physics_checkpoint_bridge_run2/seed_4/t0/artifacts",
               "results/physics_checkpoint_bridge_run2/seed_4/t1e-04/artifacts"),
    (6,  "run2", "results/physics_checkpoint_bridge_run2/seed_6/t0/artifacts",
               "results/physics_checkpoint_bridge_run2/seed_6/t1e-04/artifacts"),
    (7,  "run3", "results/physics_checkpoint_bridge_run3/seed_7/t0/artifacts",
               "results/physics_checkpoint_bridge_run3/seed_7/t1e-04/artifacts"),
    (8,  "run3", "results/physics_checkpoint_bridge_run3/seed_8/t0/artifacts",
               "results/physics_checkpoint_bridge_run3/seed_8/t1e-04/artifacts"),
    (9,  "run3", "results/physics_checkpoint_bridge_run3/seed_9/t0/artifacts",
               "results/physics_checkpoint_bridge_run3/seed_9/t1e-04/artifacts"),
]

# Milestone tags in order
MILESTONES = ["testacc_80", "testacc_90", "testacc_95", "testacc_99", "best", "final"]

BASE_CFG_PATH = PROJECT_ROOT / "configs" / "physics.yaml"


@torch.inference_mode()
def _evaluate(model, tokens, labels, batch_size=2048):
    """Returns (accuracy, cross_entropy_loss) on a split."""
    model.eval()
    total = tokens.shape[0]
    correct = 0
    total_loss = 0.0
    device = next(model.parameters()).device

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        bt = tokens[start:end].to(device)
        bl = labels[start:end].to(device)
        logits = model(bt)[:, -1, :]  # final position
        preds = logits.argmax(dim=-1)
        correct += int((preds == bl).sum().item())
        total_loss += F.cross_entropy(logits, bl, reduction="sum").item()

    acc = correct / total
    ce_loss = total_loss / total
    return acc, ce_loss


def _param_l2_norm(model):
    total = 0.0
    for p in model.parameters():
        total += float(p.norm(2).item() ** 2)
    return total ** 0.5


def process_checkpoint(ckpt_path: Path, seed: int, temperature: float, milestone: str):
    """Load checkpoint, build model+dataset from embedded config, evaluate."""
    if not ckpt_path.exists():
        return None

    payload = _load_checkpoint_payload(ckpt_path)
    epoch = payload.get("epoch", None)

    # Build config: merge embedded cfg with base for reproducible dataset split
    base_cfg = Config.from_yaml(str(BASE_CFG_PATH))
    ckpt_cfg_dict = payload.get("cfg", {})
    if isinstance(ckpt_cfg_dict, dict) and ckpt_cfg_dict:
        merged = base_cfg.to_dict()
        for k in ckpt_cfg_dict:
            merged[k] = ckpt_cfg_dict[k]
        cfg = Config.from_dict(merged)
    else:
        cfg = base_cfg
        cfg.train.seed = seed

    dataset = get_dataset(cfg)
    model = get_model(cfg)
    state_dict = payload.get("model_state_dict", payload)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    train_tokens, train_labels = dataset.train_data()
    test_tokens, test_labels = dataset.test_data()

    train_acc, train_ce = _evaluate(model, train_tokens, train_labels)
    test_acc, test_ce = _evaluate(model, test_tokens, test_labels)
    l2_norm = _param_l2_norm(model)

    return {
        "seed": seed,
        "temperature": temperature,
        "milestone": milestone,
        "epoch": epoch,
        "train_acc": round(train_acc, 6),
        "test_acc": round(test_acc, 6),
        "train_ce_loss": round(train_ce, 6),
        "test_ce_loss": round(test_ce, 6),
        "param_l2_norm": round(l2_norm, 4),
        "checkpoint_path": str(ckpt_path.relative_to(PROJECT_ROOT)),
    }


def main():
    out_dir = PROJECT_ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bridge_checkpoint_matching.csv"

    rows = []
    for seed, run_tag, t0_dir_str, t1e4_dir_str in SEEDS_RUNS:
        for temperature, temp_tag, arts_dir_str in [
            (0.0, "t0", t0_dir_str),
            (1e-4, "t1e-04", t1e4_dir_str),
        ]:
            arts_dir = PROJECT_ROOT / arts_dir_str
            for milestone in MILESTONES:
                fname = f"physics_seed{seed}_{temp_tag}_{milestone}.pt"
                ckpt_path = arts_dir / fname
                print(f"  Processing seed={seed} T={temperature} milestone={milestone} ...", flush=True)
                row = process_checkpoint(ckpt_path, seed, temperature, milestone)
                if row is None:
                    print(f"    [MISSING] {ckpt_path}")
                else:
                    rows.append(row)
                    print(f"    train_acc={row['train_acc']:.4f} test_acc={row['test_acc']:.4f} "
                          f"epoch={row['epoch']} L2={row['param_l2_norm']:.2f}")

    if not rows:
        print("No rows produced — check checkpoint paths")
        return

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
