"""Strategy 2: canonical-regime (full-batch) grokking runs.

Unlike the default minibatch trainer (which samples 64 examples with
replacement per step and never produces a clean train~1 / test~chance
plateau), this script performs FULL-BATCH AdamW updates: every train
example participates in every optimizer step. In this regime train
accuracy saturates early while test accuracy sits near chance (1/p) for
thousands of epochs before grokking.

Because test-accuracy milestones alone cannot capture the plateau phase,
the trainer also checkpoints on train-accuracy milestones and on periodic
epoch snapshots. Checkpoint payloads are byte-compatible with the existing
downstream loaders (they reuse ``_checkpoint_payload`` / ``_checkpoint_suffix_for_threshold``
from ``src.train``), so intervention scripts can load them unchanged.

No wandb.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config
from src.dataset import get_dataset
from src.metrics import evaluate_accuracy, find_grok_epoch
from src.model import get_model
from src.train import _checkpoint_payload, _checkpoint_suffix_for_threshold


TRAIN_ACC_MILESTONES = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strategy 2 canonical-regime (full-batch) grokking trainer with "
            "test-acc, train-acc, and periodic-snapshot checkpointing."
        )
    )
    parser.add_argument("--config", type=str, required=True, help="Base config YAML path.")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds to run.")
    parser.add_argument(
        "--cell-name",
        type=str,
        required=True,
        help="Short cell identifier used in run ids and default output dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/strategy2/<cell-name>).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override cfg.train.epochs.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Override cfg.train.eval_every.",
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.98, help="AdamW beta2.")
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=2000,
        help="Periodic model-only snapshot cadence in epochs.",
    )
    parser.add_argument(
        "--post-grok-evals",
        type=int,
        default=10,
        help=(
            "After the highest test milestone is saved, run this many further "
            "evaluations then stop early."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun even if a seed summary.json already exists.",
    )
    return parser.parse_args()


def _train_acc_suffix(threshold: float) -> str:
    percentage = int(round(threshold * 100))
    return f"trainacc_{percentage:02d}"


def _full_batch_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_tokens: torch.Tensor,
    train_labels: torch.Tensor,
) -> float:
    """One full-batch AdamW step. Falls back to 2-chunk gradient accumulation
    on MPS/CUDA out-of-memory. Returns the (full-batch mean) loss value."""
    optimizer.zero_grad(set_to_none=True)
    try:
        logits = model(train_tokens)[:, -1, :]
        loss = loss_fn(logits, train_labels)
        loss.backward()
        loss_value = float(loss.item())
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" not in message and "mps" not in message:
            raise
        # Fallback: split into two chunks and accumulate the gradient. Each
        # chunk's mean loss is weighted by its fraction of the full batch so
        # the accumulated gradient equals the full-batch mean gradient.
        optimizer.zero_grad(set_to_none=True)
        n = train_tokens.shape[0]
        half = n // 2
        loss_value = 0.0
        for start, end in ((0, half), (half, n)):
            weight = (end - start) / n
            chunk_logits = model(train_tokens[start:end])[:, -1, :]
            chunk_loss = loss_fn(chunk_logits, train_labels[start:end]) * weight
            chunk_loss.backward()
            loss_value += float(chunk_loss.item())
    optimizer.step()
    return loss_value


def run_seed(
    args: argparse.Namespace,
    cell: str,
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    cfg = Config.from_yaml(args.config)
    cfg.train.seed = seed
    if args.max_epochs is not None:
        cfg.train.epochs = args.max_epochs
    if args.eval_every is not None:
        cfg.train.eval_every = args.eval_every

    max_epochs = cfg.train.epochs
    eval_every = cfg.train.eval_every

    dataset = get_dataset(cfg)
    model = get_model(cfg, seed=seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    device = model.cfg.device
    train_tokens, train_labels = dataset.train_data()
    test_tokens, test_labels = dataset.test_data()
    # Move the full train set to the model device once, outside the loop.
    train_tokens = train_tokens.to(device)
    train_labels = train_labels.to(device)

    run_id = f"strategy2_{cell}_seed{seed}"
    run_dir = output_dir / f"seed_{seed}"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts_dir / f"{run_id}.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    milestones = list(cfg.train.checkpoint_milestones)
    highest_test_milestone = max(milestones)

    history: List[Dict[str, float]] = []
    milestone_epochs: Dict[str, int] = {}
    saved_test_milestones: set = set()
    saved_train_milestones: set = set()
    best_test_acc = float("-inf")
    post_grok_counter: Optional[int] = None
    epochs_run = 0
    final_row: Optional[Dict[str, float]] = None

    for epoch in range(max_epochs):
        epochs_run = epoch
        model.train()
        loss_value = _full_batch_step(
            model, optimizer, loss_fn, train_tokens, train_labels
        )

        is_final_epoch = epoch == max_epochs - 1
        should_eval = (epoch % eval_every == 0) or is_final_epoch

        # Periodic model-only snapshot.
        if args.snapshot_every > 0 and epoch % args.snapshot_every == 0:
            snapshot = {
                "epoch": epoch,
                "cfg": cfg.to_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
                "metrics": history[-1] if history else None,
                "checkpoint_type": "snapshot",
                "checkpoint_threshold": None,
            }
            torch.save(snapshot, artifacts_dir / f"{run_id}_epoch_{epoch:06d}.pt")

        if not should_eval:
            continue

        train_acc = evaluate_accuracy(model, train_tokens, train_labels)
        test_acc = evaluate_accuracy(model, test_tokens, test_labels)
        row = {
            "epoch": float(epoch),
            "train_loss": float(loss_value),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
        }
        history.append(row)
        final_row = row
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        # (a) TEST-acc milestones.
        for threshold in milestones:
            if threshold in saved_test_milestones or test_acc < threshold:
                continue
            saved_test_milestones.add(threshold)
            suffix = _checkpoint_suffix_for_threshold(threshold)
            milestone_epochs[suffix] = epoch
            torch.save(
                _checkpoint_payload(
                    cfg=cfg,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    metrics=row,
                    checkpoint_type="milestone",
                    checkpoint_threshold=threshold,
                ),
                artifacts_dir / f"{run_id}_{suffix}.pt",
            )

        # (b) TRAIN-acc milestones.
        for threshold in TRAIN_ACC_MILESTONES:
            if threshold in saved_train_milestones or train_acc < threshold:
                continue
            saved_train_milestones.add(threshold)
            suffix = _train_acc_suffix(threshold)
            milestone_epochs[suffix] = epoch
            torch.save(
                _checkpoint_payload(
                    cfg=cfg,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    metrics=row,
                    checkpoint_type="train_milestone",
                    checkpoint_threshold=threshold,
                ),
                artifacts_dir / f"{run_id}_{suffix}.pt",
            )

        # (d) best checkpoint.
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                _checkpoint_payload(
                    cfg=cfg,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    metrics=row,
                    checkpoint_type="best",
                ),
                artifacts_dir / f"{run_id}_best.pt",
            )

        print(
            f"[{cell} seed={seed}] epoch={epoch} "
            f"train={train_acc:.4f} test={test_acc:.4f} loss={loss_value:.4f}",
            flush=True,
        )

        # Early stop: once the highest test milestone is saved, run
        # post_grok_evals more evaluations then break.
        if highest_test_milestone in saved_test_milestones:
            if post_grok_counter is None:
                post_grok_counter = 0
            else:
                post_grok_counter += 1
            if post_grok_counter >= args.post_grok_evals:
                break

    # (d) final checkpoint.
    torch.save(
        _checkpoint_payload(
            cfg=cfg,
            epoch=epochs_run,
            model=model,
            optimizer=optimizer,
            metrics=final_row,
            checkpoint_type="final",
        ),
        artifacts_dir / f"{run_id}_final.pt",
    )

    # Plateau statistics.
    plateau_epochs = [
        int(r["epoch"])
        for r in history
        if r["train_acc"] >= 0.98 and r["test_acc"] <= 0.05
    ]
    plateau_evals = len(plateau_epochs)
    plateau_epoch_span = (
        (max(plateau_epochs) - min(plateau_epochs)) if plateau_epochs else 0
    )

    checkpoint_files = sorted(str(path) for path in artifacts_dir.glob("*.pt"))

    summary = {
        "seed": seed,
        "cell": cell,
        "run_id": run_id,
        "epochs_run": epochs_run,
        "best_test_acc": (max((r["test_acc"] for r in history)) if history else None),
        "last_train_acc": (final_row["train_acc"] if final_row else None),
        "last_test_acc": (final_row["test_acc"] if final_row else None),
        "grok_epoch_95": find_grok_epoch(history, threshold=0.95),
        "grok_epoch_99": find_grok_epoch(history, threshold=0.99),
        "plateau_evals": plateau_evals,
        "plateau_epoch_span": plateau_epoch_span,
        "plateau_detected": plateau_evals >= 5,
        "milestone_epochs": milestone_epochs,
        "checkpoint_files": checkpoint_files,
        "artifacts_dir": str(artifacts_dir),
        "metrics_file": str(metrics_path),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    cell = args.cell_name
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else REPO_ROOT / "results" / "strategy2" / cell
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {args.config}", flush=True)
    print(f"Cell: {cell}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"betas=({args.beta1}, {args.beta2})", flush=True)

    summary_rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        summary_path = output_dir / f"seed_{seed}" / "summary.json"
        if summary_path.exists() and not args.force:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_rows.append(existing)
            print(f"[skip] seed={seed} already completed", flush=True)
            continue
        summary = run_seed(args, cell, seed, output_dir)
        summary_rows.append(summary)
        print(
            f"[done] seed={seed} epochs_run={summary['epochs_run']} "
            f"best_test_acc={summary['best_test_acc']} "
            f"grok99={summary['grok_epoch_99']} "
            f"plateau_detected={summary['plateau_detected']}",
            flush=True,
        )

    csv_fields = [
        "seed",
        "cell",
        "run_id",
        "epochs_run",
        "best_test_acc",
        "last_train_acc",
        "last_test_acc",
        "grok_epoch_95",
        "grok_epoch_99",
        "plateau_evals",
        "plateau_epoch_span",
        "plateau_detected",
        "metrics_file",
    ]
    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key) for key in csv_fields})

    summary_jsonl = output_dir / "summary.jsonl"
    with summary_jsonl.open("w", encoding="utf-8") as handle:
        for row in summary_rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote sweep summary: {summary_csv}", flush=True)
    print(f"Wrote sweep details: {summary_jsonl}", flush=True)


if __name__ == "__main__":
    main()
