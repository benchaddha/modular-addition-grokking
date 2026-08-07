"""Strategy 3 trajectory sweep.

For each checkpoint, computes:
  - Fourier spectrum statistics of the embed/unembed maps (participation ratio,
    top-k score mass, ranked frequency identity);
  - keep-only-top-k (sufficiency) and ablate-top-k (necessity) intervention
    curves over a grid of k, with train/test accuracy, cross-entropy, mean
    prediction confidence, and mean logit norm;
  - matched random-k controls for both intervention modes.

Writes one JSONL row per (checkpoint, intervention set) plus one spectrum row
per checkpoint. Interference indices are derived downstream in
scripts/analyze_strategy3.py from the raw accuracies/CEs recorded here.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.fourier_ablation import (
    _stable_random_seed,
    build_fourier_basis,
    make_freq_ablation_hook,
    score_frequencies,
    _site_hook_name,
)
from src.dataset import get_dataset
from src.model import get_model
from src.surgery import _load_checkpoint_payload, _load_model_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--site", default="pre_unembed", choices=["pre_unembed", "post_embed"])
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--k-values",
        default="1,2,3,4,5,6,7,8,9,10,12,14,16,20,24,32,max",
        help="Comma-separated k grid; 'max' expands to (p-1)//2.",
    )
    parser.add_argument("--random-k", default="3,5,10")
    parser.add_argument("--random-repeats", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=321)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip checkpoints already present in the output file.",
    )
    return parser.parse_args()


@torch.inference_mode()
def evaluate_split(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    hook_name: Optional[str] = None,
    hook_fn=None,
) -> Dict[str, float]:
    model.eval()
    total = int(tokens.shape[0])
    correct = 0
    ce_sum = 0.0
    conf_sum = 0.0
    logit_norm_sum = 0.0
    hooks = [(hook_name, hook_fn)] if hook_name and hook_fn else []
    with model.hooks(fwd_hooks=hooks):
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_tokens = tokens[start:end].to(model.cfg.device)
            batch_labels = labels[start:end].to(model.cfg.device)
            logits = model(batch_tokens)[:, -1, :]
            correct += int((logits.argmax(dim=-1) == batch_labels).sum().item())
            ce_sum += float(
                F.cross_entropy(logits, batch_labels, reduction="sum").item()
            )
            probs = F.softmax(logits, dim=-1)
            conf_sum += float(probs.max(dim=-1).values.sum().item())
            logit_norm_sum += float(logits.norm(dim=-1).sum().item())
    return {
        "acc": correct / total,
        "ce": ce_sum / total,
        "confidence": conf_sum / total,
        "logit_norm": logit_norm_sum / total,
    }


def spectrum_stats(score_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scores = torch.tensor([float(r["combined_score"]) for r in score_rows])
    total = float(scores.sum().item())
    mass = scores / total if total > 0 else scores
    sorted_mass, _ = torch.sort(mass, descending=True)
    pr = float((scores.sum() ** 2 / (scores**2).sum()).item()) if total > 0 else 0.0
    ranked = [int(r["frequency"]) for r in score_rows]  # already rank-sorted
    return {
        "participation_ratio": pr,
        "top1_mass": float(sorted_mass[:1].sum().item()),
        "top3_mass": float(sorted_mass[:3].sum().item()),
        "top5_mass": float(sorted_mass[:5].sum().item()),
        "top10_mass": float(sorted_mass[:10].sum().item()),
        "ranked_frequencies_top10": ranked[:10],
        "num_frequencies": len(ranked),
    }


def frequency_sets(
    ranked: List[int],
    k_values: List[int],
    random_k: List[int],
    random_repeats: int,
    random_seed: int,
    checkpoint_path: str,
) -> List[Dict[str, Any]]:
    sets: List[Dict[str, Any]] = []
    max_k = len(ranked)
    for k in k_values:
        k = min(k, max_k)
        sets.append(
            {
                "label": f"top_{k}",
                "family": "top",
                "k": k,
                "frequencies": ranked[:k],
                "repeat": 0,
            }
        )
    all_freqs = torch.tensor(sorted(ranked), dtype=torch.long)
    for k in random_k:
        if k > max_k:
            continue
        for repeat in range(1, random_repeats + 1):
            generator = torch.Generator()
            generator.manual_seed(
                _stable_random_seed(
                    base_seed=random_seed,
                    checkpoint_path=checkpoint_path,
                    k=k,
                    repeat_index=repeat,
                )
            )
            perm = torch.randperm(int(all_freqs.shape[0]), generator=generator)
            sets.append(
                {
                    "label": f"random_{k}_rep{repeat}",
                    "family": "random",
                    "k": k,
                    "frequencies": sorted(int(f) for f in all_freqs[perm[:k]]),
                    "repeat": repeat,
                }
            )
    # dedupe identical top-k entries created by clamping to max_k
    seen = set()
    unique_sets = []
    for entry in sets:
        key = (entry["family"], entry["k"], entry["repeat"])
        if key in seen:
            continue
        seen.add(key)
        unique_sets.append(entry)
    return unique_sets


def process_checkpoint(
    checkpoint: str,
    site: str,
    args: argparse.Namespace,
    out_handle,
) -> None:
    checkpoint_path = Path(checkpoint)
    payload = _load_checkpoint_payload(checkpoint_path)
    cfg = Config.from_dict(payload["cfg"])
    cfg.validate()

    model = get_model(cfg, seed=cfg.train.seed)
    model.load_state_dict(_load_model_state_dict(checkpoint_path))
    model.eval()

    p = cfg.model.p
    dataset = get_dataset(cfg, data_seed=cfg.train.seed)
    train_tokens, train_labels = dataset.train_data()
    test_tokens, test_labels = dataset.test_data()

    score_rows = score_frequencies(model=model, p=p)
    stats = spectrum_stats(score_rows)
    ranked = [int(r["frequency"]) for r in score_rows]

    base = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_type": payload.get("checkpoint_type"),
        "checkpoint_threshold": payload.get("checkpoint_threshold"),
        "checkpoint_metrics": payload.get("metrics"),
        "p": p,
        "n_layers": cfg.model.n_layers,
        "seed": cfg.train.seed,
        "site": site,
    }

    baseline_train = evaluate_split(
        model, train_tokens, train_labels, args.eval_batch_size
    )
    baseline_test = evaluate_split(model, test_tokens, test_labels, args.eval_batch_size)

    out_handle.write(
        json.dumps(
            {
                **base,
                "row_type": "spectrum",
                **stats,
                "frequency_scores": [
                    {
                        "frequency": int(r["frequency"]),
                        "combined_score": float(r["combined_score"]),
                    }
                    for r in score_rows
                ],
                "baseline_train": baseline_train,
                "baseline_test": baseline_test,
            }
        )
        + "\n"
    )

    max_freq = (p - 1) // 2
    k_values = []
    for token in args.k_values.split(","):
        token = token.strip()
        k_values.append(max_freq if token == "max" else int(token))
    k_values = sorted(set(min(k, max_freq) for k in k_values))
    random_k = [int(t) for t in args.random_k.split(",") if t.strip()]

    sets = frequency_sets(
        ranked=ranked,
        k_values=k_values,
        random_k=random_k,
        random_repeats=args.random_repeats,
        random_seed=args.random_seed,
        checkpoint_path=str(checkpoint_path),
    )

    fourier_basis = build_fourier_basis(p)
    hook_name = _site_hook_name(model=model, site=site)

    for mode in ("keep_only_selected", "ablate_selected"):
        for entry in sets:
            hook_fn = make_freq_ablation_hook(
                freqs_to_ablate=entry["frequencies"],
                fourier_basis=fourier_basis,
                model=model,
                site=site,
                intervention_mode=mode,
            )
            int_train = evaluate_split(
                model,
                train_tokens,
                train_labels,
                args.eval_batch_size,
                hook_name=hook_name,
                hook_fn=hook_fn,
            )
            int_test = evaluate_split(
                model,
                test_tokens,
                test_labels,
                args.eval_batch_size,
                hook_name=hook_name,
                hook_fn=hook_fn,
            )
            out_handle.write(
                json.dumps(
                    {
                        **base,
                        "row_type": "intervention",
                        "intervention_mode": mode,
                        "family": entry["family"],
                        "label": entry["label"],
                        "k": entry["k"],
                        "repeat": entry["repeat"],
                        "frequencies": entry["frequencies"],
                        "train": int_train,
                        "test": int_test,
                        "baseline_train": baseline_train,
                        "baseline_test": baseline_test,
                    }
                )
                + "\n"
            )
    out_handle.flush()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.skip_existing and out_path.exists():
        with open(out_path) as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("row_type") == "intervention":
                    done.add(row["checkpoint_path"])

    mode = "a" if (args.skip_existing and out_path.exists()) else "w"
    with open(out_path, mode) as out_handle:
        for index, checkpoint in enumerate(args.checkpoints, start=1):
            if str(Path(checkpoint)) in done:
                print(f"[{index}/{len(args.checkpoints)}] skip {checkpoint}")
                continue
            print(f"[{index}/{len(args.checkpoints)}] {checkpoint}", flush=True)
            process_checkpoint(checkpoint, args.site, args, out_handle)

    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
