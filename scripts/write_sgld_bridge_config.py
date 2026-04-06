import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'List clean paired T=0/T=1e-4 seeds from a checkpointed physics sweep and '
            'write a Fourier sufficiency bridge config for a selected seed.'
        )
    )
    parser.add_argument(
        '--summary',
        type=str,
        required=True,
        help='Path to the checkpointed physics sweep summary.csv.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed to materialize into a Fourier sufficiency bridge config.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output YAML path. Required when --seed is provided.',
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only print candidate clean paired seeds.',
    )
    parser.add_argument(
        '--site',
        type=str,
        default='pre_unembed',
        choices=['post_embed', 'pre_unembed'],
        help='Fourier intervention site for the bridge config.',
    )
    parser.add_argument(
        '--baseline-temperature-tag',
        type=str,
        default='t0',
        help='Temperature tag for the baseline condition.',
    )
    parser.add_argument(
        '--sgld-temperature-tag',
        type=str,
        default='t1e-04',
        help='Temperature tag for the SGLD condition.',
    )
    return parser.parse_args()


def _load_rows(summary_path: Path) -> List[Dict[str, str]]:
    with summary_path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _index_rows(rows: List[Dict[str, str]]) -> Dict[Tuple[int, str], Dict[str, str]]:
    indexed: Dict[Tuple[int, str], Dict[str, str]] = {}
    for row in rows:
        indexed[(int(row['seed']), row['temperature_tag'])] = row
    return indexed


def _has_checkpoint(row: Dict[str, str], suffix: str) -> bool:
    artifacts_dir = Path(row['artifacts_dir'])
    metrics_file = Path(row['metrics_file'])
    run_id = metrics_file.stem
    return (artifacts_dir / f'{run_id}_{suffix}.pt').exists()


def _candidate_seeds(
    indexed: Dict[Tuple[int, str], Dict[str, str]],
    baseline_tag: str,
    sgld_tag: str,
) -> List[int]:
    seeds = sorted({seed for seed, _ in indexed})
    candidates: List[int] = []
    for seed in seeds:
        base = indexed.get((seed, baseline_tag))
        sgld = indexed.get((seed, sgld_tag))
        if base is None or sgld is None:
            continue
        if _has_checkpoint(base, 'testacc_99') and _has_checkpoint(sgld, 'testacc_99'):
            candidates.append(seed)
    return candidates


def _checkpoint_paths(row: Dict[str, str]) -> List[str]:
    artifacts_dir = Path(row['artifacts_dir'])
    metrics_file = Path(row['metrics_file'])
    run_id = metrics_file.stem
    suffixes = ['testacc_80', 'testacc_90', 'testacc_95', 'testacc_99']
    paths = []
    for suffix in suffixes:
        path = artifacts_dir / f'{run_id}_{suffix}.pt'
        if not path.exists():
            raise FileNotFoundError(f'Missing expected checkpoint: {path}')
        paths.append(str(path))
    return paths


def _build_config(
    *,
    seed: int,
    baseline_row: Dict[str, str],
    sgld_row: Dict[str, str],
    site: str,
    output_stem: str,
) -> Dict[str, object]:
    checkpoint_paths = _checkpoint_paths(baseline_row) + _checkpoint_paths(sgld_row)
    return {
        'model': {
            'p': 113,
            'd_model': 128,
            'n_layers': 1,
            'n_heads': 4,
            'd_head': 32,
            'd_mlp': 512,
            'n_ctx': 3,
            'act_fn': 'relu',
            'normalization_type': None,
        },
        'data': {'frac_train': 0.3},
        'optim': {'lr': 0.001, 'weight_decay': 1.0},
        'train': {
            'seed': seed,
            'batch_size': 64,
            'epochs': 10000,
            'eval_every': 100,
            'checkpoint_milestones': [0.80, 0.90, 0.95, 0.99],
        },
        'logging': {
            'wandb_project': 'grokking-demo',
            'run_name': f'fourier-sufficiency-sgld-bridge-seed{seed}',
        },
        'surgery': {
            'checkpoint_paths': [checkpoint_paths[0]],
            'probe_split': 'test',
            'probe_max_examples': 0,
            'top_k': [1],
            'random_control_repeats': 0,
            'eval_batch_size': 2048,
            'ranking_metric': 'dla_abs_score',
            'min_baseline_train_acc': 0.80,
            'min_baseline_test_acc': 0.80,
            'causal_train_floor': 0.90,
            'causal_test_chance_multiplier': 2.0,
            'seed': 123,
        },
        'fourier_ablation': {
            'checkpoint_paths': checkpoint_paths,
            'sites': [site],
            'intervention_mode': 'keep_only_selected',
            'sweep_mode': 'multi_frequency',
            'top_k_values': [1, 2, 3, 5, 10, 56],
            'bottom_k_values': [5],
            'random_k_values': [1, 2, 3, 5, 10, 56],
            'random_control_repeats': 3,
            'custom_frequency_sets': [],
            'eval_batch_size': 2048,
            'causal_train_floor': 0.90,
            'causal_test_chance_multiplier': 2.0,
            'seed': 123,
            'output_stem': output_stem,
        },
    }


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(f'Summary not found: {summary_path}')

    rows = _load_rows(summary_path)
    indexed = _index_rows(rows)
    candidates = _candidate_seeds(
        indexed,
        baseline_tag=args.baseline_temperature_tag,
        sgld_tag=args.sgld_temperature_tag,
    )

    print('Clean paired seeds:', candidates)
    if args.list_only or args.seed is None:
        return

    if args.output is None:
        raise ValueError('--output is required when --seed is provided.')
    if args.seed not in candidates:
        raise ValueError(
            f'Seed {args.seed} does not have both paired 99% checkpoints. Candidates: {candidates}'
        )

    baseline_row = indexed[(args.seed, args.baseline_temperature_tag)]
    sgld_row = indexed[(args.seed, args.sgld_temperature_tag)]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.stem
    payload = _build_config(
        seed=args.seed,
        baseline_row=baseline_row,
        sgld_row=sgld_row,
        site=args.site,
        output_stem=output_stem,
    )
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    print(f'Wrote config: {output_path}')


if __name__ == '__main__':
    main()
