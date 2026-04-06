import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.dataset import get_dataset
from src.metrics import find_grok_epoch
from src.model import get_model
from src.dataset import build_batch_schedule
from src.train_physics import clone_state_dict, train_physics_checkpointed_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run paired SGLD training with checkpoint capture for selected seeds/temperatures.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=str(PROJECT_ROOT / 'configs' / 'physics.yaml'),
        help='Base config YAML path.',
    )
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--temperatures', type=float, nargs='+', required=True)
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(PROJECT_ROOT / 'results' / 'physics_checkpoint_sweep'),
        help='Directory for checkpointed physics artifacts.',
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=None,
        help='Override physics.max_epochs.',
    )
    parser.add_argument(
        '--eval-every',
        type=int,
        default=None,
        help='Override physics.eval_every.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Rerun even if seed/temperature summary already exists.',
    )
    return parser.parse_args()


def _temperature_tag(value: float) -> str:
    if value == 0.0:
        return 't0'
    text = f'{value:.0e}'.replace('+', '')
    return f't{text}'


def _summarize_history(history: List[Dict[str, float]]) -> Dict[str, float | int | None]:
    return {
        'best_test_acc': max((float(row['test_acc']) for row in history), default=None),
        'last_test_acc': float(history[-1]['test_acc']) if history else None,
        'grok_epoch_95': find_grok_epoch(history, threshold=0.95),
        'grok_epoch_99': find_grok_epoch(history, threshold=0.99),
        'num_eval_points': len(history),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    base_cfg = Config.from_yaml(args.config)
    if args.max_epochs is not None:
        base_cfg.physics.max_epochs = args.max_epochs
    if args.eval_every is not None:
        base_cfg.physics.eval_every = args.eval_every

    print(f'Seeds: {args.seeds}')
    print(f'Temperatures: {args.temperatures}')
    print(f'Config: {args.config}')
    print(f'Output: {output_dir}')

    for seed in args.seeds:
        split_generator = torch.Generator().manual_seed(seed)
        cfg = Config.from_yaml(args.config)
        if args.max_epochs is not None:
            cfg.physics.max_epochs = args.max_epochs
        if args.eval_every is not None:
            cfg.physics.eval_every = args.eval_every
        cfg.train.seed = seed
        dataset = get_dataset(cfg, generator=split_generator)

        train_size = dataset.train_indices.shape[0]
        schedule_generator = torch.Generator().manual_seed(seed)
        batch_schedule = build_batch_schedule(
            train_size=train_size,
            batch_size=cfg.train.batch_size,
            max_epochs=cfg.physics.max_epochs,
            generator=schedule_generator,
        )

        init_model = get_model(cfg, seed=seed)
        initial_state = clone_state_dict(init_model.state_dict())
        del init_model

        for temperature in args.temperatures:
            temp_tag = _temperature_tag(float(temperature))
            run_id = f'physics_seed{seed}_{temp_tag}'
            run_dir = output_dir / f'seed_{seed}' / temp_tag
            artifacts_dir = run_dir / 'artifacts'
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = artifacts_dir / f'{run_id}.jsonl'
            summary_path = run_dir / 'summary.json'

            if summary_path.exists() and not args.force:
                existing = json.loads(summary_path.read_text(encoding='utf-8'))
                summary_rows.append(existing)
                print(f'[skip] seed={seed} temperature={temperature:g} already completed')
                continue

            noise_generator = torch.Generator().manual_seed(
                seed + cfg.physics.noise_seed_offset
            )
            history_run = train_physics_checkpointed_run(
                cfg=cfg,
                dataset=dataset,
                initial_state_dict=initial_state,
                batch_schedule=batch_schedule,
                noise_generator=noise_generator,
                temperature=float(temperature),
                seed=seed,
                max_epochs=cfg.physics.max_epochs,
                eval_every=cfg.physics.eval_every,
                run_id=run_id,
                checkpoints_dir=artifacts_dir,
                metrics_path=metrics_path,
                stop_on_all_thresholds=False,
            )
            hist_summary = _summarize_history(history_run['history'])
            checkpoint_files = sorted(str(path) for path in artifacts_dir.glob('*.pt'))
            row: Dict[str, object] = {
                'seed': seed,
                'temperature': float(temperature),
                'temperature_tag': temp_tag,
                'best_test_acc': hist_summary['best_test_acc'],
                'last_test_acc': hist_summary['last_test_acc'],
                'grok_epoch_95': hist_summary['grok_epoch_95'],
                'grok_epoch_99': hist_summary['grok_epoch_99'],
                'num_eval_points': hist_summary['num_eval_points'],
                'batch_schedule_hash': history_run['batch_schedule_hash'],
                'artifacts_dir': str(artifacts_dir),
                'metrics_file': str(metrics_path),
                'checkpoint_files': checkpoint_files,
            }
            summary_path.write_text(json.dumps(row, indent=2), encoding='utf-8')
            summary_rows.append(row)
            print(
                f"[done] seed={seed} temperature={temperature:g} "
                f"best_test_acc={float(hist_summary['best_test_acc']):.4f} "
                f"grok99={hist_summary['grok_epoch_99']}"
            )

    summary_csv = output_dir / 'summary.csv'
    fieldnames = [
        'seed',
        'temperature',
        'temperature_tag',
        'best_test_acc',
        'last_test_acc',
        'grok_epoch_95',
        'grok_epoch_99',
        'num_eval_points',
        'batch_schedule_hash',
        'artifacts_dir',
        'metrics_file',
    ]
    with summary_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    summary_jsonl = output_dir / 'summary.jsonl'
    with summary_jsonl.open('w', encoding='utf-8') as handle:
        for row in summary_rows:
            handle.write(json.dumps(row) + '\n')

    print(f'Wrote sweep summary: {summary_csv}')
    print(f'Wrote sweep details: {summary_jsonl}')


if __name__ == '__main__':
    main()
