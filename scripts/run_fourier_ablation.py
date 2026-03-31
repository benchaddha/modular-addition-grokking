import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.experiments import run_fourier_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fourier-mode ablation across configured checkpoints and sites."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "fourier_ablation.yaml"),
        help="Path to Fourier ablation YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    results = run_fourier_ablation(cfg)

    print(f"Scores: {results['scores_path']}")
    print(f"Ablations: {results['ablations_path']}")
    print(f"Summary: {results['summary_path']}")
    print(f"Heatmap: {results['heatmap_path']}")
    print(f"Report: {results['report_path']}")
    print(
        "Checkpoint/site outcomes: "
        f"{results['num_strong_h2_pass']}/{results['num_checkpoint_site_pairs']} "
        "strong H2 passes"
    )


if __name__ == "__main__":
    main()
