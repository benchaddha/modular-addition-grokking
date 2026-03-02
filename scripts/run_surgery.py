import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.experiments import run_hypothesis_b_surgery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full Hypothesis B surgery orchestration."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "surgery.yaml"),
        help="Path to surgery YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    results = run_hypothesis_b_surgery(cfg)

    print(f"Ablations: {results['ablations_path']}")
    print(f"Summary: {results['summary_path']}")
    print(f"Figure: {results['figure_path']}")
    print(f"Report: {results['report_path']}")
    print(
        "Checkpoint outcomes: "
        f"{results['num_causal_pass']}/{results['num_checkpoints']} causal passes"
    )


if __name__ == "__main__":
    main()
