import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.experiments import run_hypothesis_b_transition_pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Hypothesis B transition pilot with exhaustive subsets."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "surgery_transition_pilot.yaml"),
        help="Path to transition pilot YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    results = run_hypothesis_b_transition_pilot(cfg)

    print(f"Exhaustive rows: {results['num_rows']}")
    print(f"Checkpoints: {results['num_checkpoints']}")
    print(f"Strong H2 passes: {results['num_strong_h2_pass']}")
    print(f"Metrics: {results['exhaustive_path']}")
    print(f"Summary: {results['summary_path']}")
    print(f"Heatmap: {results['heatmap_path']}")
    print(f"Scatter: {results['scatter_path']}")
    print(f"Report: {results['report_path']}")


if __name__ == "__main__":
    main()
