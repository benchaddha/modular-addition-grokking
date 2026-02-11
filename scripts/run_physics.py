import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.experiments import run_paired_physics_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired SGLD physics sweep.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "physics.yaml"),
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    results = run_paired_physics_sweep(cfg)

    print(f"Runs recorded: {results['num_runs']}")
    print(f"Raw runs: {results['raw_path']}")
    print(f"Summary: {results['summary_path']}")
    for figure_path in results["figure_paths"]:
        print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
