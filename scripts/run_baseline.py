import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.metrics import find_grok_epoch
from src.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline grokking experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--grok-threshold",
        type=float,
        default=0.99,
        help="Test-accuracy threshold used to report grok epoch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    history = train(cfg)
    grok_epoch = find_grok_epoch(history, threshold=args.grok_threshold)

    print(f"Logged {len(history)} evaluation points.")
    if grok_epoch is None:
        print(f"No grok epoch found at threshold={args.grok_threshold:.2f}.")
    else:
        print(f"Grok epoch ({args.grok_threshold:.2f}) = {grok_epoch}.")


if __name__ == "__main__":
    main()
