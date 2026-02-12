import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.fourier_analysis import visualize_fourier_spectrum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FFT heatmap from embedding weights (W_E)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. If omitted, trains a fresh model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "results" / "figures" / "music_heatmap.html"),
        help="Output HTML path for the heatmap.",
    )
    parser.add_argument(
        "--target-test-acc",
        type=float,
        default=0.95,
        help="Only used when training from scratch.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional training epoch cap for scratch training.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Optional eval frequency for scratch training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    output_path = Path(args.output)

    result_path = visualize_fourier_spectrum(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        target_test_acc=args.target_test_acc,
        max_epochs=args.max_epochs,
        eval_every=args.eval_every,
    )
    print(f"Generated: {result_path}")


if __name__ == "__main__":
    main()
