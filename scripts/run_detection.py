from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from apple_logo_recognition import run_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect Apple logos in natural images.")
    parser.add_argument("--data", default="data", help="Directory with input images.")
    parser.add_argument("--output", default="results", help="Directory for annotations, masks and reports.")
    parser.add_argument("--min-score", type=float, default=0.56, help="Minimum detector score accepted as logo.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_dataset(args.data, args.output, min_score=args.min_score)
    for result in results:
        print(f"{Path(result.image).name}: {len(result.detections)} detection(s) -> {result.annotated_path}")


if __name__ == "__main__":
    main()
