"""
inference.py
============
Standalone inference script for the TrOCR handwriting model.

Usage
-----
    python inference.py --image path/to/word.png
    python inference.py --image path/to/word.png --checkpoint latest
    python inference.py --image path/to/word.png --checkpoint checkpoints/best
"""

import sys
import argparse
from train_trocr import predict, CONFIG

def main():
    parser = argparse.ArgumentParser(
        description="Run TrOCR inference on a handwritten word image"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input handwritten word image (PNG/JPG)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        choices=["best", "latest"],
        help="Which checkpoint to use: 'best' (lowest val CER) or 'latest'",
    )
    args = parser.parse_args()

    try:
        text = predict(args.image, checkpoint=args.checkpoint)
        print(f"\n{'='*50}")
        print(f"  Image      : {args.image}")
        print(f"  Checkpoint : {args.checkpoint}")
        print(f"  Prediction : {text}")
        print(f"{'='*50}\n")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
