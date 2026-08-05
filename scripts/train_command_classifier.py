#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Silent Vision closed-set command classifier on AMD ROCm.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--visual-encoder-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    parse_args()
    raise SystemExit("the legacy intent trainer was removed; use the fixed-phrase ROCm training command")


if __name__ == "__main__":
    main()
