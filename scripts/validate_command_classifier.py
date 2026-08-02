#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from backend.config import Settings
from command.inference import TorchCommandClassifierBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Silent Vision command classifier checkpoint.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--margin", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings(
        command_backend="torch",
        command_classifier_checkpoint=args.checkpoint,
        command_confidence_threshold=args.threshold,
        command_top1_margin=args.margin,
    )
    backend = TorchCommandClassifierBackend(settings)
    total = 0
    correct = 0
    rejected = 0
    confusion: Counter[tuple[str, str]] = Counter()
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        npy_path = item.get("mouth_roi_npy")
        if not npy_path:
            continue
        decision = backend.predict(np.load(npy_path), {"manifest": str(args.manifest)})
        expected = item["intent"]
        predicted = decision.intent
        total += 1
        correct += int(decision.accepted and predicted == expected)
        rejected += int(not decision.accepted)
        confusion[(expected, predicted)] += 1
    print(json.dumps({"total": total, "accuracy": correct / max(1, total), "rejectionRate": rejected / max(1, total), "confusion": {str(k): v for k, v in confusion.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
