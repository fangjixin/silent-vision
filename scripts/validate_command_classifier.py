#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen fixed-phrase thresholds on final partitions"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--known-manifest", type=Path, required=True)
    parser.add_argument("--unknown-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probability-override", type=float, default=None)
    parser.add_argument("--distance-override", type=float, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from command.evaluation import evaluate_checkpoint

    report = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        known_manifest=args.known_manifest,
        unknown_manifest=args.unknown_manifest,
        output_path=args.output,
        probability_override=args.probability_override,
        distance_override=args.distance_override,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
