#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from command.labels import STARTER_VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a starter Silent Vision command dataset manifest.")
    parser.add_argument("--output", type=Path, default=Path("data/commands/manifest.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for intent, variants in STARTER_VARIANTS.items():
            for variant in variants:
                record = {
                    "intent": intent.value,
                    "variant": variant,
                    "original_video": "",
                    "mouth_roi_video": "",
                    "metadata": {},
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
