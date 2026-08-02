#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from command.encoder import FrozenAutoAVSRFeatureExtractor, StatisticalVisualFeatureExtractor
from command.labels import COMMAND_LABELS, CommandIntent
from command.model import CommandConformerClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Silent Vision closed-set command classifier on AMD ROCm.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--visual-encoder-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    import torch
    from torch import nn

    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    extractor = (
        FrozenAutoAVSRFeatureExtractor(args.visual_encoder_checkpoint, args.feature_dim, device=device)
        if args.visual_encoder_checkpoint
        else StatisticalVisualFeatureExtractor(args.feature_dim)
    )
    samples = load_samples(args.manifest, extractor)
    if not samples:
        raise SystemExit("manifest does not contain trainable samples with mouth_roi_npy")
    model = CommandConformerClassifier(
        feature_dim=args.feature_dim,
        num_classes=len(COMMAND_LABELS),
        num_layers=4,
    ).model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        model.train()
        for features, label_id in samples:
            x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
            y = torch.tensor([label_id], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        print(f"epoch={epoch} loss={total_loss / len(samples):.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "labels": [label.value for label in COMMAND_LABELS]}, args.output)
    print(args.output)


def load_samples(manifest: Path, extractor) -> list[tuple[np.ndarray, int]]:
    samples = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        npy_path = item.get("mouth_roi_npy")
        if not npy_path:
            continue
        frames = np.load(npy_path)
        intent = CommandIntent(item["intent"])
        samples.append((extractor.extract(frames), COMMAND_LABELS.index(intent)))
    return samples


if __name__ == "__main__":
    main()
