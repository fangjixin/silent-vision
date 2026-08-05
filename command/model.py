from __future__ import annotations

from functools import lru_cache
from typing import Any


PARAMETER_CAP = 150_000


@lru_cache(maxsize=1)
def _fixed_phrase_model_types() -> tuple[type, type]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    class TemporalBlock(nn.Module):
        def __init__(self, channels: int, dilation: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels),
                nn.Conv1d(channels, channels, 1),
                nn.GELU(),
                nn.BatchNorm1d(channels),
            )

        def forward(self, x):
            return x + self.net(x)

    class FixedPhraseModel(nn.Module):
        def __init__(self, num_classes: int, embedding_dim: int = 64):
            super().__init__()
            self.projection = nn.Linear(512, 64)
            self.temporal = nn.Sequential(TemporalBlock(64, 1), TemporalBlock(64, 2))
            self.attention = nn.Linear(64, 1)
            self.embedding = nn.Linear(64, embedding_dim)
            self.classifier = nn.Linear(embedding_dim, num_classes)

        def forward(self, frames):
            if frames.ndim != 4 or tuple(frames.shape[-2:]) != (96, 96):
                raise ValueError("frames must have shape [B, T, 96, 96]")
            if frames.shape[0] < 1 or frames.shape[1] < 1:
                raise ValueError("frames must have non-empty B and T dimensions in [B, T, 96, 96]")

            x = frames.float()
            if x.max().item() > 1.0:
                x = x / 255.0
            small = F.interpolate(
                x.reshape(-1, 1, 96, 96), size=(16, 16), mode="bilinear", align_corners=False
            ).reshape(x.shape[0], x.shape[1], 256)
            motion = torch.zeros_like(small)
            motion[:, 1:] = small[:, 1:] - small[:, :-1]
            sequence = self.projection(torch.cat([small, motion], dim=-1))
            sequence = self.temporal(sequence.transpose(1, 2)).transpose(1, 2)
            weights = torch.softmax(self.attention(sequence), dim=1)
            pooled = (weights * sequence).sum(dim=1)
            embedding = F.normalize(self.embedding(pooled), dim=-1)
            return self.classifier(embedding), embedding

    return TemporalBlock, FixedPhraseModel


def build_fixed_phrase_model(num_classes: int, embedding_dim: int = 64) -> Any:
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool) or embedding_dim < 1:
        raise ValueError("embedding_dim must be a positive integer")
    _, fixed_phrase_model = _fixed_phrase_model_types()
    return fixed_phrase_model(num_classes, embedding_dim)


def count_trainable_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
