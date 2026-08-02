from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class VisualFeatureExtractor(Protocol):
    feature_dim: int

    def extract(self, mouth_frames: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class StatisticalVisualFeatureExtractor:
    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = feature_dim

    def extract(self, mouth_frames: np.ndarray) -> np.ndarray:
        frames = mouth_frames.astype("float32") / 255.0
        flat = frames.reshape(frames.shape[0], -1)
        mean = flat.mean(axis=1, keepdims=True)
        std = flat.std(axis=1, keepdims=True)
        motion = np.zeros_like(mean)
        if len(flat) > 1:
            motion[1:] = np.abs(flat[1:] - flat[:-1]).mean(axis=1, keepdims=True)
        base = np.concatenate([mean, std, motion], axis=1)
        repeats = int(np.ceil(self.feature_dim / base.shape[1]))
        return np.tile(base, (1, repeats))[:, : self.feature_dim]


class FrozenAutoAVSRFeatureExtractor:
    """Adapter for a frozen Auto-AVSR/AV-HuBERT visual encoder checkpoint.

    The adapter accepts checkpoints that expose an `encoder`, `visual_encoder`,
    or `avhubert` module. It is intentionally strict: if the checkpoint does not
    expose one of those modules, training should fail rather than silently train
    on the wrong features.
    """

    def __init__(self, checkpoint_path: Path, feature_dim: int, device: str = "cuda:0") -> None:
        self.feature_dim = feature_dim
        self.checkpoint_path = checkpoint_path
        self.device_name = device
        import torch

        self.torch = torch
        payload = torch.load(checkpoint_path, map_location=device)
        model = payload.get("model") if isinstance(payload, dict) else payload
        if model is None:
            raise ValueError(f"checkpoint does not contain a model object: {checkpoint_path}")
        encoder = None
        for name in ("visual_encoder", "encoder", "avhubert"):
            candidate = getattr(model, name, None)
            if candidate is not None:
                encoder = candidate
                break
        if encoder is None:
            raise ValueError("checkpoint does not expose visual_encoder, encoder, or avhubert module")
        self.encoder = encoder.to(device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def extract(self, mouth_frames: np.ndarray) -> np.ndarray:
        frames = mouth_frames.astype("float32") / 255.0
        tensor = self.torch.tensor(frames, dtype=self.torch.float32, device=self.device_name).unsqueeze(0).unsqueeze(2)
        with self.torch.no_grad():
            encoded = self.encoder(tensor)
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        features = encoded.squeeze(0).detach().cpu().numpy()
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"encoder feature dim {features.shape[-1]} != expected {self.feature_dim}")
        return features
