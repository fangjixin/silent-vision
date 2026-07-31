#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from configparser import ConfigParser
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mpc001 VSR on pre-cropped Silent Vision mouth frames.")
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--frames-npy", required=True, type=Path)
    parser.add_argument("--gpu-idx", default=0, type=int)
    return parser.parse_args()


def _resolve(repo_dir: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else repo_dir / path)


def main() -> None:
    args = _parse_args()
    repo_dir = args.repo_dir.resolve()
    config_path = args.config.resolve()
    frames_path = args.frames_npy.resolve()
    if not repo_dir.exists():
        raise SystemExit(f"repo dir does not exist: {repo_dir}")
    if not config_path.exists():
        raise SystemExit(f"config does not exist: {config_path}")
    if not frames_path.exists():
        raise SystemExit(f"frames npy does not exist: {frames_path}")

    sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)

    import torch
    from pipelines.data.transforms import VideoTransform
    from pipelines.model import AVSR

    config = ConfigParser()
    config.read(config_path)
    modality = config.get("input", "modality")
    if modality != "video":
        raise SystemExit(f"Silent Vision mouth runner only supports video modality, got {modality!r}")

    input_v_fps = config.getfloat("input", "v_fps")
    model_v_fps = config.getfloat("model", "v_fps")
    device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() and args.gpu_idx >= 0 else "cpu")

    model = AVSR(
        modality,
        _resolve(repo_dir, config.get("model", "model_path")),
        _resolve(repo_dir, config.get("model", "model_conf")),
        _resolve(repo_dir, config.get("model", "rnnlm")),
        _resolve(repo_dir, config.get("model", "rnnlm_conf")),
        config.getfloat("decode", "penalty"),
        config.getfloat("decode", "ctc_weight"),
        config.getfloat("decode", "lm_weight"),
        config.getint("decode", "beam_size"),
        device,
    )
    frames = np.load(frames_path)
    if frames.ndim != 3:
        raise SystemExit(f"expected frames shape [T,H,W], got {frames.shape}")
    sample = VideoTransform(speed_rate=input_v_fps / model_v_fps)(torch.tensor(frames.astype("uint8", copy=False)))
    print(f"hyp: {model.infer(sample)}")


if __name__ == "__main__":
    main()
