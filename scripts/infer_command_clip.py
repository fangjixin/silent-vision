#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.config import Settings
from command.inference import TorchCommandClassifierBackend
from video.clip import decode_video_clip
from video.mouth_roi import extract_mouth_roi_clip
from vision.face import create_face_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run closed-set command inference for one uploaded video clip.")
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings(command_backend="torch", command_classifier_checkpoint=args.checkpoint)
    decoded = decode_video_clip(args.clip.read_bytes(), settings.command_clip_fps)
    roi = extract_mouth_roi_clip(frames=decoded.frames, face_detector=create_face_detector(settings), settings=settings)
    decision = TorchCommandClassifierBackend(settings).predict(roi.mouth_frames, {"clip": str(args.clip)})
    print(json.dumps(decision.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
