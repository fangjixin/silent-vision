# Silent Vision

Copy contract: `scripts/generate_submission_assets.py` is the canonical copy and
layout source for the generated poster. This reviewed source must be updated with
the generator whenever technical claims or evidence status change.

## A fixed phrase. A silent clip. An inspectable decision.

Personalized visual command recognition from a 2-5 second camera clip with audio
disabled. Silent Vision is closed-set, not open-vocabulary lipreading.

### Registered phrases

- `你好，请帮我打开灯` -> `LIGHT_ON`
- `你吃饭了吗？` -> `CHAT_OTHER`

The model predicts a stable phrase ID. Exact displayed text and intent come from
the checkpoint catalog. `UNKNOWN` is a rejection result, not a trained class.

### How it works

1. Record one silent WebM clip.
2. Decode, detect, align, and crop the mouth on CPU.
3. Run the fixed-phrase Torch model on AMD Radeon through ROCm.
4. Require both probability and phrase-centroid distance to pass.
5. Return an exact phrase decision or reject it as `UNKNOWN`.

### Practical fit

Creator control input

A deliberate hands-free signal that another application can map to capture,
cue, or editing actions.

Accessible service

A small visual phrase set that supplements other communication channels.

Noisy or private spaces

A command input when a microphone is unreliable or continuous audio capture is
unwanted.

### Safety boundary

Rejected clips carry no matched phrase text and cannot execute. The current
repository returns structured decisions only; it does not control a device or
content-creation application.

### AMD Radeon + ROCm

CPU: video decode, face detection, alignment, and mouth crop.

Radeon: learned temporal model, phrase logits, and normalized embedding through
ROCm PyTorch on `cuda:0`.

ROCm execution and final evaluation remain to be recorded. Small-data smoke proves pipeline execution only.

### Source

github.com/fangjixin/silent-vision

https://github.com/fangjixin/silent-vision
