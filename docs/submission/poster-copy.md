# Silent Vision

Copy contract: `scripts/generate_submission_assets.py` is the canonical copy and
layout source for the generated poster. This file is reviewed reference copy and
must be updated alongside high-risk copy or status changes in the generator.

## Silent control when audio is not an option.

Closed-set visual command recognition from a short camera clip.

### How it works

1. Record a 2-5 second silent WebM clip.
2. Decode at 25 FPS on CPU.
3. Find one face and extract a 96 x 96 mouth sequence.
4. Classify a bounded intent and check confidence plus margin.
5. Return a structured execute, ignore, or reject result.

### Where it fits

Accessible service

A visual alternative when spoken audio is not available.

Creator studio

A planned hands-free input for recording and still capture.

Noisy worksite

A small command vocabulary when microphones are unreliable.

### Safety rule

Low-confidence and ambiguous commands do not execute.

The current repository returns structured decisions. It does not yet control a
device or create a browser media artifact.

### AMD Radeon + ROCm

Intended demo path: CPU video preprocessing, then PyTorch temporal
classification on AMD Radeon through ROCm.

Final Radeon run, trained checkpoint, validation evidence, Creator Mode actions,
and recorded demo are pending.

### Source

github.com/fangjixin/silent-vision

https://github.com/fangjixin/silent-vision
