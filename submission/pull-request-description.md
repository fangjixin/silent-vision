# Track 1, Jixin Fang, Silent Vision

## Project

Silent Vision classifies a 2-5 second silent camera clip as one bounded visual
command. It returns a structured command decision and agent result. The current
repository does not control physical lights or doors and does not yet create a
browser recording or still-image artifact.

The intended Track 1 demonstration uses CPU video preprocessing and a PyTorch
temporal classifier on AMD Radeon through ROCm. The final Radeon checkpoint,
held-out validation evidence, Creator Mode artifact flow, demo video, and video
URL must be added before this submission is presented as complete.

## Requirement map

- [x] English source and reproduction guide:
  `submissions/track1-silent-vision/README.md`
- [x] Editable six-section project profile copy:
  `submissions/track1-silent-vision/docs/submission/project-profile-source.md`
- [x] Generated project profile PDF:
  `submissions/track1-silent-vision/submission/Silent-Vision-Project-Profile.pdf`
- [x] Editable poster copy:
  `submissions/track1-silent-vision/docs/submission/poster-copy.md`
- [x] Generated poster PDF and PNG:
  `submissions/track1-silent-vision/submission/Silent-Vision-Poster.pdf` and
  `submissions/track1-silent-vision/submission/Silent-Vision-Poster.png`
- [x] Demo recording script and safety checklist:
  `submissions/track1-silent-vision/submission/demo-video-script.md`
- [ ] Demo video: added after the recorded Radeon run
- [x] Source repository: <https://github.com/fangjixin/silent-vision>

## Current technical path

The browser records WebM with audio disabled. PyAV decodes and resamples the clip
to 25 FPS. MediaPipe detects one face, and CPU preprocessing creates a stable
96 x 96 grayscale mouth sequence. The prototype backend supports calibration.
The intended demo selects `COMMAND_BACKEND=torch` so a Conformer-style temporal
classifier runs through PyTorch on Radeon/ROCm. Confidence and top-1 margin
checks reject uncertain commands.

No measured performance result is included because the final Radeon run has not
yet been recorded.
