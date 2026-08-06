# Track 1, Jixin Fang, Silent Vision

## Project

Silent Vision recognizes a personalized catalog of fixed phrases from a 2-5
second silent camera clip. It is a closed-set visual classifier, not
open-vocabulary lipreading. Its current catalog is one four-phrase bilingual
catalog: Chinese `你好，请帮我打开灯` and `你吃饭了吗？`, and English
`Hello, please turn on the light.` and `Have you eaten?`, mapped to `LIGHT_ON`
and `CHAT_OTHER`. The user selects the language before recording; it is not
inferred from visual input. `UNKNOWN` is produced only by heuristic rejection.

PyAV and MediaPipe decode the video, find one face, align it, and create a 96 x
96 grayscale mouth sequence on CPU. The fixed-phrase Torch model trains and runs
on AMD Radeon through ROCm. It uses 16 x 16 appearance and adjacent-frame motion
maps, two depthwise-separable temporal blocks, attentive pooling, a normalized
embedding, and a dynamic phrase head.

The ROCm-only Torch classifier calculates a selected-language softmax among the
enabled phrases for the user-selected language. An accepted prediction must pass
both the checkpoint probability threshold and the predicted phrase's
centroid-distance threshold. Exact text and intent then come from the checkpoint
catalog. Rejected clips return `UNKNOWN` with no matched phrase text and no
executable action. Top-1 margin is diagnostic only.

The repository returns structured decisions. It does not control a physical
device or content-creation application. The command boundary can be integrated
with a creator workflow, but that integration is not claimed here.

## Submission materials

- [x] English source, dependency list, setup, training, evaluation, and startup
  guide: `submissions/track1-silent-vision/README.md`
- [x] Versioned phrase catalog:
  `submissions/track1-silent-vision/command/phrase_catalog.json`
- [x] Project profile source and generated PDF:
  `submissions/track1-silent-vision/docs/submission/project-profile-source.md`
  and
  `submissions/track1-silent-vision/submission/Silent-Vision-Project-Profile.pdf`
- [x] Poster source, PDF, and PNG:
  `submissions/track1-silent-vision/docs/submission/poster-copy.md`,
  `submissions/track1-silent-vision/submission/Silent-Vision-Poster.pdf`, and
  `submissions/track1-silent-vision/submission/Silent-Vision-Poster.png`
- [x] Demo script and evidence checklist:
  `submissions/track1-silent-vision/submission/demo-video-script.md`
- [ ] Demo video URL: add after the recorded Radeon run
- [x] Source repository: <https://github.com/fangjixin/silent-vision>

## Evidence status

English recordings, bilingual training, the official Radeon run, untouched final
evaluation, and recorded demo remain pending. Official evidence requires 15
independent takes for each of the four phrases plus at least 15 unrelated clips
spanning both selected languages. A small-data run may be used only to prove that
the pipeline executes. This pull request does not claim bilingual accuracy,
latency, throughput, or memory results before the final report exists.
