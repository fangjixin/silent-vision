# Silent Vision Project Profile — Source Copy

Applicant: Jixin Fang

Track: 1

Repository: <https://github.com/fangjixin/silent-vision>

## 1. Project Summary

Silent Vision classifies a short, silent camera clip as one command from a
bounded vocabulary. The browser records a 2-5 second WebM clip without audio.
The backend extracts a mouth-region sequence, applies confidence and margin
checks, and returns a structured command decision plus an agent result.

The current proof of concept recognizes `LIGHT_ON`, `LIGHT_OFF`, `OPEN_DOOR`,
`CHAT_OTHER`, and `UNKNOWN`. It does not transcribe open-ended speech. It also
does not control a physical light or door and does not create a browser recording
or still-image artifact. Those downstream actions require separate, explicit
integrations.

The intended hackathon architecture uses CPU preprocessing and a PyTorch temporal
classifier on AMD Radeon through ROCm. The final Radeon checkpoint, validation
evidence, and recorded demonstration are pending.

## 2. Problem, Users, and Scenarios

Audio is not always a usable control channel. A Deaf or hard-of-hearing person
may need a visual alternative at a service desk. A creator may need hands-free
control near a loud set. An operator may work around machinery where speech
recognition is unreliable. A privacy-sensitive team may prefer a deliberate
camera gesture over an always-listening microphone.

Silent Vision narrows the problem to auditable intents. It answers “which allowed
command best fits this clip?” rather than claiming a transcript. This makes
uncertainty visible and lets an integrator decide exactly which labels may reach
a downstream system.

The current smart-space labels demonstrate the interface boundary. A planned
creator-control extension would use accepted visual commands to start and stop a
browser recording or capture a still. That extension is not implemented in the
current source.

## 3. Product Workflow and Safety Boundary

The current browser workflow is one shot:

1. The user clicks `Start` and grants camera access.
2. A countdown gives the user time to prepare.
3. The browser records one command clip for up to five seconds with audio off.
4. The server returns candidates, confidence, margin, and an acceptance reason.
5. The agent boundary returns `execute`, `ignore`, or `reject` with the intent in
   structured arguments.

Low confidence, a small top-1/top-2 margin, or an `UNKNOWN` prediction produces a
rejection. `CHAT_OTHER` may be accepted as a recognized but non-executable
intent. The current agent marks three smart-space intents executable, but there
is no device adapter in this repository. An `execute` result is data, not proof
that an external action happened.

Prototype calibration stores examples in a shared global profile. Operators
should collect 5-10 correctly labeled samples per intent with natural changes in
pace and head position. Examples can retain useful multilingual phrases such as
`你好，请帮我打开灯` and `hello, please turn on the light`.

## 4. System Architecture and Data Flow

The frontend uses `getUserMedia` with `audio: false` and `MediaRecorder` to make a
WebM clip. FastAPI creates a short-lived session and accepts binary clip data over
a WebSocket. PyAV decodes and resamples frames to 25 FPS. MediaPipe requires one
face, locates mouth landmarks, smooths the crop, and produces a 96 x 96 grayscale
temporal sequence.

Three interchangeable backends consume that sequence. Fake mode supports local
tests. Prototype mode creates a NumPy embedding and compares it with saved global
examples. Torch mode prepares simple per-frame visual features and sends them to
a temporal classifier. All backends produce the same `CommandDecision` schema,
so threshold rejection and the agent boundary stay inspectable.

Calibration samples persist under
`profiles/global/<INTENT>/<sample-id>/`. Each sample includes the original WebM,
`mouth_roi.npy`, `embedding.npy`, and JSON metadata. Optional debug mode writes
aligned-face and mouth-region videos plus command metadata.

## 5. Model and Algorithm

Prototype mode is a calibration tool. It summarizes appearance and motion from
the mouth sequence, normalizes the embedding, compares cosine similarities, and
requires both a minimum confidence and a minimum margin. It can test data
collection quickly, but it is not Radeon classifier evidence.

The Torch classifier has four Conformer-style temporal blocks. Each block uses
feed-forward layers, multi-head self-attention, a depthwise one-dimensional
convolution, and normalization. Attentive pooling reduces the temporal sequence
before the final label classifier. The current feature path uses frame-level
mean, standard deviation, and motion values expanded to the configured feature
dimension.

The repository includes scripts to train, validate, and infer from a checkpoint.
The current manifest helper makes a starter template; trainable rows still need
real `mouth_roi_npy` paths. The current scripts do not create a deterministic
held-out split automatically, so a separate validation manifest must be prepared
for honest evaluation.

## 6. AMD Radeon, Reproduction, and Next Steps

The intended demo keeps PyAV decoding, MediaPipe detection, crop stabilization,
and NumPy feature work on CPU. PyTorch temporal classification runs on a Radeon
GPU through ROCm. In PyTorch, the ROCm-backed device is named `cuda:0`; the HIP
version is checked through `torch.version.hip`.

The real startup scripts refuse to continue when HIP is absent, the accelerator
is unavailable, or a Torch checkpoint path is missing. However, the application
classifier still has a CPU fallback when started directly. The final demo must
therefore export `COMMAND_BACKEND=torch`, provide the checkpoint, run through the
ROCm guard scripts, and record the selected environment.

Next steps are concrete: finish a balanced calibration dataset, build a real
manifest, train on Radeon, evaluate a held-out manifest, save the checkpoint and
reports, implement and test the planned browser creator actions, and record the
3-5 minute end-to-end video. No accuracy or latency number should be published
until those runs exist.
