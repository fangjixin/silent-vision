# Fixed-Phrase Torch Classifier Design

**Date:** 2026-08-05
**Application:** Silent Vision
**Status:** Proposed for implementation

## 1. Objective

Replace the current intent-class Torch path with a personalized, closed-set
fixed-phrase classifier. Each registered phrase is a model class. A catalog maps
the predicted phrase to its exact display text, language, and business intent.

The result must return the registered sentence exactly when a phrase is accepted.
It returns `UNKNOWN` when class probability or distance from the predicted
phrase's learned embedding center fails its calibrated threshold. This is an
empirical open-set rejection rule, not a guarantee that every unseen phrase will
be rejected. The classifier is not presented as open-vocabulary lipreading or
arbitrary speech transcription.

The official path trains and runs the temporal Torch model on AMD Radeon through
ROCm. Video decoding, face detection, and mouth cropping remain CPU preprocessing;
there is no CPU fallback for the Torch classifier itself.

## 2. Product Boundary

### In scope

- Personalized recognition of a small catalog of fixed Chinese or English phrases.
- Multiple phrases mapping to the same business intent.
- Exact registered text returned from catalog metadata rather than generated text.
- Reject-safe routing: rejected clips never execute, and unrelated-phrase false
  acceptance is measured on a disjoint evaluation set.
- ROCm-only training and production inference.
- Reuse of existing calibration recordings without rewriting the original files.
- Reproducible manifests, checkpoints, validation reports, and runtime evidence.

### Out of scope

- Open-vocabulary visual speech transcription.
- Claiming that CMLR, LRS3, AV-HuBERT, Auto-AVSR, MISP2021, ChatCLR, or
  CAS-VSR-W1k is used by the submitted classifier.
- Using a sequence lipreading model as an automatic fallback.
- Cross-speaker, cross-camera, or broad language generalization claims without
  measurements.
- Treating a successful prediction on a training clip as validation evidence.

## 3. Phrase Catalog

The repository contains one versioned JSON catalog. Each enabled entry has:

```json
{
  "phraseId": "zh_light_on_hello",
  "text": "你好，请帮我打开灯",
  "language": "zh",
  "intent": "LIGHT_ON",
  "enabled": true
}
```

`phraseId` is the stable model label. `text` is the exact string shown to the
user. `intent` remains the safe routing label consumed by the current agent and
action boundary.

The initial catalog contains the two phrases already recorded on Radeon:

| phraseId | text | language | intent |
| --- | --- | --- | --- |
| `zh_light_on_hello` | `你好，请帮我打开灯` | `zh` | `LIGHT_ON` |
| `zh_chat_meal` | `你吃饭了吗？` | `zh` | `CHAT_OTHER` |

Additional phrases are added by extending the catalog and recording examples;
adding a phrase does not require a new enum or inference code branch as long as
its intent is already supported.

Catalog validation fails on duplicate IDs, duplicate normalized phrase text,
unknown intents, blank text, unsupported languages, or an empty enabled set.
Normalization applies Unicode NFKC, trims leading and trailing space, collapses
internal whitespace, and lowercases ASCII letters. It does not remove punctuation.
Normalization is used only to match saved metadata to the catalog. Runtime output
always uses the exact catalog `text`.

`UNKNOWN` is not a phrase class. It is a rejection result.

## 4. Dataset and Provenance

### Immutable source data

Calibration files under the persistent profile directory remain unchanged. The
manifest builder reads `metadata.json` and `mouth_roi.npy`; it never edits, moves,
or deletes a recording.

The global profile is the training source for the contest run. Personal-profile
copies are not scanned at the same time. Samples are additionally deduplicated by
sample identifier and mouth-array SHA-256 so copied recordings cannot leak across
the train, threshold-calibration, and final-evaluation splits.

### Catalog-owned label mapping

The saved phrase text selects a catalog entry. The catalog supplies the training
`phraseId` and final `intent`. The original metadata intent is retained as
`sourceIntent` for audit purposes.

If metadata intent disagrees with the catalog, the manifest report records the
mismatch. This allows the five existing `你吃饭了吗？` recordings to remain
untouched while mapping them to `zh_chat_meal` and `CHAT_OTHER`. Unknown phrase
text is excluded with an explicit error entry rather than guessed into a class.

### Split and minimum checks

The builder creates deterministic `train.jsonl`, `calibration-known.jsonl`,
`evaluation-known.jsonl`, `calibration-unknown.jsonl`,
`evaluation-unknown.jsonl`, and `inventory.json` artifacts from stable sample IDs.
Every enabled phrase must have examples in all three known-phrase partitions. The
inventory reports per-phrase counts, exclusions, duplicates, intent mismatches,
hashes, and split membership.

The current four-versus-five sample inventory is sufficient only to exercise the
pipeline with an explicit small-dataset override. It is not sufficient for a
reliable accuracy claim or an official contest evidence run.

The official evidence gate requires at least 15 independent takes per registered
phrase: at least 10 for training, 2 for threshold calibration, and 3 for final
evaluation. It also requires at least 15 unrelated-phrase clips: at least 5 for
threshold calibration and 10 for final evaluation. Training may expose an
`--allow-small-dataset` smoke-test override, but artifacts produced with that flag
are marked non-evidentiary and cannot be cited as accuracy or rejection results.

Unrelated clips have expected label `UNKNOWN` only in calibration and evaluation
data. They are not added as a learned phrase class.

## 5. Model and Training

### Model input and output

Input is a 25 FPS sequence of 96 x 96 grayscale mouth ROIs. The tensor is moved to
the selected ROCm device before classifier feature extraction. A deterministic
Torch front end normalizes it, downsamples each frame to 16 x 16, computes an
adjacent-frame difference map, and concatenates the 256-value appearance map with
the 256-value motion map. A learned projection maps those 512 visual values to 64
features per frame.

This replaces the current production feature path that repeats only frame mean,
standard deviation, and mean motion across 256 positions. That representation
discards mouth shape and is not adequate evidence for fixed-phrase visual
recognition.

The four-layer Conformer is not used for this small personalized dataset. The
phrase model uses two depthwise-separable temporal convolution blocks over the
64-dimensional sequence, attentive pooling, a normalized embedding, and a linear
phrase head. The complete trainable model must remain below 150,000 parameters;
the trainer records the count and fails the official evidence run if the cap is
exceeded.

The front end and small temporal classifier are one versioned Torch model used
identically by training and runtime inference. Its output size is dynamic and
equals the number of enabled phrase IDs in the checkpoint catalog. Feature
configuration is stored in the checkpoint. Training with one feature path and
inferring with another is forbidden.

Seeded training-only augmentation may apply small brightness, spatial-shift, and
temporal-jitter changes to training clips. Calibration and evaluation clips are
never augmented.

The classifier outputs logits over phrase IDs. It does not output intent logits
or generate text. Intent and display text come from the catalog after acceptance.

### ROCm requirement

Training uses `/opt/venv/bin/python` on the Radeon image and fails before loading
data unless:

- `torch.version.hip` is non-empty;
- `torch.cuda.is_available()` is true; and
- at least one device is visible.

The selected device is `cuda:0`, PyTorch's device namespace for ROCm. The
checkpoint training summary records the Torch version, HIP version, selected
device, random seed, catalog hash, manifest hash, and epoch count. The external
run summary records the completed checkpoint's SHA-256; the checkpoint does not
attempt to contain its own hash.

There is no automatic CPU fallback in the contest training command or production
startup. CPU preprocessing is reported honestly and is not described as GPU work.

### Rejection

After training, normalized embeddings from the training partition produce one
centroid per phrase. An accepted phrase must satisfy both a minimum top-1
probability and a maximum cosine distance from the predicted phrase's centroid.
Otherwise the decision is `UNKNOWN`, carries no phrase text, and cannot execute
an action. Top-1/top-2 margin remains debug output but is not an independent
acceptance gate for the initial two-class catalog.

Probability and per-class distance thresholds are selected using only the known
and unrelated calibration partitions. They are frozen before the final evaluation
partition is run. The evaluation report records known-phrase acceptance,
accepted-phrase accuracy, unrelated-phrase false-accept rate, and rejection rate.
The documentation calls this heuristic rejection unless the measured report is
explicitly cited.

Checkpoint thresholds are the runtime defaults. Explicit configuration overrides
are allowed for experiments, but the shared threshold resolver records the
effective value and whether it came from the checkpoint or an override in every
runtime decision and evaluation report.

## 6. Checkpoint Contract

The `.pt` checkpoint is a versioned dictionary containing:

```text
schemaVersion
modelState
phraseIds
phraseCatalog
featureConfig
modelConfig
decisionThresholds
classCentroids
trainingSummary
```

The ordered `phraseIds` list defines classifier-head indices. Runtime builds the
head from this list and validates it against `phraseCatalog`; it does not use the
hard-coded five-intent label list for logits.

Loading fails on unsupported schema versions, missing catalog entries, duplicate
phrase IDs, unknown intents, model-head shape mismatch, incompatible feature
configuration, missing or invalid centroids, a trainable parameter count above the
design cap, or an empty model state. A legacy intent-only checkpoint is rejected
with a migration message rather than silently misinterpreted.

## 7. Runtime Decision Contract

For an accepted clip, the Torch backend returns the existing intent decision plus
phrase metadata:

```json
{
  "intent": "LIGHT_ON",
  "accepted": true,
  "confidence": 0.93,
  "margin": 0.41,
  "metadata": {
    "phraseId": "zh_light_on_hello",
    "matchedPhrase": "你好，请帮我打开灯",
    "displayText": "你好，请帮我打开灯",
    "language": "zh",
    "openSetDistance": 0.12,
    "thresholdSource": "checkpoint",
    "backend": "torch"
  }
}
```

Top-K debug output is phrase-oriented and includes phrase ID, exact text, mapped
intent, and confidence. The public `intent` field remains compatible with the
existing agent policy. `CHAT_OTHER` may display its registered phrase but never
executes an action. Rejected output has `intent: UNKNOWN`, no matched phrase, and
no executable route.

## 8. Scripts and Artifacts

The implementation provides:

- A catalog validator and persisted phrase catalog.
- A manifest builder that scans global calibration samples, deduplicates them,
  records mismatches, and creates deterministic splits.
- A ROCm-only trainer that emits the versioned phrase checkpoint and JSON run
  summary.
- A validator that emits phrase accuracy, mapped-intent accuracy, accepted
  precision, known-phrase acceptance, unrelated-phrase false-accept rate,
  rejection rate, per-phrase counts, confusion matrix, effective thresholds and
  their source, checkpoint hash, partition hashes, and ROCm device evidence.
- A one-clip inference command that prints the same phrase decision contract used
  by the WebSocket runtime.

Large private recordings and checkpoints remain outside Git. The repository
contains catalog data, scripts, tests, and reproducible commands. Submission text
may cite a checkpoint hash and validation report only after they have been
produced on Radeon.

## 9. Tests

CPU-safe unit tests use tiny synthetic arrays and checkpoints; they do not claim
ROCm execution. They cover:

- catalog validation and phrase normalization;
- deterministic split and duplicate prevention;
- preservation and reporting of source-intent mismatches;
- dynamic classifier-head size;
- checkpoint schema and compatibility failures;
- accepted exact-phrase output and intent mapping;
- probability and centroid-distance rejection with no executable route;
- threshold-calibration and final-evaluation partition separation;
- checkpoint threshold defaults and logged override precedence;
- phrase-oriented top-K output;
- WebSocket output from a synthetic Torch checkpoint; and
- validation-report schema.

Radeon verification separately proves HIP availability, GPU-only training under
the parameter cap, checkpoint loading, known-phrase acceptance, unrelated-phrase
false-accept rate on the untouched evaluation partition, and saved JSON evidence.

## 10. Migration and Documentation

The local repository is the implementation source. Before training, its final
commit is synchronized to the Radeon instance; the older template checkout is not
used as evidence.

Public English documentation is updated to describe a personalized fixed-phrase
visual classifier. It must not claim open transcription, a sequence-model
fallback, CMLR/LRS3 integration, or reliable accuracy from the current nine
samples. Model downloads removed from Radeon are not part of setup or startup
instructions.

Prototype matching remains a data-collection and debugging mode. The official
contest demonstration uses `COMMAND_BACKEND=torch`, a real phrase checkpoint,
ROCm device evidence, and untouched final-evaluation output.

## 11. Completion Criteria

The feature is complete only when all of the following are true:

1. The catalog and manifest builder represent each registered phrase as its own
   class and report the existing incorrect source intent without rewriting data.
2. Training on Radeon creates a versioned `.pt` whose head size equals the phrase
   count.
3. Torch inference returns the registered original sentence and mapped intent.
4. Rejected input returns `UNKNOWN` and cannot execute; known acceptance and
   unrelated false-accept rates are measured on the untouched evaluation split.
5. Training and inference share the same feature implementation and configuration.
6. Automated tests pass locally and ROCm evidence is saved remotely.
7. Submission documents describe only measured behavior and actual dependencies.
