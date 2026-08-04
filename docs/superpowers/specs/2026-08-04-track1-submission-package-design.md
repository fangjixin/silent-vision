# Silent Vision Track 1 Submission Package Design

**Date:** 2026-08-04  
**Entrant:** Jixin Fang  
**Application:** Silent Vision  
**Pull request title:** `Track 1, Jixin Fang, Silent Vision`

## 1. Objective

Prepare an English submission package for Track 1 of the AMD AI DevMaster
Hackathon. The package must explain what the current project does, how it runs,
where AMD Radeon and ROCm are used, and how a judge can reproduce the workflow.
It must not claim performance or capabilities that have not been measured.

The submission positions Silent Vision as a visual intent interface for creator
workflows and other environments where audio is unavailable, unreliable, private,
or undesirable. It is not presented as open-vocabulary speech transcription.

## 2. Product Positioning

### Core statement

Silent Vision turns a short camera clip of a spoken command into a bounded,
auditable intent. The current proof of concept controls studio-adjacent actions
such as lighting and access. The product direction expands the same interface to
recording, capture, and editing workflows.

### Target users

- Deaf and hard-of-hearing people using public or professional services.
- Creators who need hands-free control in a noisy studio or on a set.
- Operators in industrial spaces where microphones are unreliable.
- Privacy-sensitive teams that do not want an always-listening microphone.
- Security analysts who need communication and behavior signals from silent video,
  without claiming exact transcription when the evidence is insufficient.

### Boundaries

- The current runtime recognizes a closed set of intents; it does not transcribe
  arbitrary speech.
- `UNKNOWN` and low-confidence results are rejected and never executed.
- The current executable proof-of-concept intents are `LIGHT_ON`, `LIGHT_OFF`,
  and `OPEN_DOOR`.
- Creator-specific actions are a product direction unless they are implemented and
  verified before the final submission.
- No accuracy, throughput, latency, or memory number appears without a saved run
  that produced it.

## 3. Runtime Design

### Production and demonstration path

The official demonstration uses the Torch classifier backend on AMD Radeon and
ROCm:

1. The browser records a 2-5 second WebM clip.
2. FastAPI receives the clip over WebSocket.
3. PyAV decodes and resamples the clip to 25 FPS.
4. MediaPipe detects one face and supplies mouth landmarks.
5. The backend extracts a stable 96 x 96 grayscale mouth ROI sequence.
6. A temporal Conformer classifier runs with PyTorch on `cuda:0`, which is the
   standard PyTorch device namespace for a ROCm-backed Radeon GPU.
7. Confidence and top-1 margin thresholds either accept a bounded intent or
   return `UNKNOWN`.
8. Only accepted executable intents reach the agent boundary.

The production startup must fail if ROCm is unavailable, the GPU is not visible,
or the classifier checkpoint is missing. There is no silent CPU fallback.

### Calibration path

The NumPy prototype matcher remains available for collecting and checking small
sets of examples. It is described as a calibration and development tool, not as
the production inference engine and not as evidence of GPU acceleration.

### GPU evidence

The final demonstration and documentation show:

- Python and PyTorch versions.
- `torch.version.hip`.
- `torch.cuda.is_available()` and the selected device.
- Radeon device name, when available from PyTorch.
- `COMMAND_BACKEND=torch`.
- The loaded classifier checkpoint path.
- Per-command inference latency reported by the application.

## 4. Submission Artifacts

### Root README

Rewrite the existing README as a reproducible English guide with these sections:

1. Project overview and product boundary.
2. Key use cases and current intent set.
3. End-to-end architecture.
4. AMD Radeon and ROCm execution path.
5. System requirements and dependency list.
6. Radeon setup and GPU-only startup.
7. Dataset preparation, classifier training, and validation.
8. Browser usage and calibration workflow.
9. Local fake mode for tests only.
10. Docker notes, persistence layout, tests, privacy, and known limitations.
11. Submission artifact index.

The README must distinguish tested behavior from planned work.

### Project Profile PDF

Create a polished six-page document:

1. Cover and one-paragraph project summary.
2. Background, user problem, target users, and concrete scenarios.
3. Product workflow, safety boundary, and practical value.
4. System architecture and data flow.
5. Model and algorithm description, including prototype versus Torch paths.
6. AMD Radeon/ROCm adaptation, reproducibility, current limitations, and next
   validation steps.

The PDF uses diagrams and short paragraphs rather than dense marketing copy.

### Poster

Create an A3 portrait poster in PDF and PNG formats. The poster contains:

- Product name and a plain headline: `Silent control when audio is not an option.`
- A five-step visual workflow from camera clip to bounded action.
- Three concrete scenarios: accessible service, creator studio, and noisy worksite.
- The current safety rule: low-confidence commands do not execute.
- A compact AMD Radeon/ROCm section that states what runs on the GPU.
- A repository QR code linking to `https://github.com/fangjixin/silent-vision`.

The visual language is dark charcoal, off-white, and Radeon red. The layout uses
large type, wide spacing, and one technical diagram. It avoids stock imagery and
decorative AI motifs such as glowing brains, robots, neon circuit boards, and
abstract network spheres.

### Demo recording guide

Create an English 3-5 minute script and shot list now; record the video later on
the Radeon environment. The sequence is:

1. Problem and use case.
2. ROCm and GPU verification in the terminal.
3. GPU-only server startup and checkpoint load.
4. Successful commands from the browser.
5. One rejected or unknown command.
6. Runtime latency and result inspection.
7. Closing summary and repository link.

The guide requires an uninterrupted capture of the actual command line, browser,
and result. It does not permit simulated GPU output or edited-in benchmark values.

### Submission index and PR text

Create a short English submission index that links the profile, poster, source,
README, and later demo video. Prepare an English PR description that maps each
official Track 1 requirement to one repository path.

## 5. Writing Standard

All public materials use plain English and short sentences. Copy must sound like
an engineer describing a working product to another engineer.

Avoid:

- `revolutionary`, `game-changing`, `cutting-edge`, `next-generation`, and
  `seamless` unless quoting an external source.
- `harness the power of AI`, `unlock possibilities`, and similar generic phrases.
- Unsupported superlatives, market-size claims, and invented benchmarks.
- Repeating `AI-powered` when the specific model or operation can be named.
- Long strings of em dashes, slogan-heavy prose, or overly symmetrical bullet
  lists.

Prefer:

- Concrete inputs, outputs, decisions, and failure behavior.
- Direct statements about what runs on CPU and what runs on GPU.
- Honest limitations and reproducible commands.
- Product examples tied to the current code.

## 6. Repository Layout

The project repository will contain:

```text
submission/
  README.md
  Silent-Vision-Project-Profile.pdf
  Silent-Vision-Poster.pdf
  Silent-Vision-Poster.png
  demo-video-script.md
  pull-request-description.md
docs/submission/
  project-profile-source.md
  poster-copy.md
scripts/
  generate_submission_assets.py
README.md
```

Scratch renders remain under `tmp/pdfs/` and are not submitted. Final PDFs are
also copied to `output/pdf/` for PDF workflow compliance; the stable submission
copies remain under `submission/` for the contest bundle.

## 7. Contest Fork and Pull Request

After the local package is verified:

1. Fork `AMD-DEV-CONTEST/Radeon-hackathon-2026-07` under the authenticated GitHub
   account.
2. Add the complete project under `submissions/track1-silent-vision/`, following
   the established contest-repository convention.
3. Keep generated caches, local datasets, prototype recordings, and large model
   checkpoints out of Git.
4. Open a pull request titled `Track 1, Jixin Fang, Silent Vision`.
5. Use the prepared English description and verify every link from the PR diff.

Because the local machine currently has no `gh` command, the final fork and PR
step uses an authenticated GitHub browser session, a GitHub token supplied through
the environment, or an installed GitHub CLI. No credential is written to the
repository.

## 8. Verification

Before delivery:

- Run the existing Python test suite.
- Run deployment and smoke checks appropriate to the available environment.
- Confirm that the README commands match the scripts.
- Confirm that public text contains no placeholders or unverified performance
  claims.
- Generate both PDFs, render every page to PNG, and inspect every rendered page.
- Use PDF text extraction to confirm headings and required topics are present.
- Confirm poster PDF and PNG dimensions and legibility.
- Confirm submission links use relative paths and resolve locally.
- Run a secret and large-file check before copying into the contest fork.

The demo video is not marked complete until the Radeon recording exists and has
been reviewed from start to finish.
