# Human-Centered Cinematic Poster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the report-like poster with a reproducible A3 cinematic `2 x 2` mosaic that centers four fictional, internationally diverse target users and communicates Silent Vision's honest fixed-phrase product boundary.

**Architecture:** Four repository-owned 3:2 PNG scene assets provide the human-centered visual layer. `scripts/generate_submission_assets.py` crops them into deterministic ReportLab panels, adds approved English copy and the QR code as vector content, then uses Poppler to render the matching PNG preview. Tests separate the Project Profile contract from the poster contract and verify source assets, PDF composition, exact copy, and offline bundle rebuilding.

**Tech Stack:** Built-in ImageGen, Pillow 10.4, ReportLab 4.4, qrcode, Poppler `pdftoppm`, pytest, pypdf, pdfplumber

## Global Constraints

- Do not change the six-page A4 Project Profile PDF or its source copy.
- Produce one portrait A3 poster page at `841.89 x 1190.55` points and a matching PNG of at least `1700 x 2400` pixels.
- Four fictional, internationally diverse photorealistic scenes occupy roughly 70 percent of the page in a `2 x 2` mosaic.
- Present people with dignity and agency; avoid pity, medical sensationalism, and stereotypes.
- Exact scene labels: `POST-SURGERY`, `REHABILITATION`, `ACCESSIBLE COMMUNICATION`, `SILENT CONTROL INPUT`.
- Visible qualifier: `PERSONALIZED FIXED-PHRASE PROTOTYPE`.
- Main title: `A VOICE WITHOUT SOUND.`
- Audience statement: `Visual communication for people who can form words but cannot speak them aloud.`
- Boundary: `Four registered phrases · Chinese + English · Exact phrase or safe UNKNOWN`.
- Footer: `Camera-only · No audio capture · ROCm PyTorch on AMD Radeon`.
- Do not include a concrete command-result example.
- Do not claim open vocabulary, transcription, cross-speaker generalization, measured bilingual accuracy, or direct device operation.
- `SILENT CONTROL INPUT` shows an input/decision for separate integration, never a completed device action.
- Source images live in the repository; rebuilding requires neither ImageGen nor network access.
- All visible submission copy is English.

---

### Task 1: Generate and Validate Four Cinematic Scene Assets

**Files:**
- Create: `submission/assets/poster/post-surgery.png`
- Create: `submission/assets/poster/rehabilitation.png`
- Create: `submission/assets/poster/accessible-communication.png`
- Create: `submission/assets/poster/silent-control-input.png`
- Modify: `tests/test_submission_docs.py`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-08-06-human-centered-cinematic-poster-design.md`.
- Produces: Four RGB/RGBA landscape PNGs, each at least `1400 x 900` and approximately `3:2`. Task 2 reads these exact paths.

- [ ] **Step 1: Write the failing scene-asset test**

```python
POSTER_SCENE_PATHS = (
    ROOT / "submission/assets/poster/post-surgery.png",
    ROOT / "submission/assets/poster/rehabilitation.png",
    ROOT / "submission/assets/poster/accessible-communication.png",
    ROOT / "submission/assets/poster/silent-control-input.png",
)


def test_poster_scene_sources_are_repository_owned_landscape_pngs():
    for path in POSTER_SCENE_PATHS:
        assert path.is_file(), path
        with Image.open(path) as image:
            width, height = image.size
            assert image.format == "PNG"
            assert image.mode in {"RGB", "RGBA"}
            assert width >= 1400
            assert height >= 900
            assert width / height == pytest.approx(1.5, rel=0.12)
```

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest tests/test_submission_docs.py::test_poster_scene_sources_are_repository_owned_landscape_pngs -v
```

Expected: FAIL on the first absent scene path.

- [ ] **Step 3: Generate `post-surgery.png` with built-in ImageGen**

Generate at `1536 x 1024`, inspect at original detail, and copy the accepted PNG into the repository.

```text
Use case: ads-marketing
Asset type: top-left scene in an A3 social-impact technology poster
Primary request: Photorealistic cinematic editorial image of a fictional Black woman in her forties recovering after throat surgery. She sits upright with dignity in a modern hospital room, looking toward a tablet camera while visibly forming a word with her lips.
Scene/backdrop: Calm realistic recovery room, subtle equipment, soft daylight, clinician only as an unobtrusive background figure.
Subject: Mouth, eyes, and full face sharply visible; a small clean neck dressing may suggest recovery, with no wound or distress.
Style: Restrained hopeful feature-film still, natural skin, deep blue-black shadows, warm practical light, subtle AMD-red accent.
Composition: Landscape 3:2 medium close-up, face in outer upper third, clear lower-middle title-safe zone, enough hospital context.
Avoid: Text, logos, watermark, microphone, mask, hidden mouth, gore, crying, pity pose, science-fiction HUD, malformed anatomy.
```

- [ ] **Step 4: Generate `rehabilitation.png`**

```text
Use case: ads-marketing
Asset type: top-right scene in an A3 social-impact technology poster
Primary request: Photorealistic cinematic editorial image of a fictional East Asian man in his thirties with a voice disorder in rehabilitation. He faces a tablet camera and carefully forms a visible word while a fictional Latina speech therapist observes supportively beside him.
Scene/backdrop: Contemporary rehabilitation clinic with believable materials and quiet depth.
Subject: User is protagonist; mouth, eyes, and face sharply visible; therapist is secondary and does not touch his face.
Style: Match the first scene's restrained hopeful film still, deep shadows, warm light, subtle AMD-red accent.
Composition: Landscape 3:2 medium close-up, protagonist in outer upper third, clear title-safe zone.
Avoid: Text, logos, watermark, microphone, mask, hidden mouth, pity pose, exaggerated disability cue, science-fiction HUD, malformed anatomy.
```

- [ ] **Step 5: Generate `accessible-communication.png`**

```text
Use case: ads-marketing
Asset type: bottom-left scene in an A3 social-impact technology poster
Primary request: Photorealistic cinematic editorial image of a fictional Middle Eastern woman with a hearing or speech disability communicating at a public-service counter. She faces an assistive camera display and visibly forms a word; the worker watches the shared display attentively.
Scene/backdrop: Welcoming service desk, neutral signage shapes without readable text, assistive display between both people.
Subject: Customer is confident and active; mouth, eyes, and face sharply visible. A discreet hearing aid is optional.
Style: Match the other film stills, deep blue-black shadows, warm practical light, subtle AMD-red accent.
Composition: Landscape 3:2 medium shot, customer in outer third, clear title-safe zone.
Avoid: Text, logos, watermark, microphone, mask, hidden mouth, pity pose, science-fiction HUD, malformed anatomy.
```

- [ ] **Step 6: Generate `silent-control-input.png`**

```text
Use case: ads-marketing
Asset type: bottom-right scene in an A3 social-impact technology poster
Primary request: Photorealistic cinematic editorial image of a fictional Latino man in his fifties in a noisy technical workspace. He faces a camera terminal and forms a deliberate command; the terminal has abstract red status shapes with no readable text. No device is switching or reacting.
Scene/backdrop: Believable workspace, machinery softly out of focus, safe environment, terminal reads as input rather than autonomous controller.
Subject: Capable professional; mouth, eyes, and face sharply visible; safety clothing allowed but no mask or visor.
Style: Match the other film stills, deep blue-black shadows, warm practical light, subtle AMD-red accent.
Composition: Landscape 3:2 medium close-up, person in outer upper third, clear title-safe zone.
Avoid: Completed device action, illuminated response, readable text, logos, watermark, microphone, hidden mouth, science-fiction HUD, malformed anatomy.
```

- [ ] **Step 7: Inspect all four images and regenerate only defects**

Use `view_image` at original detail. Reject malformed anatomy, hidden lips, invented text, watermark/logo, pity/gore, implausible staging, completed control, or incompatible grading.

- [ ] **Step 8: Verify GREEN and commit**

```bash
.venv/bin/python -m pytest tests/test_submission_docs.py::test_poster_scene_sources_are_repository_owned_landscape_pngs -v
git add tests/test_submission_docs.py submission/assets/poster
git commit -m "feat: add cinematic poster scene artwork"
```

Expected: PASS and one focused artwork commit.

---

### Task 2: Rebuild the Poster Copy, Layout, and Deliverables

**Files:**
- Modify: `docs/submission/poster-copy.md`
- Modify: `scripts/generate_submission_assets.py`
- Modify: `tests/test_submission_docs.py`
- Modify: `submission/Silent-Vision-Poster.pdf`
- Modify: `submission/Silent-Vision-Poster.png`
- Modify: `output/pdf/Silent-Vision-Poster.pdf`

**Interfaces:**
- Consumes: Task 1's four PNGs, existing `_label()` and `_qr_image()`, and the unchanged Project Profile generator.
- Produces: `POSTER_SCENES`, `_poster_image_reader()`, `_draw_poster_scene()`, deterministic `build_poster_pdf()`, and final PDF/PNG deliverables.

- [ ] **Step 1: Write new copy and layout tests**

Replace the combined profile/poster routing test with:

```python
def _pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def test_profile_assets_describe_bilingual_language_routing():
    source = (ROOT / "docs/submission/project-profile-source.md").read_text(encoding="utf-8")
    generated = _pdf_text(ROOT / "submission/Silent-Vision-Project-Profile.pdf")
    for text in (source, generated):
        assert "你好，请帮我打开灯" in text
        assert "你吃饭了吗？" in text
        assert "Hello, please turn on the light." in text
        assert "Have you eaten?" in text
        assert "user selects the language" in text.lower()
        assert "selected-language softmax" in text.lower()
        assert "no bilingual accuracy" in text.lower()


def test_poster_assets_use_the_approved_human_centered_copy():
    source = (ROOT / "docs/submission/poster-copy.md").read_text(encoding="utf-8")
    generated = _pdf_text(ROOT / "submission/Silent-Vision-Poster.pdf")
    required = {
        "SILENT VISION",
        "PERSONALIZED FIXED-PHRASE PROTOTYPE",
        "A VOICE WITHOUT SOUND.",
        "Visual communication for people who can form words but cannot speak them aloud.",
        "POST-SURGERY",
        "REHABILITATION",
        "ACCESSIBLE COMMUNICATION",
        "SILENT CONTROL INPUT",
        "Four registered phrases · Chinese + English · Exact phrase or safe UNKNOWN",
        "Camera-only · No audio capture · ROCm PyTorch on AMD Radeon",
    }
    forbidden = {"Please turn on the light.", "Have you eaten?", "directly controls"}
    for text in (source, generated):
        assert all(phrase in text for phrase in required)
        assert not any(phrase in text for phrase in forbidden)
```

Replace the obsolete three-card overlap test with:

```python
def test_poster_uses_four_large_scene_images_instead_of_report_cards():
    with pdfplumber.open(ROOT / "submission/Silent-Vision-Poster.pdf") as pdf:
        page = pdf.pages[0]
        large_images = [
            image
            for image in page.images
            if image["width"] >= page.width * 0.45
            and image["height"] >= page.height * 0.25
        ]
    assert len(large_images) == 4
```

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest \
  tests/test_submission_docs.py::test_profile_assets_describe_bilingual_language_routing \
  tests/test_submission_docs.py::test_poster_assets_use_the_approved_human_centered_copy \
  tests/test_submission_docs.py::test_poster_uses_four_large_scene_images_instead_of_report_cards \
  -v
```

Expected: profile PASS; both poster tests FAIL against the old poster.

- [ ] **Step 3: Replace `docs/submission/poster-copy.md`**

```markdown
# Silent Vision

Copy contract: `scripts/generate_submission_assets.py` is the canonical layout
source for the generated poster. This reviewed source must be updated with the
generator whenever product boundaries change.

## PERSONALIZED FIXED-PHRASE PROTOTYPE

## A VOICE WITHOUT SOUND.

Visual communication for people who can form words but cannot speak them aloud.

### Target-user scenes

- POST-SURGERY
- REHABILITATION
- ACCESSIBLE COMMUNICATION
- SILENT CONTROL INPUT

The four fictional people represent target-user scenarios, not demonstrated
cross-speaker performance. Silent Vision returns a structured classifier decision;
a separately integrated application may map an accepted decision to an action.
Silent Vision does not directly control a device.

### Product boundary

Four registered phrases · Chinese + English · Exact phrase or safe UNKNOWN

### AMD Radeon + ROCm

Camera-only · No audio capture · ROCm PyTorch on AMD Radeon

The current prototype is closed-set, not open-vocabulary lipreading. No measured
bilingual accuracy or cross-speaker generalization claim is made on this poster.

### Source

https://github.com/fangjixin/silent-vision
```

- [ ] **Step 4: Update the generator's reviewed-copy contract**

Delete only `POSTER_EVIDENCE_STATUS`; the profile evidence constant and profile requirements remain unchanged. Replace the `"poster"` requirement list in `_read_reviewed_copy()` with this exact contract:

```python
[
    REPOSITORY_URL,
    "SILENT VISION",
    "PERSONALIZED FIXED-PHRASE PROTOTYPE",
    "A VOICE WITHOUT SOUND.",
    "Visual communication for people who can form words but cannot speak them aloud.",
    "POST-SURGERY",
    "REHABILITATION",
    "ACCESSIBLE COMMUNICATION",
    "SILENT CONTROL INPUT",
    "Four registered phrases · Chinese + English · Exact phrase or safe UNKNOWN",
    "Camera-only · No audio capture · ROCm PyTorch on AMD Radeon",
    "does not directly control a device",
    "not open-vocabulary lipreading",
]
```

- [ ] **Step 5: Add deterministic image constants and helpers**

```python
from PIL import Image, ImageOps

POSTER_SCENE_DIR = SUBMISSION / "assets/poster"
POSTER_SCENES = (
    ("POST-SURGERY", POSTER_SCENE_DIR / "post-surgery.png", (0.50, 0.42)),
    ("REHABILITATION", POSTER_SCENE_DIR / "rehabilitation.png", (0.50, 0.42)),
    ("ACCESSIBLE COMMUNICATION", POSTER_SCENE_DIR / "accessible-communication.png", (0.50, 0.48)),
    ("SILENT CONTROL INPUT", POSTER_SCENE_DIR / "silent-control-input.png", (0.50, 0.48)),
)


def _poster_image_reader(
    path: Path,
    width: float,
    height: float,
    centering: tuple[float, float],
) -> ImageReader:
    if not path.is_file():
        raise FileNotFoundError(f"Poster scene image not found: {path}")
    target = (max(1, round(width * 2)), max(1, round(height * 2)))
    with Image.open(path) as source:
        fitted = ImageOps.fit(source.convert("RGB"), target, Image.Resampling.LANCZOS, centering=centering)
        buffer = io.BytesIO()
        fitted.save(buffer, format="JPEG", quality=92, optimize=True)
    buffer.seek(0)
    return ImageReader(buffer)


def _draw_poster_scene(
    pdf: canvas.Canvas,
    *,
    label: str,
    path: Path,
    centering: tuple[float, float],
    x: float,
    y: float,
    width: float,
    height: float,
    label_at_top: bool,
) -> None:
    reader = _poster_image_reader(path, width, height, centering)
    pdf.drawImage(reader, x, y, width, height, preserveAspectRatio=False, mask="auto")
    label_y = y + height - 34 if label_at_top else y
    pdf.saveState()
    pdf.setFillAlpha(0.72)
    pdf.setFillColor(CHARCOAL)
    pdf.rect(x, label_y, width, 34, fill=1, stroke=0)
    pdf.restoreState()
    pdf.setFillColor(WHITE)
    pdf.setFont("Helvetica-Bold", 8.5)
    pdf.drawString(x + 14, label_y + 13, label)
```

- [ ] **Step 6: Replace `build_poster_pdf()` with the explicit mosaic layout**

Implement this geometry. The first two scene labels sit at their panels' top edges and the second two at their panels' bottom edges so the central band cannot cover them. Do not call old card, arrow, catalog, or evidence-panel layout code.

```python
def build_poster_pdf(output: Path) -> None:
    """Create the one-page A3 portrait poster."""
    _read_reviewed_copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=A3, invariant=1, pageCompression=1)
    pdf.setTitle("Silent Vision Poster")
    pdf.setAuthor("Jixin Fang")
    pdf.setSubject("Track 1 project poster")
    width, height = A3

    footer_height = 340
    mosaic_height = height - footer_height
    column_width = width / 2
    row_height = mosaic_height / 2
    for index, (label, path, centering) in enumerate(POSTER_SCENES):
        column = index % 2
        top_row = index < 2
        _draw_poster_scene(
            pdf,
            label=label,
            path=path,
            centering=centering,
            x=column * column_width,
            y=footer_height + (row_height if top_row else 0),
            width=column_width,
            height=row_height,
            label_at_top=top_row,
        )

    band_height = 128
    band_y = footer_height + row_height - band_height / 2
    pdf.saveState()
    pdf.setFillAlpha(0.82)
    pdf.setFillColor(CHARCOAL)
    pdf.rect(0, band_y, width, band_height, fill=1, stroke=0)
    pdf.restoreState()
    pdf.setFillColor(WHITE)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(52, band_y + 105, "SILENT VISION")
    pdf.setFillColor(RADEON_RED)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(158, band_y + 105, "PERSONALIZED FIXED-PHRASE PROTOTYPE")
    pdf.setFillColor(WHITE)
    pdf.setFont("Helvetica-Bold", 38)
    pdf.drawString(52, band_y + 55, "A VOICE WITHOUT SOUND.")
    _paragraph(
        pdf,
        "Visual communication for people who can form words but cannot speak them aloud.",
        52,
        band_y + 27,
        width - 104,
        size=11.5,
        leading=14,
        color=WHITE,
    )

    pdf.setFillColor(OFF_WHITE)
    pdf.rect(0, 0, width, footer_height, fill=1, stroke=0)
    pdf.setFillColor(RADEON_RED)
    pdf.rect(0, footer_height - 8, width, 8, fill=1, stroke=0)
    _paragraph(
        pdf,
        "Four registered phrases · Chinese + English · Exact phrase or safe UNKNOWN",
        52,
        245,
        width - 104,
        font="Helvetica-Bold",
        size=16,
        leading=20,
    )
    _paragraph(
        pdf,
        "Camera-only · No audio capture · ROCm PyTorch on AMD Radeon",
        52,
        169,
        width - 104,
        font="Helvetica-Bold",
        size=14,
        leading=18,
        color=RADEON_RED,
    )
    _paragraph(
        pdf,
        "Target-user scenes · Structured decisions for separate integration · No direct device control",
        52,
        137,
        width - 52 - 112 - 48,
        size=9.5,
        leading=13,
        color=MUTED,
    )

    qr_size = 112
    qr_x = width - 52 - qr_size
    pdf.drawImage(
        _qr_image(),
        qr_x,
        48,
        qr_size,
        qr_size,
        preserveAspectRatio=True,
        mask="auto",
    )
    pdf.setFillColor(CHARCOAL)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(52, 72, "github.com/fangjixin/silent-vision")
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(52, 48, "Track 1 · Jixin Fang")
    pdf.showPage()
    pdf.save()
```

- [ ] **Step 7: Regenerate, test, and visually inspect**

```bash
.venv/bin/python scripts/generate_submission_assets.py
.venv/bin/python -m pytest tests/test_submission_docs.py -v
mkdir -p tmp/pdfs/human-centered-poster
pdftoppm -png -singlefile -r 150 submission/Silent-Vision-Poster.pdf tmp/pdfs/human-centered-poster/poster
```

Expected: tests PASS. Inspect both PNGs with `view_image` at original detail. Verify visible faces/mouths, readable labels/title, coherent grading, clear QR, no completed control, and no old cards/examples. Adjust only the affected scene's centering or smallest relevant geometry, regenerate, and repeat.

- [ ] **Step 8: Prove Project Profile stability and commit**

```bash
git diff --exit-code -- submission/Silent-Vision-Project-Profile.pdf output/pdf/Silent-Vision-Project-Profile.pdf
git add docs/submission/poster-copy.md scripts/generate_submission_assets.py tests/test_submission_docs.py submission/Silent-Vision-Poster.pdf submission/Silent-Vision-Poster.png output/pdf/Silent-Vision-Poster.pdf
git commit -m "feat: rebuild human-centered cinematic poster"
```

Expected: no Project Profile diff and one focused poster commit.

---

### Task 3: Verify Offline Bundle Rebuilding and Final Quality

**Files:**
- Modify: `tests/test_contest_bundle.py`

**Interfaces:**
- Consumes: Tasks 1-2 artwork, reviewed copy, and `build_poster_pdf()`.
- Produces: A bundle contract proving all source art is copied and the poster rebuilds without ImageGen or network access.

- [ ] **Step 1: Extend the bundle-content contract**

Import `subprocess` and `PdfReader`, then add:

```python
POSTER_SCENE_RELATIVE_PATHS = (
    Path("submission/assets/poster/post-surgery.png"),
    Path("submission/assets/poster/rehabilitation.png"),
    Path("submission/assets/poster/accessible-communication.png"),
    Path("submission/assets/poster/silent-control-input.png"),
)
```

In `test_bundle_contains_complete_runtime_and_public_materials()`, add:

```python
assert all(target.joinpath(path).is_file() for path in POSTER_SCENE_RELATIVE_PATHS)
```

- [ ] **Step 2: Write the offline rebuild test**

```python
def test_bundle_rebuilds_poster_from_repository_owned_assets(tmp_path):
    target = tmp_path / "submissions" / "track1-silent-vision"
    build_bundle(Path.cwd(), target)
    rebuilt = target / "rebuilt-poster.pdf"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from scripts.generate_submission_assets import build_poster_pdf; "
                "build_poster_pdf(Path('rebuilt-poster.pdf'))"
            ),
        ],
        cwd=target,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert rebuilt.is_file()
    reader = PdfReader(rebuilt)
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text() or ""
    assert "A VOICE WITHOUT SOUND." in text
    assert "PERSONALIZED FIXED-PHRASE PROTOTYPE" in text
```

- [ ] **Step 3: Run bundle tests and commit**

```bash
.venv/bin/python -m pytest tests/test_contest_bundle.py -v
git add tests/test_contest_bundle.py
git commit -m "test: verify offline cinematic poster rebuild"
```

Expected: all bundle tests PASS and the offline contract is committed.

- [ ] **Step 4: Prove deterministic regeneration**

```bash
.venv/bin/python scripts/generate_submission_assets.py
git diff --exit-code
```

Expected: successful regeneration with no tracked diff.

- [ ] **Step 5: Run complete automated verification**

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
```

Expected: Ruff passes; local tests pass with only environment-gated ROCm skips.

- [ ] **Step 6: Run final PDF metadata and visual verification**

```bash
pdfinfo submission/Silent-Vision-Poster.pdf
mkdir -p tmp/pdfs/human-centered-poster-final
pdftoppm -png -singlefile -r 150 submission/Silent-Vision-Poster.pdf tmp/pdfs/human-centered-poster-final/poster
```

Confirm one A3 portrait page, title `Silent Vision Poster`, author `Jixin Fang`, no clipping/overlap, four plausible people/environments, readable QR, and consistent color. Inspect with `view_image` at original detail, then delete only the temporary render files.

- [ ] **Step 7: Verify final repository state**

```bash
git status --short --branch
git diff --check
git log --oneline --max-count=6
```

Expected: clean feature worktree, no whitespace errors, and three poster implementation commits after the design and plan commits.
