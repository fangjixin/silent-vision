#!/usr/bin/env python3
"""Generate the reviewed Silent Vision project profile and poster assets.

This module is the canonical copy and layout source for generated submission
artifacts. The Markdown files under ``docs/submission`` are reviewed reference
copy and must be updated alongside high-risk copy or status changes here.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import qrcode
from qrcode.constants import ERROR_CORRECT_M
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.pagesizes import A3, A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
PROFILE_SOURCE = ROOT / "docs/submission/project-profile-source.md"
POSTER_SOURCE = ROOT / "docs/submission/poster-copy.md"
SUBMISSION = ROOT / "submission"
PDF_OUTPUT = ROOT / "output/pdf"
REPOSITORY_URL = "https://github.com/fangjixin/silent-vision"
PROFILE_EVIDENCE_STATUS = (
    "Pending: final Radeon checkpoint, held-out validation report, selected "
    "environment record, Creator Mode actions, and the 3-5 minute end-to-end "
    "video. No accuracy or latency number is published."
)
POSTER_EVIDENCE_STATUS = (
    "Final Radeon run, trained checkpoint, validation evidence, Creator Mode "
    "actions, and recorded demo are pending."
)

CHARCOAL = HexColor("#101418")
OFF_WHITE = HexColor("#F4F1EA")
RADEON_RED = HexColor("#ED1C24")
MUTED = HexColor("#56616A")
LINE = HexColor("#CFD1CC")
WHITE = HexColor("#FFFFFF")
PALE_RED = HexColor("#FDE7E7")
PALE_GRAY = HexColor("#E8E8E3")


def _read_reviewed_copy() -> tuple[str, str]:
    profile = PROFILE_SOURCE.read_text(encoding="utf-8")
    poster = POSTER_SOURCE.read_text(encoding="utf-8")
    required = {
        "profile": (profile, ["Applicant: Jixin Fang", REPOSITORY_URL, PROFILE_EVIDENCE_STATUS]),
        "poster": (poster, [REPOSITORY_URL, POSTER_EVIDENCE_STATUS]),
    }
    missing = [
        f"{document}: {phrase}"
        for document, (text, phrases) in required.items()
        for phrase in phrases
        if phrase not in " ".join(text.split())
    ]
    if missing:
        raise ValueError(f"Reviewed submission copy is missing: {missing}")
    return profile, poster


def _wrap(text: str, width: float, font: str, size: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or stringWidth(candidate, font, size) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _paragraph(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    *,
    font: str = "Helvetica",
    size: float = 10.5,
    leading: float = 15,
    color: Color = CHARCOAL,
) -> float:
    pdf.setFillColor(color)
    pdf.setFont(font, size)
    for line in _wrap(text, width, font, size):
        pdf.drawString(x, y, line)
        y -= leading
    return y


def _label(pdf: canvas.Canvas, text: str, x: float, y: float, color: Color = RADEON_RED) -> None:
    pdf.setFillColor(color)
    pdf.setFont("Helvetica-Bold", 8.5)
    pdf.drawString(x, y, text.upper())


def _card(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    accent: Color = RADEON_RED,
    fill: Color = WHITE,
    title_size: float = 13,
    body_size: float = 9.5,
    leading: float = 13.5,
) -> None:
    pdf.setFillColor(fill)
    pdf.setStrokeColor(LINE)
    pdf.roundRect(x, y, width, height, 9, fill=1, stroke=1)
    pdf.setFillColor(accent)
    pdf.roundRect(x, y + height - 8, width, 8, 5, fill=1, stroke=0)
    pdf.setFillColor(CHARCOAL)
    pdf.setFont("Helvetica-Bold", title_size)
    pdf.drawString(x + 14, y + height - 31, title)
    _paragraph(
        pdf,
        body,
        x + 14,
        y + height - 50,
        width - 28,
        size=body_size,
        leading=leading,
        color=MUTED,
    )


def _arrow(pdf: canvas.Canvas, x1: float, y1: float, x2: float, y2: float) -> None:
    pdf.setStrokeColor(RADEON_RED)
    pdf.setFillColor(RADEON_RED)
    pdf.setLineWidth(1.8)
    pdf.line(x1, y1, x2, y2)
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 >= x1 else -1
        points = [x2, y2, x2 - 6 * direction, y2 + 4, x2 - 6 * direction, y2 - 4]
    else:
        direction = 1 if y2 >= y1 else -1
        points = [x2, y2, x2 - 4, y2 - 6 * direction, x2 + 4, y2 - 6 * direction]
    path = pdf.beginPath()
    path.moveTo(points[0], points[1])
    path.lineTo(points[2], points[3])
    path.lineTo(points[4], points[5])
    path.close()
    pdf.drawPath(path, fill=1, stroke=0)


def _profile_shell(
    pdf: canvas.Canvas,
    page_number: int,
    title: str,
    subtitle: str,
) -> None:
    width, height = A4
    pdf.setFillColor(OFF_WHITE)
    pdf.rect(0, 0, width, height, fill=1, stroke=0)
    pdf.setFillColor(RADEON_RED)
    pdf.rect(0, height - 13, width, 13, fill=1, stroke=0)
    _label(pdf, "Silent Vision / Track 1", 42, height - 39, MUTED)
    pdf.setFillColor(CHARCOAL)
    pdf.setFont("Helvetica-Bold", 26)
    pdf.drawString(42, height - 78, title)
    _paragraph(pdf, subtitle, 42, height - 101, width - 84, size=10, leading=14, color=MUTED)
    pdf.setStrokeColor(LINE)
    pdf.line(42, 38, width - 42, 38)
    pdf.setFont("Helvetica", 8.5)
    pdf.setFillColor(MUTED)
    pdf.drawString(42, 23, "Jixin Fang  |  github.com/fangjixin/silent-vision")
    pdf.drawRightString(width - 42, 23, f"{page_number} / 6")


def _profile_page_one(pdf: canvas.Canvas) -> None:
    _profile_shell(
        pdf,
        1,
        "Silent Vision",
        "Closed-set visual command recognition from a short, silent camera clip.",
    )
    width, height = A4
    _label(pdf, "Project profile", 42, height - 144)
    pdf.setFont("Helvetica-Bold", 37)
    pdf.setFillColor(CHARCOAL)
    pdf.drawString(42, height - 192, "Silent control when")
    pdf.setFillColor(RADEON_RED)
    pdf.drawString(42, height - 235, "audio is not an option.")
    _paragraph(
        pdf,
        "The browser records a 2-5 second WebM clip without audio. The backend extracts a mouth-region sequence, checks confidence and margin, and returns a structured command decision plus an agent result.",
        42,
        height - 276,
        width - 84,
        size=11,
        leading=16,
    )
    card_width = (width - 96) / 2
    _card(
        pdf,
        42,
        355,
        card_width,
        126,
        "Bounded vocabulary",
        "LIGHT_ON, LIGHT_OFF, OPEN_DOOR, CHAT_OTHER, and UNKNOWN. Silent Vision does not claim open-ended transcription.",
    )
    _card(
        pdf,
        54 + card_width,
        355,
        card_width,
        126,
        "Inspectable result",
        "Candidates, confidence, margin, acceptance reason, and execute, ignore, or reject are returned as structured data.",
    )
    pdf.setFillColor(CHARCOAL)
    pdf.roundRect(42, 196, width - 84, 126, 10, fill=1, stroke=0)
    _label(pdf, "Current status", 60, 294, RADEON_RED)
    _paragraph(
        pdf,
        "The proof of concept does not control a physical light or door and does not create a browser recording or still-image artifact. The intended demo uses CPU preprocessing and a PyTorch temporal classifier on AMD Radeon through ROCm.",
        60,
        268,
        width - 120,
        size=10.5,
        leading=15,
        color=WHITE,
    )
    pdf.setFillColor(PALE_RED)
    pdf.roundRect(42, 102, width - 84, 64, 9, fill=1, stroke=0)
    _label(pdf, "Pending", 58, 144)
    _paragraph(
        pdf,
        "Final Radeon checkpoint, validation evidence, and recorded demonstration.",
        58,
        122,
        width - 116,
        size=10,
        leading=14,
    )


def _profile_page_two(pdf: canvas.Canvas) -> None:
    _profile_shell(
        pdf,
        2,
        "Target users",
        "A deliberate visual input for situations where audio is unavailable, unreliable, or unwanted.",
    )
    width, _ = A4
    cards = [
        ("Accessible service", "A Deaf or hard-of-hearing person may need a visual alternative at a service desk."),
        ("Creator studio - planned", "A creator may need hands-free control near a loud set. Browser recording and still capture are not implemented."),
        ("Noisy worksite", "An operator may work around machinery where speech recognition is unreliable."),
        ("Privacy-sensitive team", "A deliberate camera gesture can replace an always-listening microphone for a small command vocabulary."),
    ]
    card_width = (width - 96) / 2
    positions = [(42, 500), (54 + card_width, 500), (42, 328), (54 + card_width, 328)]
    for (title, body), (x, y) in zip(cards, positions):
        _card(pdf, x, y, card_width, 142, title, body, body_size=9.8)
    pdf.setFillColor(CHARCOAL)
    pdf.roundRect(42, 170, width - 84, 116, 10, fill=1, stroke=0)
    _label(pdf, "Product boundary", 60, 256)
    _paragraph(
        pdf,
        'Silent Vision answers "which allowed command best fits this clip?" It does not claim a transcript. An integrator decides exactly which accepted labels may reach a downstream system.',
        60,
        228,
        width - 120,
        size=11,
        leading=16,
        color=WHITE,
    )
    _paragraph(
        pdf,
        "The current smart-space labels demonstrate that interface boundary. Creator-control actions remain planned work.",
        42,
        128,
        width - 84,
        size=10,
        leading=14,
        color=MUTED,
    )


def _profile_page_three(pdf: canvas.Canvas) -> None:
    _profile_shell(
        pdf,
        3,
        "Product workflow and safety",
        "One short recording, one bounded decision, and explicit uncertainty handling.",
    )
    width, _ = A4
    steps = [
        ("01", "Start", "User grants camera access; audio remains off."),
        ("02", "Prepare", "A countdown gives the user time to frame the command."),
        ("03", "Record", "The browser records one WebM clip for up to five seconds."),
        ("04", "Classify", "The server returns candidates, confidence, margin, and a reason."),
        ("05", "Route", "The agent boundary returns execute, ignore, or reject."),
    ]
    y = 615
    for index, title, body in steps:
        pdf.setFillColor(WHITE)
        pdf.setStrokeColor(LINE)
        pdf.roundRect(60, y, width - 120, 68, 9, fill=1, stroke=1)
        pdf.setFillColor(RADEON_RED)
        pdf.setFont("Helvetica-Bold", 20)
        pdf.drawString(76, y + 25, index)
        pdf.setFillColor(CHARCOAL)
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(126, y + 39, title)
        _paragraph(pdf, body, 126, y + 20, width - 214, size=9.5, leading=12, color=MUTED)
        if index != "05":
            _arrow(pdf, width / 2, y - 3, width / 2, y - 15)
        y -= 89
    pdf.setFillColor(PALE_RED)
    pdf.roundRect(42, 104, width - 84, 68, 9, fill=1, stroke=0)
    _label(pdf, "Safety rule", 58, 150)
    _paragraph(
        pdf,
        "Low confidence, a small top-1/top-2 margin, or an UNKNOWN prediction produces a rejection. CHAT_OTHER can be recognized but non-executable.",
        58,
        128,
        width - 116,
        size=9.8,
        leading=13.5,
    )


def _profile_page_four(pdf: canvas.Canvas) -> None:
    _profile_shell(
        pdf,
        4,
        "System architecture",
        "A CPU media pipeline feeds interchangeable command backends with one decision schema.",
    )
    width, _ = A4
    box_width = 112
    x_values = [42, 173, 304, 435]
    stages = [
        ("Browser", "WebM, audio off"),
        ("FastAPI", "Session + WebSocket"),
        ("PyAV", "Decode at 25 FPS"),
        ("MediaPipe", "96 x 96 mouth ROI"),
    ]
    for i, ((title, body), x) in enumerate(zip(stages, x_values)):
        _card(pdf, x, 540, box_width, 112, title, body, body_size=8.4, leading=11.5)
        if i < len(stages) - 1:
            _arrow(pdf, x + box_width + 4, 596, x_values[i + 1] - 5, 596)
    _arrow(pdf, width / 2, 530, width / 2, 490)
    pdf.setFillColor(CHARCOAL)
    pdf.roundRect(90, 419, width - 180, 70, 9, fill=1, stroke=0)
    _label(pdf, "Shared temporal sequence", 108, 463)
    _paragraph(
        pdf,
        "One face, stabilized crop, grayscale mouth frames.",
        108,
        440,
        width - 216,
        size=10.5,
        leading=14,
        color=WHITE,
    )
    branch_width = 148
    branch_x = [42, 224, 406]
    backends = [
        ("Fake", "Deterministic local tests."),
        ("Prototype", "NumPy embedding and saved global examples."),
        ("Torch", "Temporal classifier for the intended ROCm path."),
    ]
    for (title, body), x in zip(backends, branch_x):
        _arrow(pdf, width / 2, 410, x + branch_width / 2, 365)
        _card(pdf, x, 255, branch_width, 105, title, body, body_size=8.7, leading=12)
        _arrow(pdf, x + branch_width / 2, 247, width / 2, 212)
    pdf.setFillColor(PALE_RED)
    pdf.roundRect(90, 135, width - 180, 76, 9, fill=1, stroke=0)
    _label(pdf, "CommandDecision", 108, 185)
    _paragraph(
        pdf,
        "Candidates + confidence + margin + reason, then execute, ignore, or reject.",
        108,
        160,
        width - 216,
        size=9.7,
        leading=13,
    )


def _profile_page_five(pdf: canvas.Canvas) -> None:
    _profile_shell(
        pdf,
        5,
        "Model and algorithm",
        "Prototype calibration and Torch classification share the same rejection boundary.",
    )
    width, _ = A4
    pipeline = [
        ("Frame features", "Mean, standard deviation, and motion values"),
        ("4 temporal blocks", "Feed-forward, attention, depthwise 1D convolution, normalization"),
        ("Attentive pooling", "Reduce the temporal sequence"),
        ("Label classifier", "Score the bounded vocabulary"),
    ]
    y = 590
    for i, (title, body) in enumerate(pipeline):
        pdf.setFillColor(WHITE)
        pdf.setStrokeColor(LINE)
        pdf.roundRect(60, y, width - 120, 72, 9, fill=1, stroke=1)
        pdf.setFillColor(RADEON_RED)
        pdf.circle(86, y + 36, 14, fill=1, stroke=0)
        pdf.setFillColor(WHITE)
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawCentredString(86, y + 33, str(i + 1))
        pdf.setFillColor(CHARCOAL)
        pdf.setFont("Helvetica-Bold", 12.5)
        pdf.drawString(114, y + 43, title)
        _paragraph(pdf, body, 114, y + 24, width - 204, size=9.2, leading=12, color=MUTED)
        if i < len(pipeline) - 1:
            _arrow(pdf, width / 2, y - 3, width / 2, y - 16)
        y -= 96
    card_width = (width - 96) / 2
    _card(
        pdf,
        42,
        130,
        card_width,
        112,
        "Prototype mode",
        "Normalizes an appearance-and-motion embedding, compares cosine similarity, then checks confidence and margin. It is a calibration tool, not Radeon evidence.",
        body_size=8.7,
        leading=12,
    )
    _card(
        pdf,
        54 + card_width,
        130,
        card_width,
        112,
        "Training status",
        "Scripts exist for train, validate, and infer. Real mouth_roi_npy rows and a separate held-out validation manifest are still required.",
        body_size=8.7,
        leading=12,
    )


def _profile_page_six(pdf: canvas.Canvas) -> None:
    _profile_shell(
        pdf,
        6,
        "AMD Radeon and ROCm",
        "Intended demo path: CPU preprocessing, then PyTorch temporal classification on Radeon.",
    )
    width, _ = A4
    card_width = (width - 96) / 2
    _card(
        pdf,
        42,
        500,
        card_width,
        148,
        "CPU preprocessing",
        "PyAV decoding, MediaPipe detection, crop stabilization, and NumPy feature work stay on CPU.",
        accent=MUTED,
    )
    _arrow(pdf, 54 + card_width - 8, 574, 54 + card_width + 8, 574)
    _card(
        pdf,
        54 + card_width,
        500,
        card_width,
        148,
        "Radeon classification",
        "PyTorch runs the temporal classifier through ROCm. The ROCm-backed device appears as cuda:0 and torch.version.hip records HIP availability.",
    )
    _label(pdf, "Reproduction guards", 42, 458)
    guards = [
        "HIP must be present.",
        "The accelerator must be available.",
        "A Torch checkpoint path must be provided.",
    ]
    y = 420
    for guard in guards:
        pdf.setFillColor(RADEON_RED)
        pdf.circle(53, y + 4, 4, fill=1, stroke=0)
        _paragraph(pdf, guard, 68, y, width - 110, size=10.5, leading=14)
        y -= 35
    pdf.setFillColor(PALE_RED)
    pdf.roundRect(42, 240, width - 84, 90, 9, fill=1, stroke=0)
    _label(pdf, "Evidence status", 58, 304)
    _paragraph(
        pdf,
        PROFILE_EVIDENCE_STATUS,
        58,
        278,
        width - 116,
        size=9.7,
        leading=13.5,
    )
    _label(pdf, "Next steps", 42, 207)
    _paragraph(
        pdf,
        "1. Finish a balanced dataset.   2. Build a real manifest.   3. Train on Radeon.   4. Evaluate a held-out manifest.   5. Save reports and record the demo.",
        42,
        180,
        width - 84,
        size=10,
        leading=15,
    )
    pdf.setFillColor(CHARCOAL)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(42, 105, REPOSITORY_URL)


def build_profile_pdf(output: Path) -> None:
    """Create the six-page A4 project profile."""
    _read_reviewed_copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=A4, invariant=1, pageCompression=1)
    pdf.setTitle("Silent Vision Project Profile")
    pdf.setAuthor("Jixin Fang")
    pdf.setSubject("Track 1 project profile")
    pages = [
        _profile_page_one,
        _profile_page_two,
        _profile_page_three,
        _profile_page_four,
        _profile_page_five,
        _profile_page_six,
    ]
    for page in pages:
        page(pdf)
        pdf.showPage()
    pdf.save()


def _qr_image() -> ImageReader:
    code = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_M,
        box_size=12,
        border=4,
    )
    code.add_data(REPOSITORY_URL)
    code.make(fit=True)
    image = code.make_image(fill_color="#101418", back_color="#FFFFFF")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return ImageReader(buffer)


def build_poster_pdf(output: Path) -> None:
    """Create the one-page A3 portrait poster."""
    _read_reviewed_copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=A3, invariant=1, pageCompression=1)
    pdf.setTitle("Silent Vision Poster")
    pdf.setAuthor("Jixin Fang")
    pdf.setSubject("Track 1 project poster")
    width, height = A3
    pdf.setFillColor(OFF_WHITE)
    pdf.rect(0, 0, width, height, fill=1, stroke=0)
    pdf.setFillColor(RADEON_RED)
    pdf.rect(0, height - 20, width, 20, fill=1, stroke=0)
    _label(pdf, "Track 1 / Jixin Fang", 60, height - 62, MUTED)
    pdf.setFillColor(CHARCOAL)
    pdf.setFont("Helvetica-Bold", 62)
    pdf.drawString(60, height - 137, "SILENT VISION")
    pdf.setFillColor(RADEON_RED)
    pdf.setFont("Helvetica-Bold", 29)
    pdf.drawString(60, height - 184, "Silent control when audio is not an option.")
    _paragraph(
        pdf,
        "Closed-set visual command recognition from a short camera clip.",
        60,
        height - 220,
        width - 120,
        size=17,
        leading=22,
        color=MUTED,
    )
    _label(pdf, "How it works", 60, height - 287)
    step_titles = ["Record", "Decode", "Find mouth", "Classify", "Route"]
    step_bodies = [
        "2-5 second WebM; audio off",
        "25 FPS on CPU",
        "96 x 96 sequence",
        "Intent + confidence + margin",
        "Execute, ignore, or reject",
    ]
    gap = 14
    step_width = (width - 120 - gap * 4) / 5
    y = height - 472
    for index, (title, body) in enumerate(zip(step_titles, step_bodies)):
        x = 60 + index * (step_width + gap)
        _card(
            pdf,
            x,
            y,
            step_width,
            145,
            f"{index + 1}. {title}",
            body,
            title_size=12.5,
            body_size=9.2,
            leading=13,
        )
        if index < 4:
            _arrow(pdf, x + step_width + 3, y + 72, x + step_width + gap - 3, y + 72)
    _label(pdf, "Where it fits", 60, height - 522)
    fit_width = (width - 148) / 3
    fits = [
        ("Accessible service", "A visual alternative when spoken audio is not available."),
        ("Creator studio - planned", "A future hands-free input for recording and still capture."),
        ("Noisy worksite", "A small vocabulary when microphones are unreliable."),
    ]
    for index, (title, body) in enumerate(fits):
        _card(
            pdf,
            60 + index * (fit_width + 14),
            height - 718,
            fit_width,
            155,
            title,
            body,
            body_size=10,
            leading=14,
        )
    pdf.setFillColor(CHARCOAL)
    pdf.roundRect(60, height - 850, width - 120, 92, 12, fill=1, stroke=0)
    _label(pdf, "Safety rule", 82, height - 790)
    _paragraph(
        pdf,
        "Low-confidence and ambiguous commands do not execute.",
        82,
        height - 822,
        width - 164,
        font="Helvetica-Bold",
        size=18,
        leading=23,
        color=WHITE,
    )
    _label(pdf, "AMD Radeon + ROCm", 60, height - 910)
    _paragraph(
        pdf,
        "Intended demo path: CPU video preprocessing, then PyTorch temporal classification on AMD Radeon through ROCm.",
        60,
        height - 944,
        width - 350,
        size=13,
        leading=19,
    )
    pdf.setFillColor(PALE_RED)
    pdf.roundRect(60, 118, width - 360, 92, 10, fill=1, stroke=0)
    _label(pdf, "Evidence status", 78, 182)
    _paragraph(
        pdf,
        POSTER_EVIDENCE_STATUS,
        78,
        154,
        width - 396,
        size=10.5,
        leading=15,
    )
    qr_size = 158
    qr_x = width - 60 - qr_size
    pdf.setFillColor(WHITE)
    pdf.roundRect(qr_x - 10, 104, qr_size + 20, qr_size + 45, 10, fill=1, stroke=0)
    pdf.drawImage(_qr_image(), qr_x, 132, qr_size, qr_size, preserveAspectRatio=True, mask="auto")
    pdf.setFillColor(CHARCOAL)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawCentredString(qr_x + qr_size / 2, 116, "SOURCE REPOSITORY")
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(60, 74, "github.com/fangjixin/silent-vision")
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 8.5)
    pdf.drawRightString(width - 60, 74, "Structured decisions, explicit uncertainty, no audio recording")
    pdf.showPage()
    pdf.save()


def build_poster_png(pdf: Path, output: Path) -> None:
    """Render the poster PDF to a 150 DPI PNG preview with Poppler."""
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise RuntimeError("Poppler is required: install pdftoppm to render the poster PNG")
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.with_suffix("")
    subprocess.run(
        [executable, "-png", "-singlefile", "-r", "150", str(pdf), str(prefix)],
        check=True,
    )
    if not output.exists():
        raise RuntimeError(f"Poppler did not create the expected preview: {output}")


def main() -> None:
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    PDF_OUTPUT.mkdir(parents=True, exist_ok=True)
    profile = SUBMISSION / "Silent-Vision-Project-Profile.pdf"
    poster = SUBMISSION / "Silent-Vision-Poster.pdf"
    poster_png = SUBMISSION / "Silent-Vision-Poster.png"
    build_profile_pdf(profile)
    build_poster_pdf(poster)
    build_poster_png(poster, poster_png)
    shutil.copyfile(profile, PDF_OUTPUT / profile.name)
    shutil.copyfile(poster, PDF_OUTPUT / poster.name)
    print(f"Generated {profile.relative_to(ROOT)}")
    print(f"Generated {poster.relative_to(ROOT)}")
    print(f"Generated {poster_png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
