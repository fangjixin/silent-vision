import re
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
CONTEST_PREFIX = "submissions/track1-silent-vision/"
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SCRIPT_REFERENCE = re.compile(r"(?<![\w/])(scripts/[A-Za-z0-9_.-]+)")


def _relative_markdown_links(path: Path) -> list[Path]:
    targets: list[Path] = []
    for raw_target in MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
        target = raw_target.strip("<>").split("#", maxsplit=1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        targets.append(path.parent / target)
    return targets


@pytest.mark.parametrize("path", [ROOT / "README.md", ROOT / "submission/README.md"])
def test_public_markdown_links_resolve(path):
    links = _relative_markdown_links(path)

    assert links
    assert all(link.exists() for link in links)


def test_readme_references_existing_scripts():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    scripts = {ROOT / match for match in SCRIPT_REFERENCE.findall(text)}

    assert scripts
    assert all(script.is_file() for script in scripts)


def test_submission_copy_avoids_unverified_marketing_claims():
    paths = [
        ROOT / "README.md",
        *ROOT.joinpath("submission").glob("*.md"),
        *ROOT.joinpath("docs/submission").glob("*.md"),
    ]
    banned = {
        "revolutionary",
        "game-changing",
        "cutting-edge",
        "next-generation",
        "harness the power of ai",
    }
    combined = "\n".join(
        path.read_text(encoding="utf-8").lower() for path in paths if path.exists()
    )

    assert not any(phrase in combined for phrase in banned)


def test_pr_description_contest_paths_resolve_to_public_materials():
    text = (ROOT / "submission/pull-request-description.md").read_text(
        encoding="utf-8"
    )
    relative_paths = {
        match
        for match in re.findall(rf"`{re.escape(CONTEST_PREFIX)}([^`]+)`", text)
    }

    assert relative_paths
    assert all(ROOT.joinpath(path).exists() for path in relative_paths)


def test_profile_and_poster_pdf_structure():
    profile_path = ROOT / "submission/Silent-Vision-Project-Profile.pdf"
    poster_path = ROOT / "submission/Silent-Vision-Poster.pdf"
    profile = PdfReader(profile_path)
    poster = PdfReader(poster_path)

    assert len(profile.pages) == 6
    assert len(poster.pages) == 1
    assert profile.metadata.title == "Silent Vision Project Profile"
    assert poster.metadata.title == "Silent Vision Poster"
    assert profile.metadata.author == poster.metadata.author == "Jixin Fang"
    assert all((page.extract_text() or "").strip() for page in profile.pages)
    assert (poster.pages[0].extract_text() or "").strip()

    profile_box = profile.pages[0].mediabox
    poster_box = poster.pages[0].mediabox
    assert float(profile_box.width) == pytest.approx(595.276, abs=0.01)
    assert float(profile_box.height) == pytest.approx(841.89, abs=0.01)
    assert float(poster_box.width) == pytest.approx(841.89, abs=0.01)
    assert float(poster_box.height) == pytest.approx(1190.55, abs=0.01)


def test_poster_png_is_a_full_size_a3_render():
    with Image.open(ROOT / "submission/Silent-Vision-Poster.png") as poster:
        width, height = poster.size

        assert poster.format == "PNG"
        assert width >= 1700
        assert height >= 2400
        assert height / width == pytest.approx(2**0.5, rel=0.01)
