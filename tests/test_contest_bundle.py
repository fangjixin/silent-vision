import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_contest_bundle import build_bundle


def test_bundle_contains_complete_runtime_and_public_materials(tmp_path):
    target = tmp_path / "submissions" / "track1-silent-vision"

    copied = build_bundle(Path.cwd(), target)

    assert target.joinpath("backend/main.py").exists()
    assert target.joinpath("command/phrase_catalog.json").exists()
    assert target.joinpath("frontend/websocket.js").exists()
    assert target.joinpath("README.md").exists()
    assert target.joinpath("docs/submission/project-profile-source.md").exists()
    assert target.joinpath("scripts/generate_submission_assets.py").exists()
    assert target.joinpath("submission/Silent-Vision-Project-Profile.pdf").exists()
    assert target.joinpath("submission/Silent-Vision-Poster.pdf").exists()
    assert target.joinpath("submission/Silent-Vision-Poster.png").exists()
    assert target.joinpath("submission/demo-video-script.md").exists()
    assert copied


def test_bundle_excludes_internal_and_sensitive_paths(tmp_path):
    target = tmp_path / "bundle"

    build_bundle(Path.cwd(), target)

    assert not target.joinpath("docs/superpowers").exists()
    assert not target.joinpath(".env").exists()
    assert not target.joinpath("models").exists()
    assert not target.joinpath(".git").exists()
    assert not any(path.suffix in {".pt", ".webm"} for path in target.rglob("*"))
    assert not any(path.is_symlink() for path in target.rglob("*"))


def test_bundle_rejects_an_allowlisted_symlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("# Safe source\n", encoding="utf-8")
    (source / "agent").mkdir()
    (source / "agent" / "linked.py").symlink_to(source / "README.md")

    with pytest.raises(ValueError, match="symlink"):
        build_bundle(source, tmp_path / "bundle")


def test_bundle_excludes_secrets_inside_an_allowlisted_directory(tmp_path):
    source = tmp_path / "source"
    secret = source / "backend" / "secrets" / "token.txt"
    secret.parent.mkdir(parents=True)
    secret.write_text("do not publish", encoding="utf-8")

    build_bundle(source, tmp_path / "bundle")

    assert not (tmp_path / "bundle" / "backend" / "secrets" / "token.txt").exists()


def test_bundle_rejects_a_symlink_destination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("# Safe source\n", encoding="utf-8")
    destination_target = tmp_path / "destination-target"
    destination_target.mkdir()
    destination = tmp_path / "bundle"
    destination.symlink_to(destination_target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        build_bundle(source, destination)


def test_bundle_rejects_destination_beneath_a_symlinked_ancestor(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("# Safe source\n", encoding="utf-8")
    victim = tmp_path / "victim"
    victim.mkdir()
    sentinel = victim / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(victim, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        build_bundle(source, alias / "bundle")

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_bundle_rejects_source_beneath_a_symlinked_ancestor(tmp_path):
    real_parent = tmp_path / "real-parent"
    source = real_parent / "source"
    source.mkdir(parents=True)
    (source / "README.md").write_text("# Safe source\n", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        build_bundle(alias / "source", tmp_path / "bundle")


def test_bundle_rejects_a_broken_root_allowlist_symlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").symlink_to(source / "missing-readme.md")

    with pytest.raises(ValueError, match="symlink"):
        build_bundle(source, tmp_path / "bundle")


@pytest.mark.parametrize("destination_kind", ["same", "ancestor", "descendant"])
def test_bundle_rejects_destination_that_overlaps_source(tmp_path, destination_kind):
    source = tmp_path / "source"
    backend = source / "backend"
    backend.mkdir(parents=True)
    sentinel = backend / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    destinations = {
        "same": source,
        "ancestor": tmp_path,
        "descendant": backend,
    }
    with pytest.raises(ValueError, match="overlap"):
        build_bundle(source, destinations[destination_kind])

    assert sentinel.read_text(encoding="utf-8") == "keep"
