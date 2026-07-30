from pathlib import Path


def test_docker_compose_mounts_persistence_root_and_rocm_devices():
    compose = Path("docker/docker-compose.yml").read_text()
    assert "/workspace/persistence/silent-vision:/workspace/persistence/silent-vision" in compose
    assert "/dev/kfd:/dev/kfd" in compose
    assert "/dev/dri:/dev/dri" in compose
    assert "127.0.0.1:8000:8000" in compose


def test_dockerfile_does_not_copy_model_weights():
    dockerfile = Path("docker/Dockerfile").read_text()
    assert "COPY . /app" in dockerfile
    assert "models/" not in dockerfile


def test_smoke_scripts_exist_and_are_executable():
    fake = Path("scripts/smoke_fake.sh")
    rocm = Path("scripts/smoke_rocm.sh")
    assert fake.exists()
    assert rocm.exists()
    assert fake.read_text().startswith("#!/usr/bin/env bash")
    assert rocm.read_text().startswith("#!/usr/bin/env bash")
