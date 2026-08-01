from pathlib import Path


def test_docker_compose_mounts_persistence_root_and_rocm_devices():
    compose = Path("docker/docker-compose.yml").read_text()
    assert "/workspace/persistent/silent-vision:/workspace/persistent/silent-vision" in compose
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
    setup = Path("scripts/setup_amd_real.sh")
    start = Path("scripts/start_real_rocm.sh")
    assert fake.exists()
    assert rocm.exists()
    assert setup.exists()
    assert start.exists()
    assert fake.read_text().startswith("#!/usr/bin/env bash")
    assert rocm.read_text().startswith("#!/usr/bin/env bash")
    assert setup.read_text().startswith("#!/usr/bin/env bash")
    assert start.read_text().startswith("#!/usr/bin/env bash")
    assert "Visual_Speech_Recognition_for_Multiple_Languages" in rocm.read_text()
    assert "models/avhubert/model.pt" not in rocm.read_text()
    assert "/opt/venv/bin/python" in setup.read_text()
    assert "/opt/venv/bin/python" in start.read_text()
    assert "/workspace/persistent/silent-vision" in setup.read_text()
    assert "/workspace/persistent/silent-vision" in start.read_text()


def test_frontend_stream_lifecycle_cleans_up_between_starts():
    websocket_js = Path("frontend/websocket.js").read_text()
    assert "cleanupCurrentStream();" in websocket_js
    assert "state.ws.close(1000, \"client stopped stream\");" in websocket_js
    assert "state.ws.bufferedAmount > 2_000_000" in websocket_js
    assert "state.streaming = false;" in websocket_js
    assert "mouth reused" in websocket_js


def test_real_mode_suppresses_noisy_third_party_warnings():
    minicpm = Path("llm/minicpm.py").read_text()
    face = Path("vision/face.py").read_text()
    assert "image_processor_class argument is deprecated" in minicpm
    assert "Using a slow image processor" in minicpm
    assert "SymbolDatabase.GetPrototype" in face
