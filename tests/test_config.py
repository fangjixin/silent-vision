from pathlib import Path

from backend.config import Settings
from backend.main import create_app


def test_settings_defaults_use_fake_backend_and_persistence_root():
    settings = Settings()
    assert settings.model_backend == "fake"
    assert settings.persistence_root == Path("/workspace/persistence/silent-vision")
    assert settings.window_frames == 75
    assert settings.inference_stride == 25
    assert settings.mouth_size == 96
    assert settings.capture_fps == 25


def test_settings_model_paths_are_derived_from_persistence_root():
    settings = Settings()
    assert settings.avhubert_checkpoint == Path("/workspace/persistence/silent-vision/models/avhubert/model.pt")
    assert settings.cmlr_checkpoint == Path("/workspace/persistence/silent-vision/models/cmlr/model.pth")
    assert settings.cmlr_language_model == Path("/workspace/persistence/silent-vision/models/cmlr/language-model.pth")
    assert settings.minicpm_model_path == Path("/workspace/persistence/silent-vision/models/minicpm-o-4_5")


def test_create_app_registers_health_routes():
    app = create_app(Settings())
    route_paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/health/live" in route_paths
    assert "/health/ready" in route_paths
