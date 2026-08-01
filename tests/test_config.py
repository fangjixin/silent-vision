from pathlib import Path

from backend.config import Settings
from backend.main import create_app


def test_settings_defaults_use_fake_backend_and_persistence_root():
    settings = Settings()
    assert settings.model_backend == "fake"
    assert settings.persistence_root == Path("/workspace/persistent/silent-vision")
    assert settings.window_frames == 75
    assert settings.inference_stride == 25
    assert settings.mouth_size == 96
    assert settings.capture_fps == 25


def test_settings_model_paths_are_derived_from_persistence_root():
    settings = Settings()
    assert settings.mpc001_repo_path == Path(
        "/workspace/persistent/silent-vision/repos/Visual_Speech_Recognition_for_Multiple_Languages"
    )
    assert settings.mpc001_english_config_path == Path(
        "/workspace/persistent/silent-vision/repos/Visual_Speech_Recognition_for_Multiple_Languages/configs/LRS3_V_WER19.1.ini"
    )
    assert settings.mpc001_chinese_config_path == Path(
        "/workspace/persistent/silent-vision/repos/Visual_Speech_Recognition_for_Multiple_Languages/configs/CMLR_V_WER8.0.ini"
    )
    assert settings.mpc001_mouth_runner_path.name == "mpc001_mouth_infer.py"
    assert settings.minicpm_model_path == Path("/workspace/persistent/silent-vision/models/minicpm-o-4_5")


def test_create_app_registers_health_routes():
    app = create_app(Settings())
    route_paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/health/live" in route_paths
    assert "/health/ready" in route_paths


def test_ready_route_reports_fake_models_ready():
    from fastapi.testclient import TestClient

    app = create_app(Settings(model_backend="fake"))
    response = TestClient(app).get("/health/ready")
    assert response.status_code == 200
    assert response.json()["models"] == {"backend": "fake", "ready": True}
