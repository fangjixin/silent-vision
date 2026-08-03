from fastapi.testclient import TestClient
import numpy as np

from backend.config import Settings
from backend.main import create_app
from video.clip import DecodedClip


class FakeRoiClip:
    def __init__(self) -> None:
        frames = np.zeros((25, 96, 96), dtype=np.uint8)
        frames[5:15, 24:72, 24:72] = 255
        self.mouth_frames = frames
        self.aligned_face_frames = np.zeros((25, 224, 224, 3), dtype=np.uint8)
        self.detected_frames = 25
        self.reused_frames = 0


def patch_clip_pipeline(monkeypatch):
    monkeypatch.setattr(
        "api.websocket.decode_video_clip",
        lambda data, target_fps: DecodedClip(
            frames=tuple(np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(25)),
            fps=target_fps,
            duration_ms=1000,
        ),
    )
    monkeypatch.setattr("api.websocket.extract_mouth_roi_clip", lambda **kwargs: FakeRoiClip())


def test_fake_websocket_flow_reaches_agent_result(app, monkeypatch):
    patch_clip_pipeline(monkeypatch)
    client = TestClient(app)
    session_response = client.post("/api/sessions")
    session_id = session_response.json()["sessionId"]
    with client.websocket_connect(f"/ws/{session_id}", headers={"origin": "http://localhost:8000"}) as websocket:
        ready = websocket.receive_json()
        assert ready["type"] == "session.ready"
        websocket.send_json({"type": "clip.start", "profileId": "test-profile"})
        started = websocket.receive_json()
        assert started["type"] == "clip.started"
        websocket.send_bytes(b"fake-webm")
        seen = set()
        for _ in range(20):
            message = websocket.receive_json()
            seen.add(message["type"])
            if message["type"] == "agent.result":
                assert message["action"] in {"execute", "reject", "ignore"}
                break
        assert {"clip.received", "command.result", "agent.result"} <= seen


def test_second_active_websocket_takes_over_slot(app):
    client = TestClient(app)
    first_id = client.post("/api/sessions").json()["sessionId"]
    second_id = client.post("/api/sessions").json()["sessionId"]
    with client.websocket_connect(f"/ws/{first_id}", headers={"origin": "http://localhost:8000"}) as first:
        assert first.receive_json()["type"] == "session.ready"
        with client.websocket_connect(f"/ws/{second_id}", headers={"origin": "http://localhost:8000"}) as second:
            assert second.receive_json()["type"] == "session.ready"
            first.send_json({"type": "ping"})
            replaced = first.receive_json()
            assert replaced["type"] == "error"
            assert replaced["code"] == "SESSION_REPLACED"


def test_frontend_index_is_served(app):
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Silent Vision" in response.text
    assert "startButton" in response.text


def test_websocket_allows_wildcard_origin():
    app = create_app(Settings(command_backend="fake", allowed_origins="*"))
    client = TestClient(app)
    session_id = client.post("/api/sessions").json()["sessionId"]

    with client.websocket_connect(
        f"/ws/{session_id}",
        headers={"origin": "https://rc-b80bfa02edcceefa.radeon.firstdg.ai"},
    ) as websocket:
        assert websocket.receive_json()["type"] == "session.ready"


def test_invalid_clip_processing_is_recoverable_websocket_error(monkeypatch):
    def fail_decode(data, target_fps):
        raise ValueError("bad clip")

    monkeypatch.setattr("api.websocket.decode_video_clip", fail_decode)
    app = create_app(Settings(command_backend="fake"))
    client = TestClient(app)
    session_id = client.post("/api/sessions").json()["sessionId"]

    with client.websocket_connect(f"/ws/{session_id}", headers={"origin": "http://localhost:8000"}) as websocket:
        assert websocket.receive_json()["type"] == "session.ready"
        websocket.send_json({"type": "clip.start"})
        assert websocket.receive_json()["type"] == "clip.started"
        websocket.send_bytes(b"bad-webm")
        assert websocket.receive_json()["type"] == "clip.received"
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "INTERNAL_ERROR"
        assert error["recoverable"] is True
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"
