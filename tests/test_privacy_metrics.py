from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import make_jpeg


def test_disconnect_removes_session_temp_dir(app):
    client = TestClient(app)
    session_id = client.post("/api/sessions").json()["sessionId"]
    temp_dir = Path("/tmp/silent-vision") / session_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "frame.jpg").write_bytes(b"not persisted")
    with client.websocket_connect(f"/ws/{session_id}", headers={"origin": "http://localhost:8000"}) as websocket:
        assert websocket.receive_json()["type"] == "session.ready"
    assert not temp_dir.exists()


def test_metrics_update_is_emitted(app):
    client = TestClient(app)
    session_id = client.post("/api/sessions").json()["sessionId"]
    with client.websocket_connect(f"/ws/{session_id}", headers={"origin": "http://localhost:8000"}) as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "stream.start"})
        websocket.receive_json()
        websocket.send_bytes(make_jpeg())
        seen = []
        for _ in range(20):
            message = websocket.receive_json()
            seen.append(message["type"])
            if message["type"] == "metrics.update":
                assert "receivedFps" in message
                assert "validFrameRatio" in message
                break
        assert "metrics.update" in seen
