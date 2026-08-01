from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app
from tests.conftest import make_jpeg


def test_fake_websocket_flow_reaches_agent_result(app):
    client = TestClient(app)
    session_response = client.post("/api/sessions")
    session_id = session_response.json()["sessionId"]
    with client.websocket_connect(f"/ws/{session_id}", headers={"origin": "http://localhost:8000"}) as websocket:
        ready = websocket.receive_json()
        assert ready["type"] == "session.ready"
        websocket.send_json({"type": "stream.start"})
        started = websocket.receive_json()
        assert started["type"] == "stream.started"
        for _ in range(75):
            websocket.send_bytes(make_jpeg())
        seen = set()
        for _ in range(400):
            message = websocket.receive_json()
            seen.add(message["type"])
            if message["type"] == "agent.result":
                assert message["action"] in {"respond", "confirm", "unknown"}
                break
        assert {
            "vision.result",
            "buffer.progress",
            "inference.started",
            "lip.candidates",
            "semantic.result",
            "agent.result",
        } <= seen


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
    app = create_app(Settings(model_backend="fake", allowed_origins="*"))
    client = TestClient(app)
    session_id = client.post("/api/sessions").json()["sessionId"]

    with client.websocket_connect(
        f"/ws/{session_id}",
        headers={"origin": "https://rc-b80bfa02edcceefa.radeon.firstdg.ai"},
    ) as websocket:
        assert websocket.receive_json()["type"] == "session.ready"
