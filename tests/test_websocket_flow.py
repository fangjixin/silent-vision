from fastapi.testclient import TestClient

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


def test_second_active_websocket_gets_server_busy(app):
    client = TestClient(app)
    first_id = client.post("/api/sessions").json()["sessionId"]
    second_id = client.post("/api/sessions").json()["sessionId"]
    with client.websocket_connect(f"/ws/{first_id}", headers={"origin": "http://localhost:8000"}) as first:
        assert first.receive_json()["type"] == "session.ready"
        with client.websocket_connect(f"/ws/{second_id}", headers={"origin": "http://localhost:8000"}) as second:
            error = second.receive_json()
            assert error["type"] == "error"
            assert error["code"] == "SERVER_BUSY"
