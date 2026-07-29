from datetime import timedelta
from uuid import UUID

from fastapi.testclient import TestClient

from backend.main import create_app
from session.manager import SessionManager, ServerBusyError


def test_create_pending_session_returns_secure_uuid():
    manager = SessionManager(pending_ttl=timedelta(seconds=30), window_frames=75, inference_stride=25)
    created = manager.create_pending_session()
    UUID(created.session_id)
    assert created.expires_in_seconds == 30


def test_only_one_active_streaming_session_is_allowed():
    manager = SessionManager(pending_ttl=timedelta(seconds=30), window_frames=75, inference_stride=25)
    first = manager.create_pending_session()
    second = manager.create_pending_session()
    manager.activate(first.session_id)
    try:
        manager.activate(second.session_id)
    except ServerBusyError as exc:
        assert exc.code == "SERVER_BUSY"
    else:
        raise AssertionError("second active session should be rejected")


def test_disconnect_releases_active_slot():
    manager = SessionManager(pending_ttl=timedelta(seconds=30), window_frames=75, inference_stride=25)
    first = manager.create_pending_session()
    second = manager.create_pending_session()
    manager.activate(first.session_id)
    manager.disconnect(first.session_id)
    active = manager.activate(second.session_id)
    assert active.session_id == second.session_id


def test_post_sessions_returns_session_id(settings):
    app = create_app(settings)
    client = TestClient(app)
    response = client.post("/api/sessions")
    assert response.status_code == 200
    body = response.json()
    UUID(body["sessionId"])
    assert body["expiresInSeconds"] == 30
