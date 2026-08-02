from datetime import timedelta
from uuid import UUID

import numpy as np
from fastapi.testclient import TestClient

from backend.main import create_app
from lip.base import MouthFrame
from session.manager import SessionManager, SessionReplacedError


def test_create_pending_session_returns_secure_uuid():
    manager = SessionManager(pending_ttl=timedelta(seconds=30), window_frames=75, inference_stride=25)
    created = manager.create_pending_session()
    UUID(created.session_id)
    assert created.expires_in_seconds == 30


def test_new_active_session_takes_over_active_slot():
    manager = SessionManager(pending_ttl=timedelta(seconds=30), window_frames=75, inference_stride=25)
    first = manager.create_pending_session()
    second = manager.create_pending_session()
    old_active = manager.activate(first.session_id)
    new_active = manager.activate(second.session_id)

    assert new_active.session_id == second.session_id
    assert manager.is_current(second.session_id)
    assert not manager.is_current(first.session_id)
    try:
        manager.ensure_current(old_active)
    except SessionReplacedError as exc:
        assert exc.code == "SESSION_REPLACED"
    else:
        raise AssertionError("old active session should be marked as replaced")


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


def _frame(sequence: int) -> MouthFrame:
    return MouthFrame(sequence=sequence, received_at_ms=sequence * 40, image=np.zeros((96, 96), dtype=np.uint8))


def test_first_window_appears_at_75_valid_frames():
    manager = SessionManager(pending_ttl=timedelta(seconds=30), window_frames=75, inference_stride=25)
    created = manager.create_pending_session()
    active = manager.activate(created.session_id)
    windows = [active.add_mouth_frame(_frame(sequence)) for sequence in range(1, 76)]
    assert all(window is None for window in windows[:-1])
    assert windows[-1] is not None
    assert windows[-1].start_sequence == 1
    assert windows[-1].end_sequence == 75


def test_next_window_appears_after_25_more_valid_frames():
    manager = SessionManager(pending_ttl=timedelta(seconds=30), window_frames=75, inference_stride=25)
    created = manager.create_pending_session()
    active = manager.activate(created.session_id)
    last = None
    for sequence in range(1, 101):
        last = active.add_mouth_frame(_frame(sequence))
    assert last is not None
    assert last.start_sequence == 26
    assert last.end_sequence == 100
    assert len(last.frames) == 75


def test_stop_stream_clears_buffer_and_invalidates_existing_generation():
    manager = SessionManager(pending_ttl=timedelta(seconds=30), window_frames=40, inference_stride=10)
    created = manager.create_pending_session()
    active = manager.activate(created.session_id)
    active.reset_stream()
    generation = active.stream_generation

    for sequence in range(1, 41):
        active.add_mouth_frame(_frame(sequence))

    active.latest_pending_window = active.add_mouth_frame(_frame(41))
    active.stop_stream()

    assert active.streaming is False
    assert active.stream_generation == generation + 1
    assert len(active.frames) == 0
    assert active.accepted_frame_count == 0
    assert active.last_inference_frame_count == 0
    assert active.latest_pending_window is None
