import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app
from backend.schemas import CommandDecision
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
    monkeypatch.setattr(
        "api.websocket.extract_mouth_roi_clip", lambda **kwargs: FakeRoiClip()
    )


class StubClassifier:
    def __init__(self, decision: CommandDecision):
        self.decision = decision

    def predict(self, mouth_frames, metadata):
        return self.decision.model_copy(
            update={"metadata": self.decision.metadata | metadata}
        )


def collect_clip_messages(app):
    client = TestClient(app)
    session_id = client.post("/api/sessions").json()["sessionId"]
    messages = {}
    with client.websocket_connect(
        f"/ws/{session_id}", headers={"origin": "http://localhost:8000"}
    ) as websocket:
        assert websocket.receive_json()["type"] == "session.ready"
        websocket.send_json({"type": "clip.start", "profileId": "global"})
        assert websocket.receive_json()["type"] == "clip.started"
        websocket.send_bytes(b"fake-webm")
        for _ in range(20):
            message = websocket.receive_json()
            messages[message["type"]] = message
            if message["type"] == "agent.result":
                break
    return messages


def send_stubbed_clip(app, monkeypatch, decision):
    patch_clip_pipeline(monkeypatch)
    app.state.command_classifier = StubClassifier(decision)
    return collect_clip_messages(app)


def test_accepted_light_phrase_executes_with_exact_display_text(app, monkeypatch):
    messages = send_stubbed_clip(
        app,
        monkeypatch,
        CommandDecision(
            intent="LIGHT_ON",
            accepted=True,
            executable=True,
            confidence=0.98,
            margin=0.90,
            topK=[],
            logits=[],
            reason="accepted executable intent",
            metadata={
                "phraseId": "zh_light_on_hello",
                "matchedPhrase": "你好，请帮我打开灯",
                "displayText": "你好，请帮我打开灯",
                "language": "zh",
            },
        ),
    )

    assert messages["command.result"]["metadata"]["displayText"] == "你好，请帮我打开灯"
    assert messages["agent.result"]["action"] == "execute"
    assert messages["agent.result"]["text"] == "你好，请帮我打开灯"


def test_rejected_unknown_phrase_never_executes(app, monkeypatch):
    messages = send_stubbed_clip(
        app,
        monkeypatch,
        CommandDecision(
            intent="UNKNOWN",
            accepted=False,
            executable=False,
            confidence=0.51,
            margin=0.02,
            topK=[],
            logits=[],
            reason="low_probability",
            metadata={"rejectionReason": "low_probability"},
        ),
    )

    assert messages["command.result"]["accepted"] is False
    assert messages["agent.result"]["action"] == "reject"
    assert messages["agent.result"]["action"] != "execute"


def test_accepted_chat_phrase_is_displayed_but_never_executes(app, monkeypatch):
    messages = send_stubbed_clip(
        app,
        monkeypatch,
        CommandDecision(
            intent="CHAT_OTHER",
            accepted=True,
            executable=False,
            confidence=0.97,
            margin=0.88,
            topK=[],
            logits=[],
            reason="accepted non-executable intent",
            metadata={
                "phraseId": "zh_chat_meal",
                "matchedPhrase": "你吃饭了吗？",
                "displayText": "你吃饭了吗？",
                "language": "zh",
            },
        ),
    )

    assert messages["command.result"]["accepted"] is True
    assert messages["agent.result"]["action"] == "ignore"
    assert messages["agent.result"]["text"] == "你吃饭了吗？"


def test_real_torch_websocket_accepts_catalog_text_and_rejects_without_execution(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    if (
        getattr(torch.version, "hip", None) is None
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 1
    ):
        pytest.skip("requires ROCm PyTorch and cuda:0; run on the Radeon host")

    from command.checkpoint import save_phrase_checkpoint
    from command.model import build_fixed_phrase_model

    model = build_fixed_phrase_model(2).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias.copy_(torch.tensor([10.0, -10.0]))
    checkpoint = tmp_path / "synthetic-fixed-phrase.pt"
    save_phrase_checkpoint(
        checkpoint,
        {
            "schemaVersion": "silent-vision.fixed-phrase.v2",
            "modelState": model.state_dict(),
            "phraseIds": ["zh_light_on_hello", "zh_chat_meal"],
            "phraseCatalog": [
                {
                    "phraseId": "zh_light_on_hello",
                    "text": "你好，请帮我打开灯",
                    "language": "zh",
                    "intent": "LIGHT_ON",
                    "enabled": True,
                },
                {
                    "phraseId": "zh_chat_meal",
                    "text": "你吃饭了吗？",
                    "language": "zh",
                    "intent": "CHAT_OTHER",
                    "enabled": True,
                },
            ],
            "featureConfig": {
                "fps": 25,
                "height": 96,
                "width": 96,
                "downsample": 16,
            },
            "modelConfig": {"embeddingDim": 64, "parameterCap": 150000},
            "decisionPolicy": {
                "languageSelectionRequired": True,
                "probabilityNormalization": "selected-language-softmax",
            },
            "decisionThresholds": {
                "minProbability": 0.80,
                "maxCosineDistance": {
                    "zh_light_on_hello": 1.10,
                    "zh_chat_meal": 1.10,
                },
            },
            "classCentroids": torch.zeros((2, 64), dtype=torch.float32),
            "trainingSummary": {"seed": 17, "evidentiary": False},
        },
    )
    patch_clip_pipeline(monkeypatch)

    accepted_app = create_app(
        Settings(
            command_backend="torch",
            command_classifier_checkpoint=checkpoint,
        )
    )
    accepted = collect_clip_messages(accepted_app)
    accepted_command = accepted["command.result"]
    assert accepted_command["metadata"]["backend"] == "torch"
    assert accepted_command["metadata"]["phraseId"] == "zh_light_on_hello"
    assert accepted_command["metadata"]["displayText"] == "你好，请帮我打开灯"
    assert accepted["agent.result"]["action"] == "execute"
    assert accepted["agent.result"]["text"] == "你好，请帮我打开灯"

    rejected_app = create_app(
        Settings(
            command_backend="torch",
            command_classifier_checkpoint=checkpoint,
            command_phrase_distance_override=0.0,
        )
    )
    rejected = collect_clip_messages(rejected_app)
    rejected_command = rejected["command.result"]
    assert rejected_command["intent"] == "UNKNOWN"
    assert rejected_command["accepted"] is False
    assert rejected_command["executable"] is False
    assert "phraseId" not in rejected_command["metadata"]
    assert "displayText" not in rejected_command["metadata"]
    assert rejected["agent.result"]["action"] == "reject"


def test_fake_websocket_flow_reaches_agent_result(app, monkeypatch):
    patch_clip_pipeline(monkeypatch)
    client = TestClient(app)
    session_response = client.post("/api/sessions")
    session_id = session_response.json()["sessionId"]
    with client.websocket_connect(
        f"/ws/{session_id}", headers={"origin": "http://localhost:8000"}
    ) as websocket:
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
    with client.websocket_connect(
        f"/ws/{first_id}", headers={"origin": "http://localhost:8000"}
    ) as first:
        assert first.receive_json()["type"] == "session.ready"
        with client.websocket_connect(
            f"/ws/{second_id}", headers={"origin": "http://localhost:8000"}
        ) as second:
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

    with client.websocket_connect(
        f"/ws/{session_id}", headers={"origin": "http://localhost:8000"}
    ) as websocket:
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
