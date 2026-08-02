from backend.schemas import CommandDecision
from command.inference import reject_by_thresholds
from command.labels import CommandIntent


def test_threshold_and_margin_accept_strong_light_on():
    decision = reject_by_thresholds(
        intent=CommandIntent.LIGHT_ON,
        confidence=0.92,
        second_confidence=0.51,
        threshold=0.85,
        top1_margin=0.20,
        logits=[0.1, 4.0, 0.3, 0.2, 0.1],
    )

    assert decision.accepted is True
    assert decision.intent == CommandIntent.LIGHT_ON
    assert decision.margin == 0.41


def test_threshold_rejects_low_confidence_without_accepting_tools():
    decision = reject_by_thresholds(
        intent=CommandIntent.LIGHT_ON,
        confidence=0.60,
        second_confidence=0.52,
        threshold=0.85,
        top1_margin=0.20,
        logits=[0.1, 2.0, 1.8, 0.2, 0.1],
    )

    assert decision.accepted is False
    assert decision.intent == CommandIntent.UNKNOWN
    assert decision.reason == "below confidence threshold"


def test_command_decision_schema_serializes_logits_and_metadata():
    decision = CommandDecision(
        intent=CommandIntent.CHAT_OTHER,
        accepted=True,
        executable=False,
        confidence=0.91,
        margin=0.33,
        topK=[{"intent": "CHAT_OTHER", "confidence": 0.91}],
        logits=[0.0, 0.1, 0.2],
        reason="accepted non-executable intent",
        metadata={"frames": 75},
    )

    dumped = decision.model_dump()
    assert dumped["intent"] == "CHAT_OTHER"
    assert dumped["logits"] == [0.0, 0.1, 0.2]
    assert dumped["metadata"]["frames"] == 75
