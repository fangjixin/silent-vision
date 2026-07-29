import numpy as np
import pytest
from pydantic import ValidationError

from agent.agent import AgentPolicy
from backend.schemas import LipReadingCandidate, SemanticResult
from llm.minicpm import FakeMiniCPMInterpreter, parse_minicpm_json


def test_parse_minicpm_json_accepts_strict_object():
    parsed = parse_minicpm_json('{"language":"zh","text":"请打开灯","confidence":0.8,"reason":"中文候选更可靠"}')
    assert parsed.language == "zh"
    assert parsed.text == "请打开灯"


def test_parse_minicpm_json_rejects_markdown():
    with pytest.raises(ValidationError):
        parse_minicpm_json('```json\n{"language":"zh"}\n```')


def test_fake_minicpm_selects_highest_confidence_candidate():
    candidates = [
        LipReadingCandidate(model="avhubert", language="en", text="turn on the light", confidence=0.4, latencyMs=1),
        LipReadingCandidate(model="cmlr", language="zh", text="请打开灯", confidence=0.8, latencyMs=1),
    ]
    result = FakeMiniCPMInterpreter(threshold=0.55).interpret(
        candidates,
        [np.zeros((96, 96), dtype=np.uint8)],
        {},
    )
    assert result.language == "zh"
    assert result.text == "请打开灯"


def test_fake_minicpm_returns_unknown_for_low_confidence():
    candidates = [LipReadingCandidate(model="avhubert", language="en", text="maybe", confidence=0.2, latencyMs=1)]
    result = FakeMiniCPMInterpreter(threshold=0.55).interpret(candidates, [], {})
    assert result.language == "unknown"
    assert result.text == ""


def test_agent_policy_has_no_side_effect_actions():
    result = SemanticResult(language="en", text="turn on the light", confidence=0.75, reason="candidate accepted")
    action = AgentPolicy(threshold=0.55).decide(result)
    assert action.action == "respond"
    assert action.arguments == {}
    assert action.requiresConfirmation is False
