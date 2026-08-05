from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from backend.config import Settings
from backend.schemas import CommandDecision
from command.labels import COMMAND_LABELS, EXECUTABLE_INTENTS, CommandIntent
from command.prototype import extract_roi_embedding, load_profile_prototypes, match_prototypes, sanitize_profile_id

logger = logging.getLogger(__name__)


class CommandClassifierBackend(Protocol):
    def predict(self, mouth_frames: np.ndarray, metadata: dict[str, object]) -> CommandDecision:
        raise NotImplementedError


def reject_by_thresholds(
    *,
    intent: CommandIntent,
    confidence: float,
    second_confidence: float,
    threshold: float,
    top1_margin: float,
    logits: list[float],
    top_k: list[dict[str, object]] | None = None,
    metadata: dict[str, object] | None = None,
) -> CommandDecision:
    margin = round(max(0.0, confidence - second_confidence), 6)
    executable = intent in EXECUTABLE_INTENTS
    if confidence < threshold:
        return CommandDecision(
            intent=CommandIntent.UNKNOWN,
            accepted=False,
            executable=False,
            confidence=confidence,
            margin=margin,
            topK=top_k or [],
            logits=logits,
            reason="below confidence threshold",
            metadata=metadata or {},
        )
    if margin < top1_margin:
        return CommandDecision(
            intent=CommandIntent.UNKNOWN,
            accepted=False,
            executable=False,
            confidence=confidence,
            margin=margin,
            topK=top_k or [],
            logits=logits,
            reason="top1/top2 margin too small",
            metadata=metadata or {},
        )
    if intent == CommandIntent.UNKNOWN:
        return CommandDecision(
            intent=CommandIntent.UNKNOWN,
            accepted=False,
            executable=False,
            confidence=confidence,
            margin=margin,
            topK=top_k or [],
            logits=logits,
            reason="classifier predicted UNKNOWN",
            metadata=metadata or {},
        )
    return CommandDecision(
        intent=intent,
        accepted=True,
        executable=executable,
        confidence=confidence,
        margin=margin,
        topK=top_k or [],
        logits=logits,
        reason="accepted executable intent" if executable else "accepted non-executable intent",
        metadata=metadata or {},
    )


class FakeCommandClassifierBackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def predict(self, mouth_frames: np.ndarray, metadata: dict[str, object]) -> CommandDecision:
        motion = float(np.abs(np.diff(mouth_frames.astype("float32"), axis=0)).mean()) if len(mouth_frames) > 1 else 0.0
        if motion > 1.0:
            intent = CommandIntent.LIGHT_ON
            confidence = 0.91
            second = 0.42
        else:
            intent = CommandIntent.UNKNOWN
            confidence = 0.45
            second = 0.40
        logits = [0.0 for _ in COMMAND_LABELS]
        logits[COMMAND_LABELS.index(intent)] = confidence
        top_k = [{"intent": intent.value, "confidence": confidence}]
        return reject_by_thresholds(
            intent=intent,
            confidence=confidence,
            second_confidence=second,
            threshold=self.settings.command_confidence_threshold,
            top1_margin=self.settings.command_top1_margin,
            logits=logits,
            top_k=top_k,
            metadata=metadata,
        )


class PrototypeCommandClassifierBackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.persistence_root

    def predict(self, mouth_frames: np.ndarray, metadata: dict[str, object]) -> CommandDecision:
        started = perf_counter()
        embedding = extract_roi_embedding(mouth_frames, feature_dim=self.settings.prototype_feature_dim)
        profile_id = "global"
        scopes: list[tuple[str, str | None]] = [("global", "global")]

        last_match = None
        last_scope = "none"
        for scope, candidate_profile in scopes:
            if candidate_profile is None:
                continue
            try:
                sanitized_profile = sanitize_profile_id(candidate_profile)
            except ValueError:
                continue
            samples = load_profile_prototypes(self.root, sanitized_profile)
            if not samples:
                continue
            match = match_prototypes(
                embedding,
                samples,
                confidence_threshold=self.settings.prototype_confidence_threshold,
                margin_threshold=self.settings.prototype_top1_margin,
            )
            last_match = match
            last_scope = scope
            if match.accepted:
                break

        if last_match is None:
            match = match_prototypes(
                embedding,
                [],
                confidence_threshold=self.settings.prototype_confidence_threshold,
                margin_threshold=self.settings.prototype_top1_margin,
            )
            last_match = match
            last_scope = "none"

        intent = CommandIntent(last_match.intent)
        executable = last_match.accepted and intent in EXECUTABLE_INTENTS
        matched_metadata = dict(last_match.matched_metadata)
        matched_phrase = _clean_metadata_string(matched_metadata.get("phrase"))
        matched_language = _clean_language(matched_metadata.get("language"))
        result_metadata = {
            **metadata,
            "backend": "prototype",
            "profileId": profile_id,
            "profileScope": last_scope,
            "latencyMs": int((perf_counter() - started) * 1000),
        }
        if last_match.accepted:
            result_metadata["displayText"] = matched_phrase or intent.value
            if matched_phrase:
                result_metadata["matchedPhrase"] = matched_phrase
            if matched_language:
                result_metadata["matchedLanguage"] = matched_language
                result_metadata["language"] = matched_language
            if last_match.matched_sample_path:
                result_metadata["matchedSamplePath"] = last_match.matched_sample_path
        return CommandDecision(
            intent=intent,
            accepted=last_match.accepted,
            executable=executable,
            confidence=last_match.confidence,
            margin=last_match.margin,
            topK=last_match.top_k,
            logits=last_match.logits,
            reason=last_match.reason,
            metadata=result_metadata,
        )


def _clean_metadata_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _clean_language(value: object) -> str | None:
    language = _clean_metadata_string(value)
    return language if language in {"zh", "en"} else None


def save_command_debug(settings: Settings, session_id: str, decision: CommandDecision) -> Path:
    output_dir = settings.debug_window_dir or settings.persistence_root / "logs" / "command-runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{session_id}-command-metadata.json"
    path.write_text(json.dumps(decision.model_dump(), ensure_ascii=False, indent=2))
    return path


def build_command_classifier(settings: Settings) -> CommandClassifierBackend:
    if settings.command_backend == "prototype":
        return PrototypeCommandClassifierBackend(settings)
    return FakeCommandClassifierBackend(settings)
