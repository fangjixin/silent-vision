from pathlib import Path

import pytest

from command.catalog import PhraseCatalog, load_phrase_catalog, normalize_phrase


def test_initial_catalog_has_exact_registered_text_and_intents():
    catalog = load_phrase_catalog(Path("command/phrase_catalog.json"))
    assert [(e.phrase_id, e.text, e.intent.value) for e in catalog.entries] == [
        ("zh_light_on_hello", "你好，请帮我打开灯", "LIGHT_ON"),
        ("zh_chat_meal", "你吃饭了吗？", "CHAT_OTHER"),
    ]


def test_normalization_preserves_punctuation():
    assert normalize_phrase("  Ｈello   世界？ ") == "hello 世界?"


@pytest.mark.parametrize(
    "records, message",
    [
        ([{"phraseId": "x", "text": "a", "language": "zh", "intent": "LIGHT_ON", "enabled": True},
          {"phraseId": "x", "text": "b", "language": "zh", "intent": "LIGHT_OFF", "enabled": True}], "duplicate phraseId"),
        ([{"phraseId": "x", "text": "a", "language": "zh", "intent": "NOT_REAL", "enabled": True}], "unknown intent"),
    ],
)
def test_invalid_catalog_is_rejected(records, message):
    with pytest.raises(ValueError, match=message):
        PhraseCatalog.from_records(records)
