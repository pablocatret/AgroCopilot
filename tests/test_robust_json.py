from __future__ import annotations

import pytest

from libs.robust_json import JsonParseError, extract_llm_content, parse_json_content


@pytest.mark.parametrize(
    ("raw", "status", "method"),
    [
        ('{"a": 1}', "exact", "direct"),
        ('```JSON\n{"a": 1}\n```', "recovered", "markdown_fence"),
        ('Resultado:\n{"a": 1}\nFin.', "recovered", "balanced_candidate"),
        ('\ufeff\x00 {"a": "llaves { dentro de texto"}', "exact", "direct"),
        ('{"result": {"a": 1}}', "recovered", "wrapper:result"),
    ],
)
def test_controlled_json_recovery(raw: str, status: str, method: str) -> None:
    result = parse_json_content(raw)
    expected = {"a": "llaves { dentro de texto"} if "llaves" in raw else {"a": 1}
    assert result.value == expected
    assert result.status == status
    assert result.method == method
    assert result.content_hash


def test_sdk_content_blocks_are_supported() -> None:
    result = parse_json_content(
        [{"type": "text", "text": "```json\n{\"ok\": true}\n```"}]
    )
    assert result.value == {"ok": True}
    assert result.status == "recovered"


def test_dict_arguments_are_not_serialized_and_reparsed() -> None:
    result = parse_json_content({"asset_url": "https://example.test/item"})
    assert result.value["asset_url"].startswith("https://")
    assert result.method == "dict"


def test_multiple_json_objects_are_ambiguous() -> None:
    with pytest.raises(JsonParseError) as exc_info:
        parse_json_content('{"a": 1}\n{"b": 2}')
    assert exc_info.value.result.status == "ambiguous"
    assert exc_info.value.result.candidate_count == 2


def test_truncated_response_is_classified() -> None:
    with pytest.raises(JsonParseError) as exc_info:
        parse_json_content('{"a": 1', finish_reason="length")
    assert exc_info.value.result.status == "truncated"


def test_risky_json5_is_not_silently_rewritten() -> None:
    with pytest.raises(JsonParseError):
        parse_json_content("{'a': 1}")


def test_extract_openai_compatible_response() -> None:
    content, finish_reason, warning = extract_llm_content(
        {"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]}
    )
    assert content == '{"ok": true}'
    assert finish_reason == "stop"
    assert warning is None
