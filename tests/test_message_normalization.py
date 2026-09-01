from __future__ import annotations

from agents.base import _message_field
from libs.robust_json import extract_llm_content


def test_message_field_supports_sdk_and_dict_messages() -> None:
    class SdkMessage:
        content = '{"ok": true}'
        role = "assistant"

    assert _message_field(SdkMessage(), "content") == '{"ok": true}'
    assert _message_field({"content": '{"ok": true}', "role": "assistant"}, "role") == "assistant"


def test_extract_content_supports_dict_openrouter_response() -> None:
    content, finish_reason, warning = extract_llm_content(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": '{"steps": []}'},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    assert content == '{"steps": []}'
    assert finish_reason == "stop"
    assert warning is None


def test_extract_content_supports_sdk_response_with_dict_message() -> None:
    class Choice:
        message = {"content": '{"steps": []}'}
        finish_reason = "stop"

    class Response:
        choices = [Choice()]

    content, finish_reason, warning = extract_llm_content(Response())
    assert content == '{"steps": []}'
    assert finish_reason == "stop"
    assert warning is None
