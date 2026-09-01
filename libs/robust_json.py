"""Controlled JSON recovery for LLM and tool responses.

This module deliberately separates recovery from schema validation.  Recovery
may remove an unambiguous transport/presentation wrapper, but it never fills
missing business fields or rewrites JSON5-like syntax.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal


ParseStatus = Literal["exact", "recovered", "empty", "invalid", "ambiguous", "truncated"]


class JsonParseError(ValueError):
    """Raised when controlled JSON recovery cannot produce a valid value."""

    def __init__(self, message: str, result: "JsonParseResult[Any]") -> None:
        super().__init__(message)
        self.result = result


@dataclass
class JsonParseResult:
    value: Any = None
    status: ParseStatus = "invalid"
    method: str = ""
    warnings: list[str] = field(default_factory=list)
    preview: str = ""
    content_hash: str = ""
    candidate_count: int = 0
    expected: str = "object"
    finish_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"exact", "recovered"} and self.value is not None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "method": self.method,
            "warnings": list(self.warnings),
            "preview": self.preview,
            "content_hash": self.content_hash,
            "candidate_count": self.candidate_count,
            "expected": self.expected,
            "finish_reason": self.finish_reason,
        }


def _preview(raw: str, limit: int = 500) -> str:
    return " ".join(raw[:limit].split())


def _result(raw: str, *, expected: str, finish_reason: str | None = None, **kwargs: Any) -> JsonParseResult:
    return JsonParseResult(
        preview=_preview(raw),
        content_hash=hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest(),
        expected=expected,
        finish_reason=finish_reason,
        **kwargs,
    )


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return True


def _content_to_text(content: Any) -> tuple[str, str | None]:
    """Extract text from common OpenAI-compatible content representations."""
    if isinstance(content, str):
        return content, None
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"], None
        if isinstance(content.get("output_text"), str):
            return content["output_text"], None
        if content.get("refusal"):
            return "", "model_refusal"
        return "", "non_text_object"
    if isinstance(content, list):
        pieces: list[str] = []
        for block in content:
            if isinstance(block, str):
                pieces.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("output_text")
                if isinstance(text, str):
                    pieces.append(text)
        if pieces:
            return "\n".join(pieces), None
        return "", "empty_content_blocks"
    return "", "unsupported_content_type"


def extract_llm_content(response: Any) -> tuple[Any, str | None, str | None]:
    """Return ``(content, finish_reason, extraction_warning)`` from SDK data."""
    if isinstance(response, (str, dict, list)):
        if isinstance(response, dict) and "choices" in response:
            choices = response.get("choices") or []
            if not choices:
                return "", None, "empty_choices"
            choice = choices[0]
            message = choice.get("message", choice) if isinstance(choice, dict) else choice
            finish = choice.get("finish_reason") if isinstance(choice, dict) else None
            if isinstance(message, dict):
                return message.get("content", message.get("output_text", "")), finish, None
            return message, finish, None
        return response, None, None
    choices = getattr(response, "choices", None) or []
    if not choices:
        return "", None, "empty_choices"
    choice = choices[0]
    message = getattr(choice, "message", choice)
    finish = getattr(choice, "finish_reason", None)
    if isinstance(message, dict):
        return message.get("content", message.get("output_text", "")), finish, None
    return getattr(message, "content", None) or getattr(response, "output_text", ""), finish, None


def _fenced_text(text: str) -> tuple[str, bool]:
    lines = text.strip().splitlines()
    if not lines or not lines[0].lstrip().startswith("```"):
        return text, False
    if lines[-1].strip() != "```":
        return text, False
    language = lines[0].strip()[3:].strip().lower()
    if language not in {"", "json", "jsonc"}:
        return text, False
    return "\n".join(lines[1:-1]).strip(), True


def _balanced_json_candidates(text: str) -> list[str]:
    spans: list[str] = []
    stack: list[str] = []
    start: int | None = None
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            if not stack:
                start = index
            stack.append(char)
        elif char in "}]" and stack:
            if stack[-1] != pairs[char]:
                stack.clear()
                start = None
                continue
            stack.pop()
            if not stack and start is not None:
                spans.append(text[start:index + 1])
                start = None
    return spans


def _unwrap_known_wrapper(value: Any, allow_wrappers: tuple[str, ...], expected: str) -> tuple[Any, str]:
    if isinstance(value, dict) and len(value) == 1 and allow_wrappers:
        key, wrapped = next(iter(value.items()))
        if key in allow_wrappers and _type_matches(wrapped, expected):
            return wrapped, f"wrapper:{key}"
    return value, "direct"


def parse_json_content(
    content: Any,
    *,
    expected: str = "object",
    finish_reason: str | None = None,
    allow_wrappers: tuple[str, ...] = ("result", "evaluation", "data", "output"),
    raise_on_error: bool = True,
) -> JsonParseResult:
    """Parse a JSON response using only deterministic, controlled recovery."""
    if isinstance(content, dict) and not ("text" in content and "type" in content):
        result = _result(json.dumps(content, ensure_ascii=False), expected=expected, status="exact", method="dict", value=content)
        if _type_matches(content, expected):
            return result
        result.status = "invalid"
        result.warnings.append(f"expected_{expected}")
        if raise_on_error:
            raise JsonParseError("JSON object has an unexpected type", result)
        return result

    text, extraction_warning = _content_to_text(content)
    raw = text.replace("\ufeff", "").replace("\x00", "").strip()
    if not raw:
        status: ParseStatus = "empty"
        if finish_reason == "length":
            status = "truncated"
        result = _result(text, expected=expected, finish_reason=finish_reason, status=status, method="content")
        if extraction_warning:
            result.warnings.append(extraction_warning)
        if raise_on_error:
            raise JsonParseError("Empty or non-text JSON response", result)
        return result

    candidate_text, fenced = _fenced_text(raw)
    try:
        value = json.loads(candidate_text)
        if _type_matches(value, expected):
            value, wrapper_method = _unwrap_known_wrapper(value, allow_wrappers, expected)
            result = _result(raw, expected=expected, finish_reason=finish_reason,
                             status="recovered" if (fenced or wrapper_method != "direct") else "exact",
                             method=wrapper_method if wrapper_method != "direct" else ("markdown_fence" if fenced else "direct"),
                             value=value)
            if extraction_warning:
                result.warnings.append(extraction_warning)
            return result
    except json.JSONDecodeError:
        pass

    candidates = _balanced_json_candidates(candidate_text)
    valid: list[Any] = []
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if _type_matches(value, expected):
            valid.append(value)
    if len(valid) == 1:
        value = valid[0]
        if isinstance(value, dict) and len(value) == 1 and allow_wrappers:
            key, wrapped = next(iter(value.items()))
            if key in allow_wrappers and _type_matches(wrapped, expected):
                value = wrapped
                method = f"wrapper:{key}"
            else:
                method = "balanced_candidate"
        else:
            method = "balanced_candidate"
        result = _result(raw, expected=expected, finish_reason=finish_reason,
                         status="recovered", method=method, value=value, candidate_count=len(valid))
        if extraction_warning:
            result.warnings.append(extraction_warning)
        return result

    status = "ambiguous" if len(valid) > 1 else ("truncated" if finish_reason == "length" else "invalid")
    result = _result(raw, expected=expected, finish_reason=finish_reason, status=status,
                     method="candidate_scan", candidate_count=len(valid))
    if extraction_warning:
        result.warnings.append(extraction_warning)
    if raise_on_error:
        raise JsonParseError(f"Unable to parse JSON response ({status})", result)
    return result


def parse_json_or_raise(content: Any, **kwargs: Any) -> Any:
    """Convenience wrapper returning only the parsed value."""
    return parse_json_content(content, **kwargs).value
