from __future__ import annotations

from typing import Any

import agents.case_manager as case_manager
import agents.document_analyst as document_analyst
import agents.legal as legal
import agents.organizer as organizer
import agents.spreadsheet_analyst as spreadsheet_analyst
import agents.vision_ocr as vision_ocr
import agents.writer as writer
from agents.base import openai_strict_json_schema, validate_openai_strict_json_schema


RAW_RESPONSE_SCHEMAS = [
    organizer.ORGANIZER_PLAN_SCHEMA,
    organizer.ORGANIZER_REPLAN_SCHEMA,
    case_manager.CASE_STATE_SCHEMA,
    document_analyst.DOCUMENT_ANALYSIS_SCHEMA,
    legal.LEGAL_ANSWER_SCHEMA,
    spreadsheet_analyst.SPREADSHEET_ANALYSIS_SCHEMA,
    vision_ocr.VISION_ANALYSIS_SCHEMA,
    writer.WRITER_RESPONSE_SCHEMA_BASE,
]


def _walk_schema(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_schema(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_schema(item)


def test_openai_response_schemas_are_sanitized_for_strict_structured_outputs():
    for raw_schema in RAW_RESPONSE_SCHEMAS:
        strict_schema = openai_strict_json_schema(raw_schema)

        assert validate_openai_strict_json_schema(strict_schema) == []
        for node in _walk_schema(strict_schema):
            assert "default" not in node
            assert "nullable" not in node
            if isinstance(node.get("properties"), dict):
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
