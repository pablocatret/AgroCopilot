from __future__ import annotations

import json
import logging
from typing import List

logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from backend.deps import settings
from libs.attachments import summarize_table
from libs.context_engineering import summarize_memory_context
from libs.schemas import AgentInput, AgentRef, AgentRefs, SpreadsheetAgentOutput
from libs.prompts import compose_system_prompt, render_prompt

SPREADSHEET_ANALYSIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "attachment_id": {"type": "string"},
                    "filename": {"type": "string"},
                    "used_for": {"type": "string"},
                    "confidence": {"type": ["number", "null"]},
                    "key_signals": {"type": "array", "items": {"type": "string"}, "default": []},
                    "limitations": {"type": "array", "items": {"type": "string"}, "default": []},
                },
                "required": [
                    "attachment_id",
                    "filename",
                    "used_for",
                    "confidence",
                    "key_signals",
                    "limitations",
                ],
            },
            "default": [],
        }
    },
    "required": ["tables"],
}


class SpreadsheetAnalystAgent(BaseAgent):
    name = "spreadsheet_analyst"
    output_model = SpreadsheetAgentOutput
    _provider_key = "LLM_PROVIDER_SPREADSHEET_ANALYST"

    def __init__(self) -> None:
        super().__init__()
        self.model = settings.resolve_openai_model(
            "OPENAI_MODEL_SPREADSHEET_ANALYST",
            "OPENAI_MODEL_ORGANIZER",
        )

    async def _run(self, agent_input: AgentInput) -> SpreadsheetAgentOutput:
        tables = []
        refs: List[AgentRef] = []
        for attachment in agent_input.attachments:
            fname = attachment.filename.lower()
            if not fname.endswith((".csv", ".xlsx", ".xls")):
                continue
            if not attachment.storage_path:
                continue
            summary = summarize_table(attachment.storage_path)
            tables.append(
                {
                    "attachment_id": attachment.attachment_id,
                    "filename": attachment.filename,
                    "row_count": summary.row_count,
                    "columns": summary.columns,
                    "missing": summary.missing,
                    "numeric_summary": summary.numeric_summary,
                    "sample": summary.sample,
                    "used_for": "Unclassified",
                    "confidence": None,
                    "key_signals": [],
                    "limitations": [],
                }
            )
            refs.append(
                AgentRef(
                    ref_id=f"sheet-{attachment.attachment_id}",
                    title=attachment.filename,
                    source="document",
                    url=f"attachment:{attachment.attachment_id}",
                    snippet=f"Filas: {summary.row_count}. Columnas: {', '.join(summary.columns)}",
                )
            )

        if tables and self.external_enabled():
            mission = str((agent_input.context or {}).get("mission") or "").strip()
            user = (
                f"{render_prompt('spreadsheet_user.txt', query=agent_input.query, decision_mode=agent_input.decision_mode, memory_summary=summarize_memory_context(str(agent_input.context.get('user_memory', '') or '')), extracted_context=json.dumps(tables, ensure_ascii=False))}\n\n"
                + (f"Misión: {mission}\n\n" if mission else "")
                + "Devuelve SOLO JSON conforme a este esquema:\n"
                f"{json.dumps(SPREADSHEET_ANALYSIS_SCHEMA, ensure_ascii=False)}"
            )
            try:
                data = await self.call_llm_json(
                    system=compose_system_prompt(
                        agent_name="spreadsheet_analyst",
                        body=render_prompt("spreadsheet_system.txt"),
                        output_contract="Return only valid JSON following the provided schema.",
                    ),
                    user=user,
                    schema=SPREADSHEET_ANALYSIS_SCHEMA,
                    temperature=0.1,
                )
                enriched = {
                    item["attachment_id"]: item
                    for item in (data.get("tables") or [])
                    if item.get("attachment_id")
                }
                for item in tables:
                    extra = enriched.get(item["attachment_id"])
                    if not extra:
                        continue
                    item["used_for"] = extra.get("used_for", item["used_for"])
                    item["confidence"] = extra.get("confidence")
                    item["key_signals"] = extra.get("key_signals") or []
                    item["limitations"] = extra.get("limitations") or []
            except Exception as exc:
                logger.warning("spreadsheet_enrichment_error: %s", exc)

        summary = "No spreadsheets to analyze." if not tables else "Spreadsheets analyzed."
        return SpreadsheetAgentOutput(
            agent=self.name,
            summary=summary,
            refs=AgentRefs(items=refs),
            data={"tables": tables},
        )
