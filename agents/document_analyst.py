from __future__ import annotations

import json
from typing import List

from agents.base import BaseAgent
from backend.deps import settings
from libs.attachments import extract_artifact_from_document
from libs.context_engineering import summarize_memory_context
from libs.prompts import compose_system_prompt, render_prompt
from libs.schemas import AgentInput, AgentRef, AgentRefs, DocumentAgentOutput

DOCUMENT_ANALYSIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "documents": {
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
    "required": ["documents"],
}


class DocumentAnalystAgent(BaseAgent):
    name = "document_analyst"
    output_model = DocumentAgentOutput
    _provider_key = "LLM_PROVIDER_DOCUMENT_ANALYST"

    def __init__(self) -> None:
        super().__init__()
        self.model = settings.resolve_openai_model(
            "OPENAI_MODEL_DOCUMENT_ANALYST",
            "OPENAI_MODEL_ORGANIZER",
        )

    async def _run(self, agent_input: AgentInput) -> DocumentAgentOutput:
        docs = []
        refs: List[AgentRef] = []
        for attachment in agent_input.attachments:
            fname = attachment.filename.lower()
            if not fname.endswith((".pdf", ".doc", ".docx", ".txt", ".html", ".htm")):
                continue
            if not attachment.storage_path:
                continue
            extraction_meta = (
                attachment.metadata.get("extraction", {})
                if isinstance(attachment.metadata.get("extraction"), dict)
                else {}
            )
            confidence = extraction_meta.get("confidence")
            limitations = [
                str(item).strip()
                for item in (extraction_meta.get("warnings") or [])
                if str(item).strip()
            ]
            text = (attachment.extracted_text or attachment.summary or "").strip()
            if not text:
                artifact = extract_artifact_from_document(attachment.storage_path)
                text = artifact.text
                confidence = artifact.confidence
                limitations = [item for item in artifact.warnings if item]
            excerpt = (text[:400] + "...") if len(text) > 400 else text
            summary = excerpt or "Documento sin texto extraible."
            docs.append(
                {
                    "attachment_id": attachment.attachment_id,
                    "filename": attachment.filename,
                    "word_count": len(text.split()),
                    "summary": summary,
                    "excerpt": excerpt,
                    "used_for": "Aun sin clasificar",
                    "confidence": confidence,
                    "key_signals": [],
                    "limitations": limitations,
                }
            )
            refs.append(
                AgentRef(
                    ref_id=f"doc-{attachment.attachment_id}",
                    title=attachment.filename,
                    source="document",
                    url=f"attachment:{attachment.attachment_id}",
                    snippet=excerpt,
                )
            )

        if docs and self.external_enabled():
            mission = str((agent_input.context or {}).get("mission") or "").strip()
            user = (
                f"{render_prompt('document_user.txt', query=agent_input.query, decision_mode=agent_input.decision_mode, memory_summary=summarize_memory_context(str(agent_input.context.get('user_memory', '') or '')), extracted_context=json.dumps(docs, ensure_ascii=False))}\n\n"
                + (f"Misión: {mission}\n\n" if mission else "")
                + "Devuelve SOLO JSON conforme a este esquema:\n"
                f"{json.dumps(DOCUMENT_ANALYSIS_SCHEMA, ensure_ascii=False)}"
            )
            try:
                data = await self.call_llm_json(
                    system=compose_system_prompt(
                        agent_name="document_analyst",
                        body=render_prompt("document_system.txt"),
                        output_contract="Devuelve exclusivamente JSON valido siguiendo el esquema proporcionado.",
                    ),
                    user=user,
                    schema=DOCUMENT_ANALYSIS_SCHEMA,
                    temperature=0.1,
                )
                enriched = {
                    item["attachment_id"]: item
                    for item in (data.get("documents") or [])
                    if item.get("attachment_id")
                }
                for item in docs:
                    extra = enriched.get(item["attachment_id"])
                    if not extra:
                        continue
                    item["used_for"] = extra.get("used_for", item["used_for"])
                    item["confidence"] = extra.get("confidence", item["confidence"])
                    item["key_signals"] = extra.get("key_signals") or []
                    item["limitations"] = extra.get("limitations") or item["limitations"]
            except Exception:
                for item in docs:
                    limitations = list(item.get("limitations") or [])
                    message = "Could not enrich document with LLM analysis."
                    if message not in limitations:
                        limitations.append(message)
                    item["limitations"] = limitations

        summary = "No documents to analyze." if not docs else "Documents analyzed."
        return DocumentAgentOutput(
            agent=self.name,
            summary=summary,
            refs=AgentRefs(items=refs),
            data={"documents": docs},
        )
