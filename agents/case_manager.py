from __future__ import annotations

import json

from agents.base import BaseAgent
from backend.deps import settings
from libs.context_engineering import summarize_case_history, summarize_execution_report, summarize_memory_context
from libs.prompts import compose_system_prompt, render_prompt
from libs.schemas import (
    AgentInput,
    CaseEvidenceItem,
    CaseEvidenceLedger,
    CaseEvidenceModalitySummary,
    CaseManagerAgentOutput,
    CaseState,
    CaseStateDraft,
    CaseTask,
)

CASE_STATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "case_summary": {"type": "string", "minLength": 1},
        "open_tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "priority": {"type": "string", "enum": ["alta", "media", "baja"]},
                    "status": {"type": "string", "enum": ["abierta", "bloqueada", "hecha"]},
                    "rationale": {"type": "string"},
                    "source": {
                        "type": "string",
                        "enum": [
                            "remote_sensing",
                            "document",
                            "legal",
                            "general",
                        ],
                    },
                },
                "required": ["title", "priority", "status", "rationale"],
            },
            "default": [],
        },
        "blocked_by": {"type": "array", "items": {"type": "string"}, "default": []},
        "recommended_next_input": {"type": "array", "items": {"type": "string"}, "default": []},
    },
    "required": ["case_summary", "open_tasks", "blocked_by", "recommended_next_input"],
}


def _dedupe(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.append(text)
    return seen


class CaseManagerAgent(BaseAgent):
    name = "case_manager"
    output_model = CaseManagerAgentOutput
    _provider_key = "LLM_PROVIDER_CASE_MANAGER"

    def __init__(self) -> None:
        super().__init__()
        self.model = settings.resolve_openai_model(
            "OPENAI_MODEL_CASE_MANAGER",
            "OPENAI_MODEL_WRITER",
        )

    @staticmethod
    def _priority_rank(value: str) -> int:
        return {"baja": 0, "media": 1, "alta": 2}.get(str(value or "").lower(), 1)

    @staticmethod
    def _status_rank(value: str) -> int:
        return {"hecha": 0, "abierta": 1, "bloqueada": 2}.get(str(value or "").lower(), 1)

    @staticmethod
    def _compact_text(value: str, *, limit: int = 400) -> str:
        return " ".join(str(value or "").split()).strip()[:limit]

    @classmethod
    def _is_substantive_text(cls, value: str) -> bool:
        return len(cls._compact_text(value, limit=600)) >= 18

    def _merge_task(self, base: CaseTask, extra: CaseTask) -> CaseTask:
        chosen_priority = (
            extra.priority
            if self._priority_rank(extra.priority) >= self._priority_rank(base.priority)
            else base.priority
        )
        chosen_status = (
            extra.status
            if self._status_rank(extra.status) >= self._status_rank(base.status)
            else base.status
        )
        base_rationale = self._compact_text(base.rationale)
        extra_rationale = self._compact_text(extra.rationale)
        rationale = extra_rationale if len(extra_rationale) >= len(base_rationale) else base_rationale
        return CaseTask(
            title=self._compact_text(extra.title or base.title, limit=140),
            priority=chosen_priority,
            status=chosen_status,
            rationale=rationale,
            source=extra.source or base.source,
        )

    def _merge_tasks(self, draft_tasks: list[CaseTask], llm_tasks: list[CaseTask]) -> list[CaseTask]:
        merged: dict[str, CaseTask] = {}
        for task in draft_tasks + llm_tasks:
            title = self._compact_text(task.title, limit=140)
            if not title:
                continue
            normalized = CaseTask(
                title=title,
                priority=task.priority,
                status=task.status,
                rationale=self._compact_text(task.rationale),
                source=task.source,
            )
            key = title.lower()
            current = merged.get(key)
            merged[key] = normalized if current is None else self._merge_task(current, normalized)
        return list(merged.values())[:6]

    def _merge_string_lists(
        self,
        draft_values: list[str],
        llm_values: list[str],
        *,
        limit: int,
    ) -> list[str]:
        merged: list[str] = []
        for item in list(draft_values) + list(llm_values):
            text = self._compact_text(item, limit=280)
            if text and text not in merged:
                merged.append(text)
        return merged[:limit]

    def _ledger_modalities(self, items: list[CaseEvidenceItem]) -> list[CaseEvidenceModalitySummary]:
        grouped: dict[str, list[CaseEvidenceItem]] = {}
        for item in items:
            grouped.setdefault(item.source, []).append(item)

        summaries: list[CaseEvidenceModalitySummary] = []
        for source, source_items in grouped.items():
            confidence_values = [
                item.confidence for item in source_items if isinstance(item.confidence, (int, float))
            ]
            summaries.append(
                CaseEvidenceModalitySummary(
                    source=source,
                    title=source_items[0].title,
                    usable_items=sum(1 for item in source_items if item.status == "usable"),
                    partial_items=sum(1 for item in source_items if item.status == "partial"),
                    failed_items=sum(1 for item in source_items if item.status == "failed"),
                    missing_items=sum(1 for item in source_items if item.status == "missing"),
                    confidence=(
                        round(sum(confidence_values) / len(confidence_values), 3)
                        if confidence_values
                        else None
                    ),
                    key_signals=[
                        self._compact_text(item.summary, limit=180)
                        for item in source_items
                        if self._is_substantive_text(item.summary)
                    ][:3],
                    limitations=[
                        self._compact_text(str(limit_text), limit=180)
                        for item in source_items
                        for limit_text in (item.metadata.get("limitations") or [])
                        if self._compact_text(str(limit_text), limit=180)
                    ][:4],
                    metadata={"items": len(source_items)},
                )
            )
        return summaries

    @staticmethod
    def _rs_prompt_context(rs_output: Any) -> str:
        rs_data = getattr(rs_output, "data", None)
        if not rs_data:
            return "Sin contexto RS ampliado."
        lines: list[str] = []
        overview = str(getattr(rs_data, "overview", "") or "").strip()
        if overview:
            lines.append(f"Vision general: {overview}")
        changes = list(getattr(rs_data, "temporal_changes", []) or [])
        if changes:
            primary = sorted(
                changes,
                key=lambda change: (
                    0 if getattr(change, "reliable", False) else 1,
                    {"alta": 0, "media": 1, "baja": 2}.get(getattr(change, "severity", "media"), 1),
                ),
            )[0]
            lines.append(f"Cambio principal: {primary.label}. {primary.detail}")
            if getattr(primary, "limitations", None):
                lines.append(f"Cautela RS: {primary.limitations[0]}")
        focus_areas = list(getattr(rs_data, "focus_areas", []) or [])
        if focus_areas:
            lines.append(f"Validacion sugerida: {focus_areas[0].title}. {focus_areas[0].detail}")
        return "\n".join(lines[:4]) if lines else "Sin contexto RS ampliado."

    def _build_evidence_ledger(self, agent_input: AgentInput) -> CaseEvidenceLedger:
        context = agent_input.context or {}
        items: list[CaseEvidenceItem] = []

        legal_output = context.get("legal")
        if legal_output:
            legal_data = getattr(legal_output, "data", None)
            items.append(
                CaseEvidenceItem(
                    source="legal",
                    title="Normativa y criterios legales",
                    summary=getattr(legal_output, "summary", "") or "Sin resumen legal.",
                    confidence=0.7 if getattr(legal_data, "checklist", None) else 0.45,
                    status="usable" if getattr(legal_output, "status", "ok") == "ok" else "failed",
                    metadata={
                        "limitations": list(getattr(legal_data, "limitations", []) or []),
                        "checklist_items": len(getattr(legal_data, "checklist", []) or []),
                    },
                )
            )

        for key, source_name, label in (
            ("document_analyst", "document", "Adjuntos documentales"),
            ("spreadsheet_analyst", "spreadsheet", "Tablas y hojas de cálculo"),
            ("vision_ocr", "vision", "OCR e imágenes"),
        ):
            output = context.get(key)
            if not output:
                continue
            payload = getattr(output, "data", {}) or {}
            entries = []
            if isinstance(payload, dict):
                entries = payload.get("documents") or payload.get("tables") or payload.get("images") or []
            usable_entries = sum(1 for item in entries if item)
            limitations = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                for limit_text in entry.get("limitations") or []:
                    text = self._compact_text(limit_text, limit=180)
                    if text and text not in limitations:
                        limitations.append(text)
            items.append(
                CaseEvidenceItem(
                    source=source_name,
                    title=label,
                    summary=getattr(output, "summary", "") or f"Sin resumen en {label.lower()}.",
                    confidence=0.7 if usable_entries else 0.35,
                    status="usable" if usable_entries else "partial",
                    metadata={"items": usable_entries, "limitations": limitations[:4]},
                )
            )

        rs_output = context.get("rs_analyst")
        if rs_output:
            rs_data = getattr(rs_output, "data", None)
            changes = getattr(rs_data, "temporal_changes", []) or []
            items.append(
                CaseEvidenceItem(
                    source="remote_sensing",
                    title="Seguimiento satelital",
                    summary=getattr(rs_output, "summary", "") or "Sin resumen satelital.",
                    confidence=0.75 if changes else 0.5,
                    status="usable" if changes else "partial",
                    metadata={
                        "temporal_changes": len(changes),
                        "limitations": [
                            self._compact_text(limit_text, limit=180)
                            for change in changes[:3]
                            for limit_text in getattr(change, "limitations", [])[:2]
                            if self._compact_text(limit_text, limit=180)
                        ][:4],
                    },
                )
            )

        memory_text = str(context.get("user_memory", "") or "").strip()
        if memory_text:
            relevant = memory_text[:600]
            for header in ("## Contexto agronomico", "## Preguntas abiertas"):
                idx = memory_text.find(header)
                if idx >= 0:
                    relevant = memory_text[idx:idx + 600]
                    break
            items.append(
                CaseEvidenceItem(
                    source="memory",
                    title="Memoria de usuario",
                    summary=relevant,
                    confidence=0.65,
                    status="usable",
                    metadata={"limitations": []},
                )
            )

        if not items:
            items.append(
                CaseEvidenceItem(
                    source="general",
                    title="Caso sin evidencias auxiliares",
                    summary="No hay evidencias auxiliares estructuradas; solo está disponible la consulta actual.",
                    confidence=0.2,
                    status="missing",
                    metadata={"limitations": []},
                )
            )
        return CaseEvidenceLedger(items=items, modalities=self._ledger_modalities(items))

    def _sanitize_case_state(self, case_state: CaseState, ledger: CaseEvidenceLedger) -> CaseState:
        open_tasks = []
        seen_titles: set[str] = set()
        for task in case_state.open_tasks:
            title = " ".join(str(task.title).split()).strip()
            if not title:
                continue
            key = title.lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            open_tasks.append(
                CaseTask(
                    title=title[:140],
                    priority=task.priority,
                    status=task.status,
                    rationale=" ".join(str(task.rationale).split()).strip()[:400],
                    source=task.source,
                )
            )

        blocked_by = [
            item
            for item in _dedupe(case_state.blocked_by)
            if "faltan adjuntos" not in item.lower()
            and "falta evidencia normativa" not in item.lower()
        ][:6]
        if any(
            modality.usable_items > 0
            for modality in ledger.modalities
            if modality.source in {"document", "vision", "spreadsheet"}
        ):
            blocked_by = [
                item
                for item in blocked_by
                if "no hay evidencia documental" not in item.lower()
                and "sin documentos" not in item.lower()
            ]
        recommended_next_input = _dedupe(case_state.recommended_next_input)[:6]
        summary = " ".join(str(case_state.case_summary).split()).strip() or "Caso en seguimiento."
        return CaseState(
            case_summary=summary[:500],
            open_tasks=open_tasks[:6],
            blocked_by=blocked_by,
            recommended_next_input=recommended_next_input,
            evidence_ledger=ledger,
        )

    def _deterministic_case_state_draft(self, agent_input: AgentInput) -> CaseState:
        return self._fallback(agent_input)

    def _merge_case_state_with_draft(
        self,
        draft: CaseState,
        llm_state: CaseState,
        *,
        ledger: CaseEvidenceLedger,
    ) -> CaseState:
        case_summary = (
            llm_state.case_summary
            if self._is_substantive_text(llm_state.case_summary)
            else draft.case_summary
        ) or draft.case_summary or llm_state.case_summary
        merged = CaseState(
            case_summary=case_summary,
            open_tasks=self._merge_tasks(draft.open_tasks, llm_state.open_tasks),
            blocked_by=self._merge_string_lists(draft.blocked_by, llm_state.blocked_by, limit=6),
            recommended_next_input=self._merge_string_lists(
                draft.recommended_next_input,
                llm_state.recommended_next_input,
                limit=6,
            ),
            evidence_ledger=ledger,
        )
        return self._sanitize_case_state(merged, ledger)

    def _fallback(self, agent_input: AgentInput) -> CaseState:
        legal_output = agent_input.context.get("legal")
        legal = getattr(legal_output, "summary", "") or ""
        legal_data = getattr(legal_output, "data", None)
        checklist = getattr(legal_data, "checklist", []) or [] if legal_data else []
        document_summary = getattr(agent_input.context.get("document_analyst"), "summary", "") or ""
        spreadsheet_summary = getattr(agent_input.context.get("spreadsheet_analyst"), "summary", "") or ""
        vision_summary = getattr(agent_input.context.get("vision_ocr"), "summary", "") or ""
        rs_output = agent_input.context.get("rs_analyst")
        rs_data = getattr(getattr(rs_output, "data", None), "temporal_changes", []) or [] if rs_output else []
        rs_focus = getattr(getattr(rs_output, "data", None), "focus_areas", []) or [] if rs_output else []
        rs_summary = getattr(rs_output, "summary", "") or "" if rs_output else ""
        query = getattr(agent_input, "query", "") or ""

        actionable_changes = [
            change
            for change in rs_data
            if getattr(change, "delta_mean", None) is not None
            and (getattr(change, "reliable", False) or getattr(change, "confidence", 0) >= 0.65)
        ]

        summary_parts: list[str] = []
        if rs_summary:
            summary_parts.append(rs_summary)
        if legal:
            summary_parts.append(legal)
        document_context = document_summary or spreadsheet_summary or vision_summary
        if document_context:
            summary_parts.append(document_context)
        summary = " ".join(summary_parts)[:500] if summary_parts else "Case in progress with incomplete information."

        tasks: list[CaseTask] = []
        for change in actionable_changes[:2]:
            label = getattr(change, "label", "change")
            detail = getattr(change, "detail", "")
            parcel = getattr(change, "from_item_id", "") or ""
            tasks.append(
                CaseTask(
                    title=f"Field-check: confirm {label.lower()} detected by satellite",
                    priority="alta",
                    status="abierta",
                    rationale=f"Satellite detected {label.lower()}. {detail}".strip()[:400],
                    source="remote_sensing",
                )
            )
        for area in rs_focus[:2]:
            title_str = getattr(area, "title", "") or "area"
            parcel = getattr(area, "parcel", None) or "target parcel"
            detail = getattr(area, "detail", "")
            tasks.append(
                CaseTask(
                    title=f"Review {title_str.lower()} in {parcel}",
                    priority="media",
                    status="abierta",
                    rationale=detail[:400] if detail else f"Focus area requires review in {parcel}.",
                    source="remote_sensing",
                )
            )
        pending_legal = [f for f in checklist if getattr(f, "status", "") != "cumple"]
        if pending_legal:
            req = getattr(pending_legal[0], "requirement", "regulatory requirement")
            tasks.append(
                CaseTask(
                    title=f"Verify compliance: {req[:100]}",
                    priority="alta",
                    status="abierta",
                    rationale=f"Regulatory requirement not yet confirmed as met. {legal[:200]}".strip()[:400],
                    source="legal",
                )
            )
        if document_context and not tasks:
            tasks.append(
                CaseTask(
                    title="Review attached documents and cross-reference with findings",
                    priority="alta",
                    status="abierta",
                    rationale=document_context[:400],
                    source="document",
                )
            )
        tasks = tasks[:5]

        blocked_by: list[str] = []
        if not legal and not actionable_changes:
            blocked_by.append(
                "No regulatory evidence or technical signal available to make a high-confidence decision."
            )
        if pending_legal:
            first_req = getattr(pending_legal[0], "requirement", "regulatory requirement")
            blocked_by.append(
                f"Regulatory requirement not verified: {first_req[:150]}."
            )

        recommended_next_input: list[str] = []
        if not document_context:
            recommended_next_input.append(
                "Upload relevant documents, field photos, or data tables to strengthen the decision."
            )
        for change in actionable_changes[:1]:
            label = getattr(change, "label", "change").lower()
            recommended_next_input.append(
                f"Field photo this week to confirm: {label}"
            )

        ledger = self._build_evidence_ledger(agent_input)
        return self._sanitize_case_state(
            CaseState(
                case_summary=summary,
                open_tasks=tasks,
                blocked_by=_dedupe(blocked_by)[:6],
                recommended_next_input=_dedupe(recommended_next_input)[:4],
                evidence_ledger=ledger,
            ),
            ledger,
        )

    async def _run(self, agent_input: AgentInput) -> CaseManagerAgentOutput:
        legal_summary = getattr(agent_input.context.get("legal"), "summary", "") or ""
        document_summary = getattr(agent_input.context.get("document_analyst"), "summary", "") or ""
        spreadsheet_summary = (
            getattr(agent_input.context.get("spreadsheet_analyst"), "summary", "") or ""
        )
        vision_summary = getattr(agent_input.context.get("vision_ocr"), "summary", "") or ""
        document_context = document_summary or spreadsheet_summary or vision_summary
        rs_output = agent_input.context.get("rs_analyst")
        rs_context = self._rs_prompt_context(rs_output)
        rs_summary = "" if rs_context and rs_context != "Sin contexto RS ampliado." else (getattr(rs_output, "summary", "") or "")
        execution = (
            agent_input.context.get("_execution")
            if isinstance(agent_input.context.get("_execution"), dict)
            else {}
        )
        ledger = self._build_evidence_ledger(agent_input)

        draft_state = self._deterministic_case_state_draft(agent_input)
        draft_payload = CaseStateDraft.model_validate(
            draft_state.model_dump(exclude={"evidence_ledger"})
        )

        if not self.external_enabled():
            case_state = draft_state
            return CaseManagerAgentOutput(
                agent=self.name, summary=case_state.case_summary, data=case_state
            )

        user = (
            f"{render_prompt('case_manager_user.txt', query=agent_input.query, decision_mode=agent_input.decision_mode, memory_summary=summarize_memory_context(str(agent_input.context.get('user_memory', '') or '')), case_history=summarize_case_history(agent_input.context.get('case_history', [])), legal_summary=legal_summary, readiness_summary=document_context, rs_summary=rs_summary, rs_context=rs_context, execution_summary=summarize_execution_report(execution), evidence_ledger=json.dumps([item.model_dump() for item in ledger.modalities], ensure_ascii=False), deterministic_draft=json.dumps(draft_payload.model_dump(), ensure_ascii=False))}\n\n"
            "Devuelve SOLO JSON conforme a este esquema:\n"
            f"{json.dumps(CASE_STATE_SCHEMA, ensure_ascii=False)}"
        )
        try:
            data = await self.call_llm_json(
                system=compose_system_prompt(
                    agent_name="case_manager",
                    body=render_prompt("case_manager_system.txt"),
                    output_contract="Devuelve exclusivamente JSON válido siguiendo el esquema proporcionado.",
                ),
                user=user,
                schema=CASE_STATE_SCHEMA,
                temperature=0.1,
            )
        except Exception:
            data = {}

        if not data:
            case_state = draft_state
            return CaseManagerAgentOutput(
                agent=self.name, summary=case_state.case_summary, data=case_state
            )

        open_tasks = []
        for item in data.get("open_tasks") or []:
            open_tasks.append(CaseTask(**item))

        llm_state = CaseState(
            case_summary=str(data.get("case_summary") or "").strip() or "Caso en seguimiento.",
            open_tasks=open_tasks,
            blocked_by=[item for item in (data.get("blocked_by") or []) if isinstance(item, str)],
            recommended_next_input=[
                item for item in (data.get("recommended_next_input") or []) if isinstance(item, str)
            ],
            evidence_ledger=ledger,
        )
        case_state = self._merge_case_state_with_draft(draft_state, llm_state, ledger=ledger)
        return CaseManagerAgentOutput(
            agent=self.name, summary=case_state.case_summary, data=case_state
        )
