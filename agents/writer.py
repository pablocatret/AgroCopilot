from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Iterable, List, Mapping

from loguru import logger

from agents.base import BaseAgent
from libs.prompts import compose_system_prompt, render_prompt
from libs.temporal_selection import PREFERRED_MIN_GAP_DAYS, select_temporal_pair
from libs.context_engineering import (
    summarize_agent_context_blocks,
    summarize_conversation_history,
    summarize_execution_report,
    summarize_memory_context,
    summarize_memory_reuse,
    summarize_refs,
)
from libs.schemas import (
    AgentInput,
    AgentRef,
    AgentRefs,
    CapAdvice,
    CaseEvidenceLedger,
    DocumentReadiness,
    FinalAnswer,
    FieldIntakeAdvice,
    ImageInsights,
    LegalFindings,
    MemoryUsage,
    Reference,
    StacResults,
    TemporalComparison,
    TemporalSceneSummary,
    WebResearch,
    WriterFastPathTrace,
    WriterAgentOutput,
)

WRITER_RESPONSE_SCHEMA_BASE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "executive_summary": {"type": "string", "minLength": 1},
        "report_md": {"type": "string", "minLength": 1},
        "recommendations": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "next_actions": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "missing_information": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "documents_needed": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "evidence_summary": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
    },
}

_SCHEMA_REQUIRED_BY_MODE = {
    "BRIEFING": ["executive_summary", "report_md"],
    "STANDARD": ["executive_summary", "report_md", "recommendations", "next_actions", "limitations", "evidence_summary"],
    "DEEP_DIVE": ["executive_summary", "report_md", "recommendations", "limitations", "next_actions", "missing_information", "documents_needed", "evidence_summary"],
}


def get_writer_response_schema(length_mode: str = "STANDARD") -> dict:
    required = _SCHEMA_REQUIRED_BY_MODE.get(length_mode, _SCHEMA_REQUIRED_BY_MODE["STANDARD"])
    return {**WRITER_RESPONSE_SCHEMA_BASE, "required": required}


def _merge_refs(*refs: AgentRefs) -> List[AgentRef]:
    merged: List[AgentRef] = []
    seen: set[str] = set()
    for ref_block in refs:
        for ref in ref_block.items:
            key = ref.ref_id or ref.url or ref.title
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(ref)
    return merged


def _format_refs(refs: List[AgentRef]) -> str:
    if not refs:
        return "No hay referencias para citar."
    return "\n".join(
        [f"[{idx}] {ref.title} â€” {ref.url or ref.ref_id}" for idx, ref in enumerate(refs, start=1)]
    )


@dataclass
class ConversationEvidenceBundle:
    response_path: str
    writer_search_allowed: bool
    fast_path: WriterFastPathTrace
    escalation_reason: str | None
    search_used: bool
    research: WebResearch | None
    web_refs: List[AgentRef]
    combined_refs: List[AgentRef]
    agent_summary: str
    execution_summary: str
    memory_summary: str
    context_window: str


class WriterAgent(BaseAgent):
    name = "writer"
    output_model = WriterAgentOutput
    requires_llm = True
    _provider_key = "LLM_PROVIDER_WRITER"

    def __init__(self) -> None:
        super().__init__()
        from backend.deps import settings

        self.model = settings.resolve_openai_model("OPENAI_MODEL_WRITER")

    @staticmethod
    def _semantic_key(text: str) -> str:
        return " ".join(str(text).strip().lower().replace(".", "").split())

    @classmethod
    def _semantic_dedupe(cls, values: Iterable[str]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value).strip()
            if not text:
                continue
            key = cls._semantic_key(text)
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out

    @staticmethod
    def _looks_generic_catchall(text: str) -> bool:
        normalized = text.strip().lower()
        normalized = unicodedata.normalize("NFKD", normalized).encode("ascii", "ignore").decode()
        generic_fragments = [
            "cualquier otro",
            "cualquier otra",
            "otros documentos",
            "otras pruebas",
            "mas documentacion",
            "documentacion adicional",
            "otro documento relevante",
        ]
        return any(fragment in normalized for fragment in generic_fragments)

    @staticmethod
    def _looks_document_like(text: str) -> bool:
        normalized = text.strip().lower()
        doc_terms = [
            "certificado",
            "justificante",
            "solicitud",
            "informe",
            "registro",
            "presupuesto",
            "anÃ¡lisis",
            "analisis",
            "documento",
            "factura",
            "csv",
            "expediente",
        ]
        return any(term in normalized for term in doc_terms)

    @staticmethod
    def _contains_internal_detail(text: str) -> bool:
        normalized = str(text).strip().lower()
        internal_terms = [
            "retry",
            "typeerror",
            "traceback",
            "campaign_tracker",
            "campaña_tracker",
            "_tracker",
            "devolvio error tecnico",
            "stac disabled",
            "stac#",
            "stac/rs",
            "rs_analyst",
            "rs disabled",
            "stack trace",
            "routing",
            "final_level",
            "_execution",
        ]
        return any(term in normalized for term in internal_terms)

    @staticmethod
    def _has_action_purpose(text: str) -> bool:
        normalized = str(text).strip().lower()
        purpose_markers = [
            " para ",
            " con el fin de ",
            " de cara a ",
            " porque ",
            " debido a ",
            " ante ",
        ]
        return any(marker in normalized for marker in purpose_markers)

    @classmethod
    def _has_substantive_actions(cls, actions: List[str]) -> bool:
        if len(actions) < 4:
            return False
        return sum(1 for item in actions if cls._has_action_purpose(item)) >= 3

    @staticmethod
    def _professional_execution_limitation(
        agent: str, level: str, info: Mapping[str, Any]
    ) -> str | None:
        normalized_agent = str(agent).strip().lower()
        normalized_level = str(level).strip().lower()
        if normalized_level in {"ok", "info", "none"}:
            return None
        if normalized_agent in {"stac", "rs_analyst", "remote_sensing"}:
            return (
                "No hay analisis temporal de teledeteccion disponible con evidencia suficiente; "
                "las decisiones de campo deben apoyarse en observacion directa o datos recientes."
            )
        if normalized_agent in {"vision_ocr", "ocr", "attachment_reader"}:
            return (
                "La lectura automatica de adjuntos no aporta evidencia suficiente; conviene validar "
                "los originales antes de cerrar la decision."
            )
        if normalized_agent in {"legal", "cap_advisor", "document_readiness"}:
            return (
                "La evidencia documental o normativa es incompleta; no debe asumirse cumplimiento "
                "sin verificar los documentos concretos indicados."
            )
        if normalized_agent in {"campaign_tracker", "case_manager"}:
            return None
        return (
            "Una fuente auxiliar no aporto evidencia suficiente; la recomendacion queda limitada "
            "a los datos disponibles en el caso."
        )

    @classmethod
    def _clean_visible_items(cls, values: Iterable[str]) -> List[str]:
        cleaned: List[str] = []
        for value in values:
            text = str(value).strip()
            if not text or cls._contains_internal_detail(text):
                continue
            if text not in cleaned:
                cleaned.append(text)
        return cleaned

    @classmethod
    def _clean_visible_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if not cls._contains_internal_detail(text):
            return text
        cleaned_lines = [
            line for line in text.splitlines() if not cls._contains_internal_detail(line)
        ]
        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def _action_from_open_task(task) -> str:
        if hasattr(task, "title") and not isinstance(task, str):
            title = task.title
            rationale = getattr(task, "rationale", None) or ""
        else:
            title = str(task)
            rationale = ""
        clean_title = " ".join(str(title).strip().split())
        if not clean_title:
            return ""
        if rationale and len(rationale.strip()) > 10:
            return f"{clean_title} ({rationale.strip()})"
        return f"{clean_title}."

    @classmethod
    def _prune_list_noise(
        cls,
        *,
        next_actions: List[str],
        missing_information: List[str],
        documents_needed: List[str],
        max_actions: int = 10,
    ) -> tuple[List[str], List[str], List[str]]:
        docs = [
            item
            for item in cls._semantic_dedupe(documents_needed)
            if not cls._looks_generic_catchall(item)
        ]
        missing: List[str] = []
        doc_keys = {cls._semantic_key(item) for item in docs}
        for item in cls._semantic_dedupe(missing_information):
            if cls._looks_generic_catchall(item):
                continue
            key = cls._semantic_key(item)
            if key in doc_keys:
                continue
            if cls._looks_document_like(item) and any(
                key in cls._semantic_key(doc) or cls._semantic_key(doc) in key for doc in docs
            ):
                continue
            missing.append(item)
        actions = [
            item
            for item in cls._semantic_dedupe(next_actions)
            if not cls._looks_generic_catchall(item) and not cls._contains_internal_detail(item)
        ]
        return actions[:max_actions], missing[:6], docs[:6]

    @staticmethod
    def _format_agent_data(data: Any) -> str:
        def is_irrelevant_key(key: str) -> bool:
            normalized = key.lower()
            return normalized in {
                "id",
                "ref_id",
                "uuid",
                "created_at",
                "updated_at",
                "timestamp",
                "source_id",
            }

        def clean_value(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, str):
                return value.strip() or None
            if isinstance(value, Mapping):
                cleaned: dict[str, Any] = {}
                for k, v in value.items():
                    if is_irrelevant_key(str(k)):
                        continue
                    cleaned_value = clean_value(v)
                    if cleaned_value in (None, "", [], {}):
                        continue
                    cleaned[str(k)] = cleaned_value
                return cleaned or None
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                cleaned_items = [clean_value(item) for item in value]
                cleaned_items = [item for item in cleaned_items if item not in (None, "", [], {})]
                return cleaned_items or None
            return value

        def to_markdown(value: Any, indent: int = 0) -> List[str]:
            prefix = "  " * indent
            lines: List[str] = []
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if isinstance(item, (Mapping, list, tuple)):
                        lines.append(f"{prefix}- **{key}**:")
                        lines.extend(to_markdown(item, indent + 1))
                    else:
                        lines.append(f"{prefix}- **{key}**: {item}")
                return lines
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                for item in value:
                    if isinstance(item, (Mapping, list, tuple)):
                        lines.append(f"{prefix}-")
                        lines.extend(to_markdown(item, indent + 1))
                    else:
                        lines.append(f"{prefix}- {item}")
                return lines
            return [f"{prefix}- {value}"]

        if data is None:
            return "Sin datos."
        payload = data.model_dump(exclude_none=True) if hasattr(data, "model_dump") else data
        cleaned = clean_value(payload)
        if cleaned in (None, "", [], {}):
            return "Sin datos relevantes."
        if isinstance(cleaned, str):
            return cleaned
        return "\n".join(to_markdown(cleaned))

    @staticmethod
    def _format_execution_report(execution: Mapping[str, Any] | None) -> str:
        if not execution:
            return "Sin incidencias de ejecución registradas."
        lines: List[str] = []
        for agent, info in execution.items():
            final_level = str(info.get("final_level", "ok"))
            instances = info.get("instances", []) or []
            details = ", ".join(
                str(item.get("message", "")).strip() for item in instances if item.get("message")
            )
            if details:
                lines.append(f"- {agent}: {final_level} ({details})")
            else:
                lines.append(f"- {agent}: {final_level}")
        return "\n".join(lines) if lines else "Sin incidencias de ejecución registradas."

    @staticmethod
    def _resolve_evidence_ledger(case_output: Any) -> CaseEvidenceLedger:
        case_state = getattr(case_output, "data", None) if case_output else None
        ledger = getattr(case_state, "evidence_ledger", None)
        return ledger if isinstance(ledger, CaseEvidenceLedger) else CaseEvidenceLedger()

    @staticmethod
    def _normalize_query(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(text or "").strip().lower())
        return normalized.encode("ascii", "ignore").decode()

    @classmethod
    def _fast_path_escalation_reason(cls, agent_input: AgentInput) -> str | None:
        if agent_input.attachments:
            return "Hay adjuntos y la respuesta necesita extractores especializados."
        query = cls._normalize_query(agent_input.query)
        escalation_rules = (
            (r"\b(pdf|docx|doc|csv|xlsx|adjunto|archivo|factura|expediente|analisis|informe)\b",
             "La consulta depende de documentos o evidencias adjuntas."),
            (r"\b(sentinel|landsat|stac|ndvi|ndmi|teledeteccion|satelit|comparar|serie temporal|monitor)\b",
             "La consulta requiere teledeteccion o comparacion temporal especializada."),
            (r"\b(reglamento|normativa|boe|ue|globalg\.?a\.?p|certificacion|cumplimiento|legal)\b",
             "La consulta requiere validacion normativa o legal especializada."),
        )
        for pattern, reason in escalation_rules:
            if re.search(pattern, query):
                return reason
        return None

    @classmethod
    def _should_use_fast_path_search(cls, query: str) -> bool:
        normalized = cls._normalize_query(query)
        if len(normalized.split()) <= 2:
            return False
        search_markers = (
            "precio",
            "precios",
            "cuanto cuesta",
            "coste",
            "costo",
            "mercado",
            "cotizacion",
            "subvencion",
            "ayuda",
            "pac",
            "reglamento",
            "normativa",
            "certificacion",
            "requisitos",
            "plaga",
            "enfermedad",
            "tratamiento",
            "fertilizante",
            "variedad",
            "cosecha",
            "plantacion",
            "riego",
            "hoy",
            "actual",
            "actualizado",
            "ultima",
            "ultimas",
            "reciente",
            "2025",
            "2026",
        )
        return any(marker in normalized for marker in search_markers)

    @staticmethod
    def _has_specialized_context(ctx: Mapping[str, Any]) -> bool:
        return any(
            name
            not in {
                "_execution",
                "_memory",
                "_plan",
                "_conversation_history",
                "_memory_reuse",
                "_case_context",
                "_case_context_usage",
                "rs_config",
            }
            and value
            for name, value in ctx.items()
        )

    async def _run_fast_path_search(self, query: str) -> tuple[WebResearch, List[AgentRef]]:
        from libs.search_tool import WebSearchTool

        search_tool = WebSearchTool(
            description=(
                "Busqueda puntual para el writer fast path. Solo usar para contexto factual "
                "actualizado cuando no haga falta escalar a agentes especializados."
            ),
            max_fetch=4,
            do_fetch=True,
        )
        payload = await search_tool.run({"query": query, "max_results": 4})
        results = payload.get("results") or []
        refs: List[AgentRef] = []
        references: List[Reference] = []
        for index, item in enumerate(results, start=1):
            title = str(item.get("title") or "Fuente web").strip()
            url = str(item.get("url") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            if not url:
                continue
            refs.append(
                AgentRef(
                    ref_id=f"writer-web-{index}",
                    title=title,
                    source="web",
                    url=url,
                    snippet=snippet,
                    metadata={"verification": "writer_fast_path_search"},
                )
            )
            references.append(Reference(title=title, url=url, snippet=snippet))
        return WebResearch(findings=[], references=references), refs

    async def _assemble_conversation_evidence(
        self,
        agent_input: AgentInput,
        ctx: Mapping[str, Any],
        *,
        refs_blocks: List[AgentRefs],
        agent_summary: str,
        execution_summary: str,
        memory_summary: str,
        context_window: str,
    ) -> ConversationEvidenceBundle:
        plan_meta = ctx.get("_plan") if isinstance(ctx.get("_plan"), Mapping) else {}
        plan_steps = [
            str(step).strip().lower()
            for step in (plan_meta.get("steps") or [])
            if str(step).strip()
        ]
        plan_policy = (
            plan_meta.get("policy") if isinstance(plan_meta.get("policy"), Mapping) else {}
        )
        writer_search_allowed = bool(plan_policy.get("writer_search_allowed"))
        raw_fast_path = (
            plan_policy.get("fast_path") if isinstance(plan_policy.get("fast_path"), Mapping) else {}
        )
        has_specialized_context = self._has_specialized_context(ctx)
        is_single_writer_plan = plan_steps == ["writer"] or (
            not plan_steps and not has_specialized_context
        )
        response_path = (
            "single_agent_fast_path"
            if is_single_writer_plan and not has_specialized_context
            else "multi_agent_synthesis"
        )
        escalation_reason = (
            None
            if response_path == "multi_agent_synthesis"
            else self._fast_path_escalation_reason(agent_input)
        )
        search_used = False
        research = None
        web_refs: List[AgentRef] = []
        if (
            writer_search_allowed
            and escalation_reason is None
            and self.external_enabled()
            and self._should_use_fast_path_search(agent_input.query)
        ):
            research, web_refs = await self._run_fast_path_search(agent_input.query)
            search_used = bool(web_refs)
        base_refs = _merge_refs(*refs_blocks) if refs_blocks else []
        combined_refs = _merge_refs(*refs_blocks, AgentRefs(items=web_refs)) if web_refs else base_refs
        fast_path = WriterFastPathTrace(
            enabled=bool(raw_fast_path.get("enabled", response_path == "single_agent_fast_path")),
            search_allowed=bool(raw_fast_path.get("allow_search", writer_search_allowed)),
            search_used=search_used,
            disclose_search_use=bool(raw_fast_path.get("disclose_search_use", True)),
            disclose_sources=bool(raw_fast_path.get("disclose_sources", True)),
            escalation_required=bool(escalation_reason),
            escalation_reason=escalation_reason,
        )
        return ConversationEvidenceBundle(
            response_path=response_path,
            writer_search_allowed=writer_search_allowed,
            fast_path=fast_path,
            escalation_reason=escalation_reason,
            search_used=search_used,
            research=research,
            web_refs=web_refs,
            combined_refs=combined_refs,
            agent_summary=agent_summary,
            execution_summary=execution_summary,
            memory_summary=memory_summary,
            context_window=context_window,
        )

    @classmethod
    def _conversation_prompt_context(
        cls,
        agent_input: AgentInput,
        evidence: ConversationEvidenceBundle,
    ) -> str:
        rs_context = cls._build_rs_reasoning_context(agent_input.context or {})
        conversation_history = summarize_conversation_history(
            (agent_input.context or {}).get("_conversation_history")
        )
        memory_reuse = summarize_memory_reuse((agent_input.context or {}).get("_memory_reuse"))
        case_context = str((agent_input.context or {}).get("_case_context") or "").strip()
        return "\n\n".join(
            [
                f"Consulta:\n{agent_input.query}",
                f"Historial conversacional:\n{conversation_history}",
                (
                    f"Escalado recomendado:\n{evidence.escalation_reason}"
                    if evidence.escalation_reason
                    else "Escalado recomendado:\nNo"
                ),
                (
                    f"Busqueda puntual del writer:\n{summarize_refs(evidence.web_refs)}"
                    if evidence.web_refs
                    else "Busqueda puntual del writer:\nNo usada"
                ),
                f"Resumen de evidencias:\n{evidence.agent_summary}",
                f"Contexto RS para razonar:\n{rs_context}",
                f"Memoria:\n{evidence.memory_summary}",
                f"Contexto confirmado del caso:\n{case_context or 'Sin caso activo asociado.'}",
                f"Reutilizacion estructurada de memoria:\n{memory_reuse}",
                f"Ejecucion:\n{evidence.execution_summary}",
                f"Contexto compacto:\n{evidence.context_window}",
                f"Referencias:\n{summarize_refs(evidence.combined_refs)}",
            ]
        )

    @staticmethod
    def _fallback_next_actions(
        recommendations: List[str],
        legal_output: Any,
        cap_advice: CapAdvice | None,
        document_readiness: DocumentReadiness | None,
        limitations: List[str],
    ) -> List[str]:
        actions = [
            item.strip() for item in recommendations if isinstance(item, str) and item.strip()
        ]
        if legal_output and isinstance(getattr(legal_output, "data", None), LegalFindings):
            pending = [
                finding.requirement
                for finding in legal_output.data.checklist
                if finding.status != "cumple"
            ]
            for item in pending[:2]:
                action = f"Revisar requisito pendiente: {item}"
                if action not in actions:
                    actions.append(action)
        if cap_advice:
            for item in cap_advice.next_steps[:2]:
                if item not in actions:
                    actions.append(item)
        if document_readiness:
            for item in document_readiness.next_steps[:2]:
                if item not in actions:
                    actions.append(item)
        if not actions and limitations:
            actions.append(
                "Aportar observaciones de campo adicionales para detallar la recomendación."
            )
        return actions[:8]

    @staticmethod
    def _fallback_missing_information(
        execution_report: Mapping[str, Any],
        limitations: List[str],
        field_intake: FieldIntakeAdvice | None,
        cap_advice: CapAdvice | None,
    ) -> List[str]:
        missing: List[str] = []
        for agent, info in execution_report.items():
            level = str(info.get("final_level", "ok"))
            if level == "ok":
                continue
            limitation = WriterAgent._professional_execution_limitation(agent, level, info)
            if limitation and limitation not in missing:
                missing.append(limitation)
        for item in limitations:
            text = str(item).strip()
            if text and text not in missing:
                missing.append(text)
        if field_intake:
            for item in field_intake.required_questions[:3]:
                text = item.question.strip()
                if text and text not in missing:
                    missing.append(text)
        if cap_advice:
            for item in cap_advice.gaps[:3]:
                text = str(item).strip()
                if text and text not in missing:
                    missing.append(text)
        return missing[:6]

    @staticmethod
    def _fallback_documents_needed(
        legal_output: Any,
        cap_advice: CapAdvice | None,
        document_readiness: DocumentReadiness | None,
    ) -> List[str]:
        docs: List[str] = []
        if legal_output and isinstance(getattr(legal_output, "data", None), LegalFindings):
            for finding in legal_output.data.checklist:
                if finding.status == "cumple":
                    continue
                docs.append(finding.requirement)
        if cap_advice:
            docs.extend(cap_advice.documents_required)
        if document_readiness:
            docs.extend(document_readiness.missing_documents)
            docs.extend(document_readiness.unclear_documents)
        deduped: List[str] = []
        for item in docs:
            text = str(item).strip()
            if text and text not in deduped:
                deduped.append(text)
        return deduped[:6]

    @staticmethod
    def _fallback_evidence_summary(
        refs: List[AgentRef],
        cap_advice: CapAdvice | None,
        document_readiness: DocumentReadiness | None,
    ) -> List[str]:
        summary = [ref.title for ref in refs[:4] if ref.title]
        if cap_advice:
            for pathway in cap_advice.pathways[:2]:
                item = f"Vía orientativa detectada: {pathway.title}"
                if item not in summary:
                    summary.append(item)
        if document_readiness:
            for item in document_readiness.verified_documents[:2]:
                label = f"Documento con soporte: {item.name}"
                if label not in summary:
                    summary.append(label)
        return summary[:6]

    @classmethod
    def _build_rs_reasoning_context(cls, ctx: Mapping[str, Any]) -> str:
        rs_output = ctx.get("rs_analyst")
        rs_data = getattr(rs_output, "data", None)
        if not isinstance(rs_data, ImageInsights):
            return "Sin evidencia de teledeteccion relevante."

        lines: List[str] = []
        overview = str(getattr(rs_data, "overview", "") or "").strip()
        if overview:
            lines.append(f"Vision general: {overview}")

        changes = list(getattr(rs_data, "temporal_changes", []) or [])
        primary_change = None
        if changes:
            primary_change = sorted(
                changes,
                key=lambda change: (
                    0 if getattr(change, "reliable", False) else 1,
                    {"alta": 0, "media": 1, "baja": 2}.get(getattr(change, "severity", "media"), 1),
                    abs(getattr(change, "delta_mean", 0.0) or 0.0) * -1,
                ),
            )[0]

        is_single_scene = "escena unica" in (getattr(primary_change, "label", "") or "").lower() if primary_change else False

        if primary_change:
            if is_single_scene:
                lines.append(
                    f"Escena unica: {primary_change.detail}"
                )
            else:
                lines.append(
                    "Hallazgo principal: "
                    f"{primary_change.label}. {primary_change.detail}"
                )
                robustness = (
                    "senal suficientemente robusta para orientar decision"
                    if getattr(primary_change, "reliable", False)
                    else "senal util como orientacion, pero no robusta para cerrar diagnostico"
                )
                lines.append(f"Robustez: {robustness}.")
                if getattr(primary_change, "limitations", None):
                    lines.append(
                        "Limitacion principal: "
                        f"{primary_change.limitations[0]}"
                    )
            trend_context = str(getattr(primary_change, "trend_context", "") or "").strip()
            if trend_context:
                lines.append(f"Contexto temporal: {trend_context}")

        if not is_single_scene:
            trends = [
                trend
                for trend in (getattr(rs_data, "trends", {}) or {}).values()
                if getattr(trend, "r_squared", 0) > 0.5 and getattr(trend, "direction", "stable") != "stable"
            ]
            if trends and not any(line.startswith("Contexto temporal:") for line in lines):
                trend = trends[0]
                direction_map = {"ascending": "ascendente", "descending": "descendente"}
                direction = direction_map.get(getattr(trend, "direction", "stable"), getattr(trend, "direction", "stable"))
                lines.append(
                    "Contexto temporal: "
                    f"tendencia {direction} en {trend.metric}; solo usar como contexto, no como prediccion."
                )

        for insight in list(getattr(rs_data, "insights", []) or []):
            visual = getattr(insight, "llm_interpretation", None)
            if not visual:
                continue
            visual_parts: List[str] = []
            if getattr(visual, "visible_patterns", None):
                visual_parts.append(visual.visible_patterns[0])
            if getattr(visual, "health_indicators", None):
                visual_parts.append(f"salud/vigor: {visual.health_indicators[0]}")
            if getattr(visual, "anomalies", None):
                visual_parts.append(f"anomalias: {visual.anomalies[0]}")
            conflict_note = ""
            if getattr(visual, "supports_index_signal", "unclear") == "conflicts":
                conflict_note = "la lectura visual no refuerza claramente la senal cuantitativa"
            display_parts = visual_parts[:2]
            if conflict_note:
                display_parts.append(conflict_note)
            if display_parts:
                lines.append(
                    "Observacion visual auxiliar: "
                    f"{' ; '.join(display_parts)}. No tratar como verdad cerrada."
                )
                break

        meteo = ctx.get("meteo")
        pii = getattr(meteo, "precipitation_irregularity_index", None) if meteo else None
        total_precip = getattr(meteo, "total_precip_mm", None) if meteo else None
        if pii is not None and pii <= -1.0:
            lines.append("Contexto meteo: posible sequia relativa durante el periodo analizado.")
        elif total_precip is not None:
            lines.append(f"Contexto meteo: precipitacion acumulada del periodo {total_precip:.1f} mm.")

        max_lines = 8
        if len(lines) > max_lines:
            logger.info(
                "rs_context_truncated: kept={} total={} dropped={}",
                max_lines,
                len(lines),
                len(lines) - max_lines,
            )
        return "\n".join(lines[:max_lines]) if lines else "Sin evidencia de teledeteccion relevante."

    @staticmethod
    def _scene_preview(item: Any) -> str | None:
        assets = getattr(item, "assets", None) or []
        for asset in assets:
            thumb = getattr(asset, "thumbnail", None)
            if thumb:
                return thumb
        for asset in assets:
            href = getattr(asset, "href", None)
            if href:
                return href
        return None

    @classmethod
    def _fallback_temporal_comparison(
        cls,
        stac: StacResults | None,
        remote_sensing: ImageInsights | None,
    ) -> TemporalComparison | None:
        if not stac or len(stac.items) < 2:
            return None
        comparable_items = [
            item
            for item in stac.items
            if (getattr(item, "product_type", None) or "").lower() != "landcover"
            and (getattr(item, "index_name", None) or "").upper() != "ESA_WORLDCOVER"
        ]
        if len(comparable_items) < 2:
            return None
        temporal_changes = list(remote_sensing.temporal_changes or []) if remote_sensing else []
        metric_priority = {"NDVI": 0, "NDMI": 1, "NDWI": 2}
        primary_change = None
        if temporal_changes:
            primary_change = sorted(
                temporal_changes,
                key=lambda change: (
                    metric_priority.get((getattr(change, "metric", None) or "").upper(), 50),
                    0 if getattr(change, "reliable", False) else 1,
                ),
            )[0]
        primary_metric = (getattr(primary_change, "metric", None) or "").upper()
        if primary_metric:
            metric_items = [
                item
                for item in comparable_items
                if (getattr(item, "index_name", None) or "").upper() == primary_metric
            ]
            if len(metric_items) >= 2:
                comparable_items = metric_items
        selection = getattr(stac, "temporal_selection", None)
        pair = select_temporal_pair(
            comparable_items,
            preferred_min_gap_days=PREFERRED_MIN_GAP_DAYS,
            selected_previous_id=getattr(selection, "previous_item_id", None),
            selected_current_id=getattr(selection, "current_item_id", None),
        )
        if pair is None:
            return None
        previous_item = pair.previous
        current_item = pair.current
        insights_by_id = {
            insight.item_id: insight.summary
            for insight in (remote_sensing.insights if remote_sensing else [])
            if getattr(insight, "item_id", None)
        }
        previous = TemporalSceneSummary(
            item_id=previous_item.id,
            datetime=previous_item.datetime,
            preview_href=cls._scene_preview(previous_item),
            summary=insights_by_id.get(previous_item.id),
            product_label=getattr(previous_item, "product_label", None),
            stats=getattr(previous_item, "index_stats", None),
            quality=getattr(previous_item, "quality", None),
        )
        current = TemporalSceneSummary(
            item_id=current_item.id,
            datetime=current_item.datetime,
            preview_href=cls._scene_preview(current_item),
            summary=insights_by_id.get(current_item.id),
            product_label=getattr(current_item, "product_label", None),
            stats=getattr(current_item, "index_stats", None),
            quality=getattr(current_item, "quality", None),
        )
        first_change = primary_change if primary_change else None
        changes: List[str] = []
        if temporal_changes:
            ordered_changes = sorted(
                temporal_changes,
                key=lambda change: (
                    0 if change is first_change else 1,
                    metric_priority.get((getattr(change, "metric", None) or "").upper(), 50),
                ),
            )
            for change in ordered_changes[:3]:
                changes.append(f"{change.label}: {change.detail}")
                if getattr(change, "limitations", None):
                    changes.extend([f"Cautela: {item}" for item in change.limitations[:2]])
        else:
            changes.append(
                (
                    f"Evidencia temporal disponible entre {previous.datetime or 'la escena de referencia'} "
                    f"y {current.datetime or 'la escena actual'}, sin metrica suficiente para declarar cambio."
                )
            )
        if current.summary:
            changes.append(f"Escena actual ({current.item_id}): {current.summary}")
        if previous.summary:
            changes.append(f"Escena de referencia ({previous.item_id}): {previous.summary}")
        if not changes:
            changes.append(
                "Hay al menos dos escenas disponibles para comparar evolución entre fechas."
            )
        return TemporalComparison(
            available=True,
            label=(
                "Lectura multisenal"
                if len(
                    {(change.metric or "").upper() for change in temporal_changes if change.metric}
                )
                > 1
                else "Cambio satelital medido" if first_change else "Evidencia temporal disponible"
            ),
            rationale=getattr(selection, "rationale", None) or pair.rationale,
            previous=previous,
            current=current,
            key_changes=changes[:3],
            metric=getattr(first_change, "metric", None),
            delta_mean=getattr(first_change, "delta_mean", None),
            severity=getattr(first_change, "severity", None) if first_change else None,
            confidence=getattr(first_change, "confidence", None) if first_change else None,
            limitations=getattr(first_change, "limitations", []) if first_change else [],
            change_preview_href=(
                getattr(first_change, "preview_href", None) if first_change else None
            ),
        )

    async def _run(self, agent_input: AgentInput) -> WriterAgentOutput:
        ctx = agent_input.context or {}
        length_mode = agent_input.writer_mode or "STANDARD"
        execution_report = (
            ctx.get("_execution") if isinstance(ctx.get("_execution"), Mapping) else {}
        )
        plan_meta = ctx.get("_plan") if isinstance(ctx.get("_plan"), Mapping) else {}
        memory_meta = ctx.get("_memory") if isinstance(ctx.get("_memory"), Mapping) else {}
        memory_context = str(memory_meta.get("context") or "").strip()

        summaries: List[str] = []
        refs_blocks: List[AgentRefs] = []
        for name, output in ctx.items():
            if name in ("_execution", "rs_analyst"):
                continue
            if not output:
                continue
            summary = getattr(output, "summary", None)
            if summary:
                summaries.append(f"- {name}: {summary}")
            refs = getattr(output, "refs", None)
            if refs:
                refs_blocks.append(refs)
        context_window = summarize_agent_context_blocks(ctx)
        agent_summary = "\n".join(summaries) or "Sin resumen disponible."
        execution_summary = summarize_execution_report(execution_report)
        memory_summary = summarize_memory_context(memory_context)

        refs = _merge_refs(*refs_blocks) if refs_blocks else []

        legal_output = ctx.get("legal")
        rs_output = ctx.get("rs_analyst")
        stac_output = ctx.get("stac")
        case_output = ctx.get("case_manager")
        evidence_ledger = self._resolve_evidence_ledger(case_output)

        if agent_input.response_mode == "conversation":
            evidence = await self._assemble_conversation_evidence(
                agent_input,
                ctx,
                refs_blocks=refs_blocks,
                agent_summary=agent_summary,
                execution_summary=execution_summary,
                memory_summary=memory_summary,
                context_window=context_window,
            )
            conversation_context = self._conversation_prompt_context(agent_input, evidence)
            if self.external_enabled():
                message_md = await self.call_llm_text(
                    system=compose_system_prompt(
                        agent_name="writer",
                        body=(
                            "Responde a la consulta agricola de forma directa y util. Usa Markdown normal. "
                            "Da prioridad a la intencion del usuario sobre cualquier esquema interno. "
                            "Si el contexto indica escalado recomendado, dilo de forma explicita, explica por "
                            "que faltan agentes especializados y no simules una resolucion cerrada. Si hay "
                            "evidencias recuperadas, integralas para resolver el problema y deja claras las "
                            "incertidumbres que cambian la decision. No menciones agentes internos, prompts, "
                            "reintentos ni mecanica del sistema. No impongas una plantilla: usa solo la "
                            "estructura que mejore la respuesta. Si hay teledeteccion, usala como evidencia "
                            "de apoyo sin volcar tecnicismos ni listas innecesarias."
                        ),
                        output_contract="Devuelve solo Markdown visible para el usuario.",
                    ),
                    user=conversation_context,
                    temperature=0.35,
                )
            else:
                message_md = ""
            message_md = self._clean_visible_text(message_md) or (
                "Puedo ayudarte con ese caso. Con la informacion disponible, empezaria por concretar "
                "la parcela, el cultivo, la fecha de la observacion y cualquier documento o imagen "
                "relevante antes de cerrar una recomendacion."
            )
            executive_summary = message_md.split("\n\n", 1)[0].replace("#", "").strip()
            limitations: List[str] = []
            for agent, info in execution_report.items():
                level = str(info.get("final_level", "ok"))
                if level == "ok":
                    continue
                item = self._professional_execution_limitation(agent, level, info)
                if item and item not in limitations:
                    limitations.append(item)
            answer = FinalAnswer(
                executive_summary=executive_summary,
                message_md=message_md,
                report_md=message_md,
                response_path=evidence.response_path,
                search_used=evidence.search_used,
                escalation_required=bool(evidence.escalation_reason),
                escalation_reason=evidence.escalation_reason,
                fast_path=evidence.fast_path,
                legal=legal_output.data if legal_output else None,
                remote_sensing=rs_output.data if rs_output else None,
                research=evidence.research,
                stac=stac_output.data if stac_output else None,
                case_state=case_output.data if case_output else None,
                temporal_comparison=self._fallback_temporal_comparison(
                    stac_output.data if stac_output else None,
                    rs_output.data if rs_output else None,
                ),
                limitations=limitations,
                evidence_summary=self._fallback_evidence_summary(evidence.combined_refs, None, None),
                references=evidence.combined_refs,
                attachments=agent_input.attachments,
                language=agent_input.language,
                memory=MemoryUsage(
                    enabled=bool(memory_meta.get("enabled")),
                    user_id=memory_meta.get("user_id"),
                    used_sections=list(memory_meta.get("used_sections") or []),
                ),
                evidence_ledger=evidence_ledger,
            )
            return WriterAgentOutput(
                agent=self.name,
                summary=executive_summary,
                refs=AgentRefs(items=evidence.combined_refs),
                data=answer,
            )

        prompt = render_prompt(
            "writer_user.txt",
            language=agent_input.language,
            query=agent_input.query,
            decision_mode=agent_input.decision_mode,
            agent_summary=agent_summary,
            execution_summary=execution_summary,
            memory_summary=memory_summary,
            length_mode=length_mode,
            rs_context=self._build_rs_reasoning_context(ctx),
            context_window=context_window,
            references_block=summarize_refs(refs),
        )

        if self.external_enabled():
            required_fields = _SCHEMA_REQUIRED_BY_MODE.get(length_mode, _SCHEMA_REQUIRED_BY_MODE["STANDARD"])
            fields_str = ", ".join(required_fields)
            llm_payload = await self.call_llm_json(
                system=compose_system_prompt(
                    agent_name="writer",
                    body=render_prompt("writer_system.txt", length_mode=length_mode),
                    output_contract=(
                        f"Devuelve exclusivamente JSON válido con {fields_str}. "
                        "No expliques el proceso ni menciones agentes internos."
                    ),
                ),
                user=prompt,
                schema=get_writer_response_schema(length_mode),
                temperature=0.3,
            )
        else:
            llm_payload = {}

        if isinstance(llm_payload, dict) and llm_payload:
            llm_payload = self._validate_report_quality(llm_payload, length_mode)

        report_md = llm_payload.get("report_md", "") if isinstance(llm_payload, dict) else ""
        report_md = self._clean_visible_text(report_md)
        exec_summary = (
            llm_payload.get("executive_summary", "") if isinstance(llm_payload, dict) else ""
        ) or report_md.split("\n\n", 1)[0].replace("#", "").strip()
        exec_summary = self._clean_visible_text(exec_summary)
        recommendations = [
            item
            for item in (llm_payload.get("recommendations") or [])
            if isinstance(item, str) and item.strip()
        ]
        recommendations = self._clean_visible_items(recommendations)
        limitations = [
            item
            for item in (llm_payload.get("limitations") or [])
            if isinstance(item, str) and item.strip()
        ]
        limitations = self._clean_visible_items(limitations)
        if legal_output and isinstance(getattr(legal_output, "data", None), LegalFindings):
            for item in legal_output.data.limitations:
                if item and item not in limitations:
                    limitations.append(item)
        for agent, info in execution_report.items():
            level = str(info.get("final_level", "ok"))
            if level == "ok":
                continue
            item = self._professional_execution_limitation(agent, level, info)
            if item and item not in limitations:
                limitations.append(item)
        case_state = case_output.data if case_output else None
        next_actions = [
            item
            for item in (llm_payload.get("next_actions") or [])
            if isinstance(item, str) and item.strip()
        ] or self._fallback_next_actions(
            recommendations, legal_output, None, None, limitations
        )
        next_actions = self._clean_visible_items(next_actions)
        missing_information = [
            item
            for item in (llm_payload.get("missing_information") or [])
            if isinstance(item, str) and item.strip()
        ] or self._fallback_missing_information(
            execution_report, limitations, None, None
        )
        missing_information = self._clean_visible_items(missing_information)
        documents_needed = [
            item
            for item in (llm_payload.get("documents_needed") or [])
            if isinstance(item, str) and item.strip()
        ] or self._fallback_documents_needed(legal_output, None, None)
        documents_needed = self._clean_visible_items(documents_needed)
        evidence_summary = [
            item
            for item in (llm_payload.get("evidence_summary") or [])
            if isinstance(item, str) and item.strip()
        ] or self._fallback_evidence_summary(refs, None, None)
        evidence_summary = self._clean_visible_items(evidence_summary)
        temporal_comparison = self._fallback_temporal_comparison(
            stac_output.data if stac_output else None,
            rs_output.data if rs_output else None,
        )
        has_rs_in_summary = any(
            "teledeteccion" in item.lower() or "ndvi" in item.lower() or "ndwi" in item.lower()
            for item in evidence_summary
        )
        if rs_output and isinstance(getattr(rs_output, "data", None), ImageInsights) and not has_rs_in_summary:
            rs_data = rs_output.data
            for change in (rs_data.temporal_changes or [])[:1]:
                metric = getattr(change, "metric", None) or "indice"
                item = f"Teledeteccion {metric}: {change.label}. {change.detail}"
                if item not in evidence_summary:
                    evidence_summary.append(item)
            for insight in rs_data.insights[:1]:
                if insight.summary and insight.summary not in evidence_summary:
                    evidence_summary.append(insight.summary)
                visual = getattr(insight, "llm_interpretation", None)
                if visual and visual.health_indicators:
                    visual_item = f"Visual IA: {visual.health_indicators[0]}"
                    if visual_item not in evidence_summary:
                        evidence_summary.append(visual_item)
            for trend_key, trend in list((rs_data.trends or {}).items())[:1]:
                if trend.r_squared > 0.5 and trend.direction != "stable":
                    direction_map = {"ascending": "ascendente", "descending": "descendente"}
                    trend_item = f"Tendencia {direction_map.get(trend.direction, trend.direction)} en {trend.metric}"
                    if trend_item not in evidence_summary:
                        evidence_summary.append(trend_item)
        if case_state and case_state.open_tasks:
            for task in case_state.open_tasks[:4]:
                action = self._action_from_open_task(task)
                if action and action not in next_actions:
                    next_actions.append(action)
        if case_state and case_state.recommended_next_input:
            for item in case_state.recommended_next_input[:3]:
                if item not in missing_information and not self._contains_internal_detail(item):
                    missing_information.append(item)
        limitations = self._clean_visible_items(limitations)
        recommendations = self._clean_visible_items(recommendations)
        next_actions = self._clean_visible_items(next_actions)
        missing_information = self._clean_visible_items(missing_information)
        documents_needed = self._clean_visible_items(documents_needed)
        evidence_summary = self._clean_visible_items(evidence_summary)
        next_actions, missing_information, documents_needed = self._prune_list_noise(
            next_actions=next_actions,
            missing_information=missing_information,
            documents_needed=documents_needed,
        )
        if not exec_summary:
            exec_summary = (
                "Respuesta generada con evidencia parcial; revisar las acciones y limitaciones "
                "antes de tomar una decision."
            )
        if not report_md:
            report_md = exec_summary

        answer = FinalAnswer(
            executive_summary=exec_summary,
            legal=legal_output.data if legal_output else None,
            remote_sensing=rs_output.data if rs_output else None,
            research=None,
            stac=stac_output.data if stac_output else None,
            case_state=case_state,
            temporal_comparison=temporal_comparison,
            recommendations=recommendations,
            limitations=limitations,
            next_actions=next_actions,
            evidence_summary=evidence_summary,
            missing_information=missing_information,
            documents_needed=documents_needed,
            report_md=report_md,
            references=refs,
            attachments=agent_input.attachments,
            language=agent_input.language,
            memory=MemoryUsage(
                enabled=bool(memory_meta.get("enabled")),
                user_id=memory_meta.get("user_id"),
                used_sections=list(memory_meta.get("used_sections") or []),
            ),
            evidence_ledger=evidence_ledger,
        )
        return WriterAgentOutput(
            agent=self.name,
            summary=exec_summary,
            refs=AgentRefs(items=refs),
            data=answer,
        )


class DirectResponseWriterAgent(WriterAgent):
    name = "direct_writer"
