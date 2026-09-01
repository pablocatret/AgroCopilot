"""Run the real multi-agent system with different model overrides.

The historical module name is retained for compatibility. ``run_system``
executes ``ChatOrchestratorService``; it does not implement a monolithic
baseline.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from copy import deepcopy
from typing import Any

from evaluation.llm_support import LLMCallTracker, llm_enabled, provider_enabled
from evaluation.schemas import CaseSpec, NormalizedOutput, ConversationTurn
from libs.costs.pricing import get_model_price
from libs.costs.tracker import finish_cost_capture, start_cost_capture
from libs.schemas import AttachmentMeta


# ── Ejecución del sistema ────────────────────────────────────────────


def _build_attachments(case: CaseSpec) -> list[AttachmentMeta]:
    """Convierte adjuntos del caso al formato que espera el sistema.

    Los fixtures de evaluación son archivos versionados bajo
    ``evaluation/cases/attachments``. Resolverlos aquí evita una evaluación
    engañosa en la que el router ve un PDF o una imagen declarados, pero el
    extractor recibe ``storage_path=None`` y no puede leer su contenido.
    """
    attachments: list[AttachmentMeta] = []
    for index, att in enumerate(case.attachments, start=1):
        original_filename = str(
            att.get("original_filename") or att.get("filename") or ""
        )
        suffix = Path(original_filename).suffix.lower()
        prompt_filename = str(
            att.get("prompt_filename") or f"attachment_{index}{suffix}"
        )
        metadata = deepcopy(att.get("metadata") or {})
        metadata["original_filename"] = original_filename
        metadata["filename_sanitized_for_evaluation"] = True
        raw_path = att.get("storage_path") or att.get("fixture_path")
        if raw_path:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
        else:
            candidate = Path(__file__).resolve().parent / "cases" / "attachments" / att.get("filename", "")
        storage_path = str(candidate.resolve()) if candidate.is_file() else None
        attachments.append(
            AttachmentMeta(
                attachment_id=att.get("attachment_id", ""),
                filename=prompt_filename,
                content_type=att.get("content_type", ""),
                size_bytes=att.get("size_bytes", 0),
                storage_path=storage_path,
                extracted_text=att.get("extracted_text"),
                summary=att.get("summary"),
                metadata=metadata,
            )
        )
    return attachments


def validate_case_attachments(case: CaseSpec) -> list[str]:
    """Validate local fixtures before a benchmark can consume a case."""
    issues: list[str] = []
    built = _build_attachments(case)
    for raw, attachment in zip(case.attachments, built):
        label = f"{case.case_id}:{attachment.filename}"
        if not attachment.storage_path:
            issues.append(f"{label}:missing_file")
            continue
        expected = str(raw.get("sha256") or "").lower().strip()
        if expected:
            digest = hashlib.sha256(Path(attachment.storage_path).read_bytes()).hexdigest()
            if digest != expected:
                issues.append(f"{label}:sha256_mismatch")
    return issues


def _configure_evaluation_agent(
    agent: Any,
    *,
    model_id: str,
    provider: str,
    vision_model_id: str | None = None,
    vision_provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> None:
    """Apply the evaluated model to every specialist instance explicitly.

    The application settings are loaded before evaluation agents are created, so
    mutating environment variables cannot provide a reliable model override.

    If vision_model_id is provided, the vision_ocr agent uses that model
    instead of the text model. All other agents use model_id.
    """
    is_vision_agent = getattr(agent, "name", "") == "vision_ocr"
    if is_vision_agent and vision_model_id:
        agent.model = vision_model_id
        if hasattr(agent, "_provider"):
            agent._provider = vision_provider or provider
    else:
        agent.model = model_id
        if hasattr(agent, "_provider"):
            agent._provider = provider
    if hasattr(agent, "_client"):
        agent._client = None
    agent.evaluation_temperature = temperature
    agent.evaluation_max_tokens = max_tokens


def _record_captured_costs(tracker: LLMCallTracker | None, events: list[dict[str, Any]]) -> None:
    if tracker is None:
        return
    for event in events:
        model = str(event.get("model") or "unknown")
        tracker.record(
            model=model,
            provider=str(event.get("provider") or "unknown"),
            prompt_tokens=int(event.get("input_tokens") or 0),
            completion_tokens=int(event.get("output_tokens") or 0),
            cost_usd=float(event.get("cost_usd") or 0.0),
            cost_known=get_model_price(model, "text") is not None,
            latency_ms=0.0,
            operation=f"system.{event.get('operation') or 'unknown'}",
        )


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _extract_execution_report(result: Any) -> dict[str, Any]:
    """Extract execution metadata from the current and legacy answer shapes."""
    if not isinstance(result, dict):
        return {}
    answer = result.get("answer")
    current: dict[str, Any] | None = None
    if isinstance(answer, dict):
        raw = answer.get("execution")
        if isinstance(raw, dict):
            current = raw
    if current is None:
        legacy = result.get("execution")
        current = legacy if isinstance(legacy, dict) else None
    if current is None:
        return {}

    # The current conversation contract stores agent states directly under
    # answer.execution, while older artifacts used execution.agents.
    if isinstance(current.get("agents"), dict):
        agents = current["agents"]
        metadata = {k: v for k, v in current.items() if k != "agents"}
    else:
        metadata = {}
        agents = {
            k: v for k, v in current.items()
            if isinstance(v, dict) and ("final_level" in v or "instances" in v)
        }
        if not agents:
            # Preserve explicit legacy error-only reports.
            metadata = dict(current)
    normalized = {"agents": agents, **metadata}
    if isinstance(result.get("plan"), dict):
        normalized["plan"] = result["plan"]
    if isinstance(result.get("replan"), dict):
        normalized["replan"] = result["replan"]
    return normalized


async def run_system(
    case: CaseSpec,
    *,
    model_id: str | None = None,
    model_provider: str | None = None,
    vision_model_id: str | None = None,
    vision_provider: str | None = None,
    model_temperature: float | None = None,
    model_max_tokens: int | None = None,
    tracker: LLMCallTracker | None = None,
    query_override: str | None = None,
    conversation_id: str | None = None,
) -> tuple[NormalizedOutput, dict[str, Any]]:
    """Ejecuta el sistema multi-agente completo con modelo override.

    Args:
        case: Caso de evaluación.
        model_id: ID del modelo a usar (ej: 'gpt-5-mini').
        model_provider: Proveedor explícito del modelo evaluado.
        vision_model_id: ID del modelo de visión (opcional, para vision_ocr).
        vision_provider: Proveedor del modelo de visión.
        tracker: Tracker de métricas LLM.
        query_override: Query a usar en vez de case.query (para multi-turn).
        conversation_id: ID de conversación para mantener contexto entre turns.

    Returns:
        Tupla (NormalizedOutput, execution_report_dict) donde
        execution_report_dict contiene qué agentes se invocaron y su estado.
    """
    if not llm_enabled():
        return (
            NormalizedOutput(
                executive_summary="La evaluación requiere EVALUATION_ENABLE_LLM=1.",
                parse_status="failed",
            ),
            {"error": "evaluation_llm_disabled", "agents": {}, "final_level": 0},
        )
    if model_provider and not provider_enabled(model_provider):
        return (
            NormalizedOutput(
                executive_summary=f"No hay credenciales para el proveedor de evaluación '{model_provider}'.",
                parse_status="failed",
            ),
            {"error": "evaluation_provider_unavailable", "agents": {}, "final_level": 0},
        )

    try:
        from agents.organizer import OrganizerAgent
        from agents.base import _build_client
        from agents.legal import LegalAgent
        from agents.writer import DirectResponseWriterAgent
        from agents.case_manager import CaseManagerAgent
        from agents.stac_search import StacSearchAgent
        from agents.rs_analyst import RSAnalystAgent
        from agents.document_analyst import DocumentAnalystAgent
        from agents.spreadsheet_analyst import SpreadsheetAnalystAgent
        from agents.vision_ocr import VisionOcrAgent
        from agents.free import FreeAgent
        from backend.services.chat_orchestrator import ChatOrchestratorService
        from backend.events import EventBroker

        agents = {
            "organizer": OrganizerAgent(),
            "legal": LegalAgent(),
            "case_manager": CaseManagerAgent(),
            "stac": StacSearchAgent(),
            "rs_analyst": RSAnalystAgent(),
            "direct_writer": DirectResponseWriterAgent(),
            "report_writer": DirectResponseWriterAgent(),
            "document_analyst": DocumentAnalystAgent(),
            "spreadsheet_analyst": SpreadsheetAnalystAgent(),
            "vision_ocr": VisionOcrAgent(),
            "free": FreeAgent(),
        }
        if model_id and model_provider:
            for agent in agents.values():
                _configure_evaluation_agent(
                    agent,
                    model_id=model_id,
                    provider=model_provider,
                    vision_model_id=vision_model_id,
                    vision_provider=vision_provider,
                    temperature=model_temperature,
                    max_tokens=model_max_tokens,
                )
            # OrganizerAgent owns an eager client rather than BaseAgent's lazy client.
            agents["organizer"].client = _build_client(model_provider)

        broker = EventBroker()
        service = ChatOrchestratorService(agents=agents, broker=broker)

        # Preparar attachments
        attachments = _build_attachments(case)

        # Usar query_override para multi-turn, o case.query para single-turn
        query = query_override if query_override is not None else case.query

        capture_token = start_cost_capture()
        try:
            result = await service.execute(
                query=query,
                language="es",
                user_id=case.context.user_role or "eval_user",
                decision_mode="case",
                response_mode="conversation",
                memory_enabled=False,
                # El corpus ya proporciona la identidad y el historial del caso.
                # Desactivar la resolución automática evita que casos de otros
                # experimentos del mismo workspace provoquen una aclaración
                # espuria antes de que el organizador pueda planificar.
                continuity_mode="off",
                attachments=attachments if attachments else None,
                conversation_id=conversation_id,
            )
        finally:
            _record_captured_costs(tracker, finish_cost_capture(capture_token))

        # Extraer execution_report del resultado
        execution_report = _extract_execution_report(result)
        execution_report["evaluation_model"] = model_id
        execution_report["evaluation_provider"] = model_provider

        # Older/current conversation payloads may expose the vision agent
        # state without serializing the derived visual_evidence block. Keep
        # the evaluation contract explicit in that case instead of making the
        # metric layer infer it from an absent field.
        if case.attachments and "visual_evidence" not in execution_report:
            vision_state = (execution_report.get("agents") or {}).get("vision_ocr", {})
            final_level = vision_state.get("final_level") if isinstance(vision_state, dict) else None
            execution_report["visual_evidence"] = {
                "status": "available" if final_level == "ok" else "insufficient",
                "signals": [],
                "used_for": [],
                "confidence": [],
                "limitations": [],
                "used_in_final": False,
                "source": "vision_agent_state_fallback",
            }

        answer = result.get("answer", {})
        execution_error = execution_report.get("error")
        agent_states = execution_report.get("agents", {})
        has_hard_failure = any(
            isinstance(state, dict)
            and state.get("final_level") == "hard_error"
            for state in agent_states.values()
        ) if isinstance(agent_states, dict) else False
        has_soft_failure = any(
            isinstance(state, dict)
            and state.get("final_level") == "soft_error"
            for state in agent_states.values()
        ) if isinstance(agent_states, dict) else False
        visible_text = _as_text(answer.get("message_md") or answer.get("report_md") or answer.get("executive_summary")) if isinstance(answer, dict) else _as_text(answer)
        invalid_output = not visible_text.strip()
        if isinstance(answer, dict):
            output = NormalizedOutput(
                executive_summary=_as_text(answer.get("executive_summary")),
                report_text=_as_text(answer.get("report_md")),
                message_md=_as_text(answer.get("message_md")),
                evidence_summary=_as_list(answer.get("evidence_summary")),
                next_actions=_as_list(answer.get("next_actions")),
                missing_information=_as_list(answer.get("missing_information")),
                documents_needed=_as_list(answer.get("documents_needed")),
                limitations=_as_list(answer.get("limitations")),
                references=_as_list(answer.get("references")),
                structured_fields_present=[
                    field for field in (
                        "executive_summary", "report_md", "message_md", "evidence_summary",
                        "next_actions", "missing_information", "documents_needed", "limitations",
                        "references",
                    ) if field in answer
                ],
                parse_status=(
                    "failed" if execution_error or has_hard_failure or invalid_output
                    else "partial" if has_soft_failure else "ok"
                ),
            )
        else:
            output = NormalizedOutput(
                executive_summary=_as_text(answer)[:500],
                report_text=_as_text(answer),
                parse_status="failed" if execution_error or invalid_output else "ok",
            )

        return output, execution_report

    except Exception as exc:
        error_report = {
            "error": str(exc),
            "agents": {},
            "final_level": 0,
        }
        return (
            NormalizedOutput(
                executive_summary=f"Error en el sistema: {exc}",
                report_text=f"Error: {exc}",
                parse_status="failed",
            ),
            error_report,
        )
