from __future__ import annotations

import asyncio
import datetime as dt
import uuid
from dataclasses import dataclass
from typing import Any

from loguru import logger

from backend.deps import settings
from backend.cost_store import cost_store
from backend.conversation_store import conversation_store
from backend.case_store import case_store
from backend.continuity import CaseResolution, resolve_case, should_create_case
from backend.events import EventBroker, write_trace_report
from backend.memory_reuse import build_remote_sensing_artifact, resolve_memory_reuse_state
from backend.memory_store import memory_store
from libs.costs.tracker import current_conversation_id
from libs.meteo import fetch_meteo_context_async
from libs.query_rewriter import rewrite_query
from libs.schemas import (
    AgentInput,
    AgentPlan,
    ClarificationOption,
    ContextUsage,
    CostSummary,
    ContinuitySummary,
    FinalAnswer,
    MemoryUsage,
    RSAnalysisConfig,
    StacResults,
)
from libs.content_blocks import auto_generate_blocks, parse_block_markers, resolve_citations


def _now_ts() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _normalize_iso_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return value[:10] if len(value) >= 10 else None


_AGENT_TOOLS: dict[str, str] = {
    "legal": "Legalize RAG; conditional web_search for official sources/currency",
    "case_manager": "consolidated context and memory; no external tools",
    "stac": "geocode_place, search_satellite_images, inspect_region",
    "rs_analyst": "deterministic StacResults analysis; no external tools",
    "document_analyst": "local PDF/DOC/DOCX/TXT extraction + LLM enrichment",
    "spreadsheet_analyst": "local CSV/XLS/XLSX profiling + LLM enrichment",
    "vision_ocr": "local PNG/JPG/TIF OCR + LLM enrichment",
    "free": "web_search for general research; LLM reasoning on assigned tasks",
    "writer": "single-agent fast path with targeted web search and final synthesis; no specialized tools",
    "direct_writer": "direct response with targeted web search; no specialized tools",
    "organizer": "planning and delegation; no external tools",
}


def _build_context_summary(agent: str, user_query: Any, memory_context: str, steps: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "query": (user_query.query or "")[:200] if user_query else "",
        "has_memory": bool(memory_context),
        "has_attachments": bool(getattr(user_query, "attachments", None)),
    }
    if agent == "rs_analyst":
        summary["input_agents"] = [s for s in ["stac"] if s in steps]
    elif agent == "case_manager":
        summary["input_agents"] = [
            s for s in ["legal", "stac", "rs_analyst", "document_analyst", "spreadsheet_analyst", "vision_ocr"]
            if s in steps
        ]
    elif agent in {"writer", "direct_writer"}:
        summary["input_agents"] = [s for s in steps if s != "writer" and s != agent]
    return summary


@dataclass
class ExecutionAssessment:
    level: str
    message: str


class ChatOrchestratorService:
    def __init__(self, *, agents: dict[str, Any], broker: EventBroker) -> None:
        self.agents = agents
        self.broker = broker

    @staticmethod
    def _apply_memory_reuse_guard(plan: AgentPlan, memory_reuse: Any) -> AgentPlan:
        raw = memory_reuse.model_dump(exclude_none=True) if hasattr(memory_reuse, "model_dump") else memory_reuse
        rs = raw.get("remote_sensing") if isinstance(raw, dict) else None
        if not isinstance(rs, dict) or str(rs.get("status") or "miss") != "hit":
            return plan
        filtered_steps = [step for step in plan.steps if step not in {"stac", "rs_analyst"}]
        if filtered_steps == plan.steps:
            return plan
        plan.steps = filtered_steps or ["writer"]
        plan.runs = {step: count for step, count in dict(plan.runs or {}).items() if step in plan.steps}
        plan.dependencies = {
            step: [dep for dep in deps if dep in plan.steps]
            for step, deps in dict(plan.dependencies or {}).items()
            if step in plan.steps
        }
        plan.diagnostics.rationale = (
            (plan.diagnostics.rationale + " ") if plan.diagnostics.rationale else ""
        ) + "Se reutiliza evidencia remota valida de memoria y se omiten STAC/RS redundantes."
        return plan

    async def execute(
        self,
        *,
        query: str,
        language: str = "es",
        conversation_id: str | None = None,
        user_id: str | None = None,
        decision_mode: str = "case",
        response_mode: str = "conversation",
        memory_enabled: bool = False,
        continuity_mode: str = "auto",
        attachments: list[Any] | None = None,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        attachments = attachments or []
        response_mode = "conversation"
        provided_id = (conversation_id or "").strip() or None
        resolved_conversation_id = provided_id or str(uuid.uuid4())
        resolved_case_id = (case_id or "").strip() or None
        workspace_id = (user_id or "local").strip() or "local"
        existing_conversation = conversation_store.get_conversation(resolved_conversation_id)
        linked_case_id = (
            existing_conversation.get("case_id") if existing_conversation else None
        )
        continuity_resolution = resolve_case(
            case_store.list_cases(workspace_id=workspace_id, limit=50)
            if continuity_mode != "off"
            else [],
            query,
            explicit_case_id=resolved_case_id,
            linked_case_id=linked_case_id,
        )
        if resolved_case_id is None:
            resolved_case_id = continuity_resolution.case_id
        if case_id and resolved_case_id is None:
            case_store.get_case(case_id, workspace_id=workspace_id)
            resolved_case_id = case_id
            continuity_resolution = CaseResolution(case_id, "explicit")
        case_context = ""
        case_context_usage: dict[str, Any] = {"case_id": None, "context_run_id": None, "items": []}
        if resolved_case_id:
            case_store.get_case(resolved_case_id, workspace_id=workspace_id)
            case_context_usage = case_store.build_context(
                case_id=resolved_case_id,
                workspace_id=workspace_id,
                query=query,
                conversation_id=resolved_conversation_id,
            )
            case_context = str(case_context_usage.get("text") or "")
        current_conversation_id.set(resolved_conversation_id)

        conversation_history: list[dict[str, Any]] = []
        if resolved_conversation_id:
            existing = conversation_store.get_conversation(resolved_conversation_id)
            if not existing:
                title = " ".join(query.strip().split()[:8])
                if len(query.strip()) > len(title):
                    title += "..."
                conversation_store.save_conversation(
                    resolved_conversation_id,
                    user_id=user_id or "",
                    title=title,
                    case_id=resolved_case_id,
                )
            elif resolved_case_id:
                conversation_store.link_case(resolved_conversation_id, resolved_case_id)
            if resolved_case_id:
                case_store.rebind_conversation(case_id=resolved_case_id, conversation_id=resolved_conversation_id, workspace_id=workspace_id)
            conversation_history = conversation_store.get_messages(
                resolved_conversation_id, limit=10
            )
            file_names = [a.filename for a in attachments] if attachments else []
            conversation_store.save_message(
                resolved_conversation_id,
                role="user",
                query=query,
                response_mode=response_mode,
                file_names=file_names,
            )

        memory_context = ""
        memory_used_sections: list[str] = []
        memory_meta: Any = None
        remote_sensing_artifacts: list[Any] = []
        memory_reuse = None
        if continuity_mode != "off" and user_id:
            memory = memory_store.load(user_id, include_legacy=False)
            memory_meta = memory
            memory_context = memory_store.render_context(user_id)
            memory_used_sections = memory.used_sections
            remote_sensing_artifacts = memory_store.load_remote_sensing_artifacts(user_id)
            memory_reuse = resolve_memory_reuse_state(
                query=query,
                decision_mode=decision_mode,
                observations=[],
                remote_sensing_artifacts=remote_sensing_artifacts,
            )

        if conversation_history and len(conversation_history) >= 2:
            query = await rewrite_query(query, conversation_history)

        user_query = AgentInput(
            query=query,
            language=language or "es",
            attachments=attachments,
            context={
                **({"user_memory": memory_context} if memory_context else {}),
                "case_context": case_context,
                "case_id": resolved_case_id,
                "_conversation_history": conversation_history,
                "_memory_reuse": memory_reuse.model_dump(exclude_none=True) if memory_reuse else {},
            },
            user_id=user_id,
            decision_mode=decision_mode,
            response_mode=response_mode,
            memory_enabled=memory_enabled,
        )
        log = logger.bind(conversation_id=resolved_conversation_id)

        async def emit(event: dict) -> None:
            enriched = dict(event)
            enriched.setdefault("ts", _now_ts())
            payload = {"conversation_id": resolved_conversation_id, **enriched}
            await self.broker.publish(resolved_conversation_id, payload)

        async def log_event(message: str, *, level: str = "INFO", agent: str | None = None) -> None:
            ts = _now_ts()
            await emit(
                {
                    "type": "log",
                    "level": level,
                    "agent": agent,
                    "actor": agent or "system",
                    "timestamp": ts,
                    "ts": ts,
                    "message": message,
                }
            )

        organizer_agent = self.agents["organizer"]
        writer_agent_name = "direct_writer" if "direct_writer" in self.agents else "writer"
        internal_agents = {"organizer", "writer", "direct_writer"}
        available_agents = {name for name in self.agents if name not in internal_agents} | {"writer"}

        await emit(
            {
                "type": "status",
                "stage": "received",
                "message": "Consulta recibida",
                "actor": "system",
            }
        )
        await log_event("Consulta recibida")
        if continuity_mode != "off" and user_id:
            await log_event(
                f"Contexto heredado disponible ({len(memory_used_sections)} secciones con contenido)",
                agent="system",
            )
            if memory_reuse is not None:
                await log_event(
                    f"Evidencia remota heredada: {memory_reuse.remote_sensing.status}",
                    agent="system",
                )

        if continuity_resolution.reason == "ambiguous":
            cases_by_id = {
                item["case_id"]: item
                for item in case_store.list_cases(workspace_id=workspace_id, limit=100)
            }
            options = [
                ClarificationOption(
                    key=case_id,
                    label=cases_by_id.get(case_id, {}).get("title", case_id),
                    description=(
                        cases_by_id.get(case_id, {}).get("summary")
                        or "Seguimiento activo"
                    ),
                    enriched_query=query,
                )
                for case_id in continuity_resolution.candidates
            ]
            question = "¿Con cuál seguimiento quieres continuar?"
            clarification = {
                "question": question,
                "options": [option.model_dump() for option in options],
                "rationale": "Hay varios seguimientos activos que podrían corresponder a esta consulta.",
            }
            answer = FinalAnswer(
                executive_summary=question,
                message_md=question,
                response_path="multi_agent_synthesis",
                continuity=ContinuitySummary(
                    status="ambiguous",
                    candidates=continuity_resolution.candidates,
                ),
            )
            await emit(
                {
                    "type": "status",
                    "stage": "completed",
                    "message": "Seguimiento por confirmar",
                    "actor": "system",
                }
            )
            await log_event("Seguimiento ambiguo; se solicita una única aclaración", agent="system")
            return {
                "plan": {
                    "steps": [],
                    "runs": {},
                    "dependencies": {},
                    "allow_replan": False,
                    "writer_mode": None,
                    "writer_agent": None,
                    "response_mode": response_mode,
                },
                "answer": answer.model_dump(),
                "conversation_id": resolved_conversation_id,
                "clarification": clarification,
            }

        await emit(
            {
                "type": "agent_status",
                "agent": "organizer",
                "actor": "organizer",
                "status": "running",
                "message": "Organizer started",
            }
        )
        await log_event("Organizer started", agent="organizer")

        plan = await organizer_agent.plan(user_query)
        plan = self._apply_memory_reuse_guard(plan, memory_reuse)

        # Si el planner devolvió una solicitud de clarificación, retornarla sin ejecutar agentes
        if plan.clarification:
            clarification_payload = plan.clarification.model_dump()
            await emit(
                {
                    "type": "status",
                    "stage": "completed",
                    "message": "Clarificación solicitada",
                    "actor": "organizer",
                }
            )
            await log_event("Clarificación solicitada al usuario", agent="organizer")
            return {
                "plan": {
                    "steps": [],
                    "runs": {},
                    "dependencies": {},
                    "allow_replan": False,
                    "writer_mode": None,
                    "writer_agent": None,
                    "response_mode": response_mode,
                    "policy": plan.policy.model_dump(),
                    "diagnostics": plan.diagnostics.model_dump(),
                },
                "answer": {
                    "executive_summary": plan.clarification.question,
                    "message_md": plan.clarification.question,
                    "response_path": "multi_agent_synthesis",
                    "search_used": False,
                    "escalation_required": False,
                    "recommendations": [],
                    "limitations": [],
                    "next_actions": [],
                    "evidence_summary": [],
                    "missing_information": [],
                    "documents_needed": [],
                    "references": [],
                    "execution": {},
                    "attachments": [],
                    "evidence_ledger": {},
                },
                "conversation_id": resolved_conversation_id,
                "clarification": clarification_payload,
            }

        steps = plan.steps or []
        retry_candidates = {
            item for item in (plan.policy.retry_candidates or []) if item in self.agents
        }
        unknown_steps = [step for step in steps if step not in available_agents]
        if unknown_steps:
            message = (
                f"Plan invalido: agentes no registrados: {', '.join(sorted(set(unknown_steps)))}"
            )
            await emit({"type": "error", "actor": "organizer", "error": message})
            await log_event(message, level="ERROR", agent="organizer")
            raise RuntimeError(message)
        plan_runs = dict(plan.runs or {})
        for step in steps:
            plan_runs.setdefault(step, 1)
        plan_runs.setdefault("writer", 1)
        plan.runs = {name: 1 for name in plan_runs}
        dependencies = {
            step: [dep for dep in deps if dep in steps and dep != step]
            for step, deps in dict(plan.dependencies or {}).items()
            if step in steps
        }
        if "rs_analyst" in steps and "stac" in steps:
            dependencies.setdefault("rs_analyst", ["stac"])
        if "case_manager" in steps:
            dependencies.setdefault(
                "case_manager",
                [
                    dep
                    for dep in [
                        "legal",
                        "stac",
                        "rs_analyst",
                        "document_analyst",
                        "spreadsheet_analyst",
                        "vision_ocr",
                    ]
                    if dep in steps
                ],
            )
        if "writer" in steps:
            dependencies.setdefault("writer", [step for step in steps if step != "writer"])

        await emit(
            {
                "type": "plan",
                "actor": "organizer",
                "steps": steps,
                "agents": steps,
                "runs": plan_runs,
                "dependencies": dependencies,
                "policy": plan.policy.model_dump(),
                "diagnostics": plan.diagnostics.model_dump(),
            }
        )
        if plan.diagnostics.fallback_reason:
            await log_event(
                f"Planner fallback activado: {plan.diagnostics.fallback_reason}",
                agent="organizer",
            )
        await emit(
            {
                "type": "agent_status",
                "agent": "organizer",
                "actor": "organizer",
                "status": "done",
                "message": "Organizer completed plan",
            }
        )
        await log_event("Organizer completed planning", agent="organizer")

        def total_instances(agent: str) -> int:
            return 1

        def run_key(agent: str, instance_id: int) -> str:
            return f"{agent}#{instance_id}"

        async def emit_status(
            agent: str,
            instance_id: int,
            total_runs: int,
            attempt: int,
            limit: int,
            *,
            status: str,
            message: str,
        ) -> None:
            await emit(
                {
                    "type": "agent_status",
                    "agent": agent,
                    "actor": agent,
                    "status": status,
                    "message": message,
                    "run_id": instance_id,
                    "total_runs": total_runs,
                    "attempt": attempt,
                    "attempt_limit": limit,
                    "run_key": run_key(agent, instance_id),
                }
            )

        agent_results: dict[str, list[object | None]] = {}
        agent_failures: list[str] = []
        execution_report: dict[str, dict[str, Any]] = {}

        def record_result(agent: str, instance_id: int, value: object | None) -> None:
            bucket = agent_results.setdefault(agent, [])
            while len(bucket) < instance_id:
                bucket.append(None)
            bucket[instance_id - 1] = value

        def record_failure(message: str) -> None:
            if message not in agent_failures:
                agent_failures.append(message)

        def professional_failure_limitation(message: str) -> str | None:
            normalized = str(message or "").strip().lower()
            if not normalized:
                return None
            if "stac" in normalized or "rs_analyst" in normalized or "rs disabled" in normalized:
                return (
                    "No hay analisis temporal de teledeteccion disponible con evidencia suficiente; "
                    "las decisiones de campo deben apoyarse en observacion directa o datos recientes."
                )
            if "vision_ocr" in normalized or "ocr" in normalized:
                return (
                    "La lectura automatica de adjuntos no aporta evidencia suficiente; conviene validar "
                    "los originales antes de cerrar la decision."
                )
            if "legal" in normalized or "document" in normalized:
                return (
                    "La evidencia documental o normativa es incompleta; no debe asumirse cumplimiento "
                    "sin verificar los documentos concretos indicados."
                )
            if "case_manager" in normalized:
                return None
            return (
                "Una fuente auxiliar no aporto evidencia suficiente; la recomendacion queda limitada "
                "a los datos disponibles en el caso."
            )

        def classify_result(
            agent: str, instance_id: int, result: object | None
        ) -> ExecutionAssessment:
            if result is None:
                return ExecutionAssessment("hard_error", f"{agent}#{instance_id}: sin resultado")
            status = getattr(result, "status", "ok")
            summary = (getattr(result, "summary", "") or "").strip()
            errors = [
                str(item).strip()
                for item in (getattr(result, "errors", None) or [])
                if str(item).strip()
            ]
            data = getattr(result, "data", None)
            if status == "error":
                detail = summary or (errors[0] if errors else "resultado marcado como error")
                return ExecutionAssessment("soft_error", f"{agent}#{instance_id}: {detail}")
            if data is None:
                detail = summary or "sin datos utiles"
                return ExecutionAssessment("insufficient_data", f"{agent}#{instance_id}: {detail}")
            return ExecutionAssessment("ok", f"{agent}#{instance_id}: completado")

        def store_execution_state(
            agent: str, instance_id: int, assessment: ExecutionAssessment
        ) -> None:
            agent_state = execution_report.setdefault(agent, {"instances": [], "final_level": "ok"})
            agent_state["instances"].append(
                {
                    "instance_id": instance_id,
                    "level": assessment.level,
                    "message": assessment.message,
                }
            )
            priority = {"ok": 0, "insufficient_data": 1, "soft_error": 2, "hard_error": 3}
            current = agent_state.get("final_level", "ok")
            if priority.get(assessment.level, 0) >= priority.get(current, 0):
                agent_state["final_level"] = assessment.level

        def latest_result(agent: str) -> object | None:
            bucket = agent_results.get(agent)
            if not bucket:
                return None
            for item in reversed(bucket):
                if item is not None:
                    return item
            return None

        meteo_cache: dict[tuple[float, float, float, float, str, str], object | None] = {}

        async def _fetch_meteo_for_stac(stac_results: StacResults, ctx: dict[str, Any]) -> object:
            items_with_bbox = [i for i in stac_results.items if i.bbox and len(i.bbox) == 4]
            bbox = items_with_bbox[0].bbox if items_with_bbox else None
            if not bbox:
                return None
            datetimes = sorted(
                [i.datetime for i in stac_results.items if i.datetime],
            )
            start = _normalize_iso_date(datetimes[0] if datetimes else None)
            end = _normalize_iso_date(datetimes[-1] if datetimes else None)
            if not start or not end:
                return None
            cache_key = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), start, end)
            if cache_key in meteo_cache:
                return meteo_cache[cache_key]
            try:
                meteo_cache[cache_key] = await fetch_meteo_context_async(
                    bbox=bbox,
                    start_date=start,
                    end_date=end,
                )
                return meteo_cache[cache_key]
            except Exception as exc:
                log.bind(agent="rs_analyst").warning("Meteo fetch failed: {}", exc)
                meteo_cache[cache_key] = None
                return None

        _missions_dict = {m.agent: m.instruction for m in plan.missions}

        async def run_step(name: str) -> object | None:
            if name == "rs_analyst":
                stac = latest_result("stac") or StacResults(items=[])
                stac_results = getattr(stac, "data", stac)
                if not isinstance(stac_results, StacResults):
                    stac_results = StacResults(items=[])
                crop_type_ctx = (user_query.context or {}).get("crop_type")
                growth_stage_ctx = (user_query.context or {}).get("growth_stage")
                meteo = await _fetch_meteo_for_stac(stac_results, user_query.context)
                config_raw = (user_query.context or {}).get("rs_config")
                rs_config = config_raw if isinstance(config_raw, RSAnalysisConfig) else RSAnalysisConfig()
                rs_input = AgentInput(
                    query=user_query.query,
                    language=user_query.language,
                    attachments=user_query.attachments,
                    context={
                        "mission": _missions_dict.get("rs_analyst", ""),
                        "stac_results": stac_results,
                        "user_memory": memory_context,
                        "case_context": case_context,
                        "crop_type": crop_type_ctx,
                        "growth_stage": growth_stage_ctx,
                        "meteo": meteo,
                        "rs_config": rs_config,
                    },
                    user_id=user_query.user_id,
                    decision_mode=user_query.decision_mode,
                    response_mode=user_query.response_mode,
                    memory_enabled=user_query.memory_enabled,
                )
                return await self.agents["rs_analyst"].run(rs_input)
            if name == "free":
                task_parts = [
                    f"Consulta del usuario: {user_query.query}",
                    f"Razon del planner: {plan.diagnostics.rationale or 'Investigacion general solicitada.'}",
                ]
                if memory_context:
                    task_parts.append(f"Memoria del usuario: {memory_context[:400]}")
                if case_context:
                    task_parts.append(f"Contexto del caso: {case_context[:300]}")
                if conversation_history:
                    recent = conversation_history[-4:]
                    hist_lines = [
                        f"- {str(msg.get('role', ''))}: {str(msg.get('content', ''))[:150]}"
                        for msg in recent
                        if msg.get('content')
                    ]
                    if hist_lines:
                        task_parts.append("Historial reciente:\n" + "\n".join(hist_lines))
                free_input = AgentInput(
                    query=user_query.query,
                    language=user_query.language,
                    attachments=user_query.attachments,
                    context={
                        "task_description": "\n\n".join(task_parts),
                        "user_memory": memory_context,
                        "vision_ocr": latest_result("vision_ocr"),
                        "_conversation_history": conversation_history,
                    },
                    user_id=user_query.user_id,
                    decision_mode=user_query.decision_mode,
                    response_mode=user_query.response_mode,
                    memory_enabled=user_query.memory_enabled,
                )
                return await self.agents["free"].run(free_input)
            if name in {"case_manager"}:
                contextual_input = AgentInput(
                    query=user_query.query,
                    language=user_query.language,
                    attachments=user_query.attachments,
                    context={
                        "user_memory": memory_context,
                        "legal": latest_result("legal"),
                        "document_analyst": latest_result("document_analyst"),
                        "spreadsheet_analyst": latest_result("spreadsheet_analyst"),
                        "vision_ocr": latest_result("vision_ocr"),
                        "case_manager": latest_result("case_manager"),
                        "rs_analyst": latest_result("rs_analyst"),
                        "stac": latest_result("stac"),
                        "case_context": case_context,
                        "case_id": resolved_case_id,
                        "context_usage": case_context_usage,
                        "_conversation_history": conversation_history,
                        "_memory_reuse": memory_reuse.model_dump(exclude_none=True) if memory_reuse else {},
                        "_execution": execution_report,
                    },
                    user_id=user_query.user_id,
                    decision_mode=user_query.decision_mode,
                    response_mode=user_query.response_mode,
                    memory_enabled=user_query.memory_enabled,
                )
                return await self.agents[name].run(contextual_input)
            if name in self.agents and name not in internal_agents:
                mission = _missions_dict.get(name, "")
                if mission:
                    step_input = AgentInput(
                        query=user_query.query,
                        language=user_query.language,
                        attachments=user_query.attachments,
                        context={
                            **(user_query.context or {}),
                            "mission": mission,
                        },
                        user_id=user_query.user_id,
                        decision_mode=user_query.decision_mode,
                        response_mode=user_query.response_mode,
                        memory_enabled=user_query.memory_enabled,
                    )
                    return await self.agents[name].run(step_input)
                return await self.agents[name].run(user_query)
            return None

        semaphore = asyncio.Semaphore(settings.AGENT_CONCURRENCY_LIMIT)

        async def execute_instance(agent: str, instance_id: int, total_runs: int) -> None:
            attempt = 1
            limit = 1 + plan.policy.max_rounds if (
                plan.policy.allow_retries and agent in retry_candidates
            ) else 1
            while True:
                await emit_status(
                    agent,
                    instance_id,
                    total_runs,
                    attempt,
                    limit,
                    status="running",
                    message=f"Agente {agent} en ejecucion",
                )
                await log_event(
                    f"Iniciando agente {agent} (instancia {instance_id}, intento {attempt})",
                    agent=agent,
                )
                log.bind(agent=agent).info("Agent started", instance=instance_id, attempt=attempt)
                try:
                    async with semaphore:
                        result = await run_step(agent)
                except Exception as exc:  # pragma: no cover
                    assessment = ExecutionAssessment("hard_error", f"{agent}#{instance_id}: {exc}")
                    store_execution_state(agent, instance_id, assessment)
                    if attempt < limit:
                        await emit_status(
                            agent,
                            instance_id,
                            total_runs,
                            attempt,
                            limit,
                            status="retrying",
                            message=f"Reintentando {agent} tras error",
                        )
                        await log_event(
                            f"Reintento de {agent} tras error: {exc}",
                            agent=agent,
                            level="WARNING",
                        )
                        attempt += 1
                        continue
                    record_result(agent, instance_id, None)
                    record_failure(assessment.message)
                    await emit_status(
                        agent,
                        instance_id,
                        total_runs,
                        attempt,
                        limit,
                        status="error",
                        message=str(exc),
                    )
                    await log_event(f"Error en {agent}: {exc}", agent=agent, level="ERROR")
                    log.bind(agent=agent).exception("Agent failed", instance=instance_id)
                    return

                record_result(agent, instance_id, result)
                assessment = classify_result(agent, instance_id, result)
                store_execution_state(agent, instance_id, assessment)
                if assessment.level in {"soft_error", "hard_error"} and attempt < limit:
                    await emit_status(
                        agent,
                        instance_id,
                        total_runs,
                        attempt,
                        limit,
                        status="retrying",
                        message=f"Reintentando {agent} tras salida degradada",
                    )
                    await log_event(
                        f"Reintento de {agent} tras salida degradada: {assessment.message}",
                        agent=agent,
                        level="WARNING",
                    )
                    attempt += 1
                    continue
                if assessment.level != "ok":
                    record_failure(assessment.message)
                await emit_status(
                    agent,
                    instance_id,
                    total_runs,
                    attempt,
                    limit,
                    status="done",
                    message=f"Agente {agent} completado (instancia {instance_id})",
                )
                await log_event(
                    f"Agente {agent} completado (instancia {instance_id}, intento {attempt})",
                    agent=agent,
                )
                log.bind(agent=agent).info("Agent finished", instance=instance_id, attempt=attempt)

                agent_obj = self.agents.get(agent)
                trace_data = None
                trace_obj = getattr(result, "trace", None) if result else None
                if trace_obj is not None:
                    trace_data = {
                        "duration_ms": getattr(trace_obj, "duration_ms", 0) or 0,
                        "tokens_input": getattr(trace_obj, "tokens_input", 0) or 0,
                        "tokens_output": getattr(trace_obj, "tokens_output", 0) or 0,
                        "cost_usd": getattr(trace_obj, "cost_usd", 0) or 0,
                    }
                await emit({
                    "type": "agent_detail",
                    "agent": agent,
                    "run_key": f"{agent}#{instance_id}",
                    "model": getattr(agent_obj, "model", None) if agent_obj else None,
                    "provider": getattr(agent_obj, "provider", None) if agent_obj else None,
                    "mission": _missions_dict.get(agent, ""),
                    "tools_available": _AGENT_TOOLS.get(agent, "no tools declared"),
                    "tools_used": [],
                    "context_summary": _build_context_summary(agent, user_query, memory_context, steps),
                    "output_preview": (getattr(result, "summary", "") or "")[:300] if result else "",
                    "execution_level": assessment.level,
                    "trace": trace_data,
                })

                break

        async def execute_agent(agent: str) -> None:
            total = total_instances(agent)
            tasks = [
                execute_instance(agent, instance_id, total) for instance_id in range(1, total + 1)
            ]
            await asyncio.gather(*tasks)

        for step in steps:
            visible_step = writer_agent_name if step == "writer" else step
            total = total_instances(step)
            for instance_id in range(1, total + 1):
                await emit_status(
                    visible_step,
                    instance_id,
                    total,
                    1,
                    1,
                    status="queued",
                    message="Agente preparado",
                )

        completed: set[str] = set()
        pending_steps = [s for s in steps if s != "writer"]

        while pending_steps:
            ready = [
                step
                for step in pending_steps
                if all(dep in completed or dep not in pending_steps for dep in dependencies.get(step, []))
            ]
            if not ready:
                blocked = ", ".join(sorted(pending_steps))
                message = f"Plan bloqueado: dependencias sin resolver para {blocked}"
                await emit({"type": "error", "actor": "organizer", "error": message})
                await log_event(message, level="ERROR", agent="organizer")
                raise RuntimeError(message)
            await asyncio.gather(
                *[
                    execute_agent(step)
                    for step in ready
                    if step in self.agents
                ]
            )
            completed.update(step for step in ready if step in self.agents)
            pending_steps = [step for step in pending_steps if step not in ready]

        replan_result = {
            "attempted": False,
            "applied": False,
            "extra_steps": [],
            "diagnostics": None,
        }
        has_critical_failures = any(
            info.get("final_level") in {"hard_error", "insufficient_data"}
            for info in execution_report.values()
        )
        replan_cost_limit = settings.REPLAN_MAX_COST_USD if hasattr(settings, "REPLAN_MAX_COST_USD") else 2.0
        current_cost = cost_store.summarize_conversation(resolved_conversation_id).get("total_cost_usd", 0.0)
        should_replan = plan.allow_replan and current_cost < replan_cost_limit and hasattr(organizer_agent, "replan")
        if should_replan:
            replan_result["attempted"] = True
            replan_context = {
                agent: {
                    "summary": getattr(latest_result(agent), "summary", ""),
                    "execution": execution_report.get(agent),
                }
                for agent in steps
                if agent not in {"writer"}
            }
            extra_plan = await organizer_agent.replan(user_query, replan_context)
            extra_plan = self._apply_memory_reuse_guard(extra_plan, memory_reuse)
            replan_result["diagnostics"] = extra_plan.diagnostics.model_dump()
            extra_steps = [
                step
                for step in (extra_plan.steps or [])
                if step not in completed and step not in {"writer"} and step in available_agents
            ]
            replan_result["extra_steps"] = list(extra_steps)
            await emit(
                {
                    "type": "replan",
                    "actor": "organizer",
                    "attempted": True,
                    "applied": bool(extra_steps),
                    "extra_steps": list(extra_steps),
                    "diagnostics": extra_plan.diagnostics.model_dump(),
                }
            )
            await log_event(
                (
                    f"Replan aplicado con pasos extra: {', '.join(extra_steps)}"
                    if extra_steps
                    else "Replan ejecutado sin pasos adicionales"
                ),
                agent="organizer",
            )
            if extra_steps:
                replan_result["applied"] = True
                extra_dependencies = {
                    step: [dep for dep in deps if dep in completed or dep in extra_steps]
                    for step, deps in dict(extra_plan.dependencies or {}).items()
                    if step in extra_steps
                }
                steps = [step for step in steps if step != "writer"] + extra_steps + ["writer"]
                dependencies.update(extra_dependencies)
                dependencies["writer"] = [step for step in steps if step != "writer"]
                for step in extra_steps:
                    plan_runs.setdefault(step, 1)
                    await emit_status(step, 1, 1, 1, 1, status="queued", message="Agente anadido tras replanteo")
                await emit(
                    {
                        "type": "plan",
                        "actor": "organizer",
                        "steps": steps,
                        "agents": steps,
                        "runs": plan_runs,
                        "dependencies": dependencies,
                        "replanned": True,
                        "policy": plan.policy.model_dump(),
                        "diagnostics": extra_plan.diagnostics.model_dump(),
                    }
                )
                pending_extra = list(extra_steps)
                while pending_extra:
                    ready_extra = [
                        step
                        for step in pending_extra
                        if all(dep in completed or dep not in pending_extra for dep in dependencies.get(step, []))
                    ]
                    if not ready_extra:
                        break
                    await asyncio.gather(*(execute_agent(step) for step in ready_extra))
                    completed.update(ready_extra)
                    pending_extra = [step for step in pending_extra if step not in ready_extra]
                writer_mode = extra_plan.writer_mode or plan.writer_mode
                plan.writer_mode = writer_mode

        vision_result = latest_result("vision_ocr")
        vision_data = getattr(vision_result, "data", None) if vision_result else None
        vision_images = vision_data.get("images", []) if isinstance(vision_data, dict) else []
        visual_signals = [
            str(signal).strip()
            for image in vision_images
            if isinstance(image, dict)
            for signal in (image.get("key_signals") or [])
            if str(signal).strip()
        ]
        visual_used_for = [
            str(image.get("used_for")).strip()
            for image in vision_images
            if isinstance(image, dict)
            and str(image.get("used_for") or "").strip()
            and str(image.get("used_for")).strip().lower() != "aun sin clasificar"
        ]
        if vision_result is not None:
            execution_report["visual_evidence"] = {
                "status": "available" if (visual_signals or visual_used_for) else "insufficient",
                "signals": visual_signals[:12],
                "used_for": visual_used_for[:6],
                "confidence": [
                    image.get("confidence")
                    for image in vision_images
                    if isinstance(image, dict) and image.get("confidence") is not None
                ][:6],
                "limitations": [
                    str(item).strip()
                    for image in vision_images
                    if isinstance(image, dict)
                    for item in (image.get("limitations") or [])
                    if str(item).strip()
                ][:12],
                "used_in_final": False,
            }

        await emit_status(
            writer_agent_name, 1, 1, 1, 1, status="running", message="Redaccion final en curso"
        )
        await log_event("Redaccion final en curso", agent=writer_agent_name)
        log.bind(agent=writer_agent_name).info("Writer started")
        writer_mode = plan.writer_mode or "BRIEFING"
        writer_context = {
            # Put visual evidence first so the compact context cannot evict it
            # when several specialists are present.
            "visual_evidence": getattr(latest_result("vision_ocr"), "data", None),
            **{
                name: latest_result(name)
                for name in self.agents.keys()
                if name not in internal_agents
            },
        }
        writer_input = AgentInput(
            query=user_query.query,
            language=user_query.language,
            attachments=user_query.attachments,
            context=writer_context,
            writer_mode=writer_mode,
            user_id=user_query.user_id,
            decision_mode=user_query.decision_mode,
            response_mode=user_query.response_mode,
            memory_enabled=user_query.memory_enabled,
        )
        writer_input.context["_execution"] = execution_report
        writer_input.context["_plan"] = {
            "steps": list(steps),
            "policy": plan.policy.model_dump(),
            "diagnostics": plan.diagnostics.model_dump(),
            "writer_agent": writer_agent_name,
        }
        writer_input.context["_memory"] = {
            "enabled": memory_enabled and bool(user_id),
            "user_id": user_id,
            "memory_id": getattr(memory_meta, "memory_id", None),
            "memory_name": getattr(memory_meta, "memory_name", None),
            "used_sections": memory_used_sections,
            "context": memory_context,
        }
        writer_input.context["_conversation_history"] = conversation_history
        writer_input.context["_case_context"] = case_context
        writer_input.context["_case_context_usage"] = case_context_usage
        writer_input.context["_memory_reuse"] = (
            memory_reuse.model_dump(exclude_none=True) if memory_reuse else {}
        )
        stac_writer = latest_result("stac")
        stac_writer_results = getattr(stac_writer, "data", stac_writer) if stac_writer else None
        if isinstance(stac_writer_results, StacResults):
            writer_input.context["meteo"] = await _fetch_meteo_for_stac(stac_writer_results, user_query.context)
        writer_input.context["rs_config"] = RSAnalysisConfig()
        try:
            writer_output = await self.agents[writer_agent_name].run(writer_input)
        except Exception as exc:
            # Keep the writer visible in the execution report even when its
            # call fails before producing an AgentOutput.
            writer_output = None
            record_result("writer", 1, None)
            writer_assessment = ExecutionAssessment("hard_error", f"writer#1: {exc}")
        else:
            writer_assessment = classify_result("writer", 1, writer_output)
        store_execution_state("writer", 1, writer_assessment)
        if writer_assessment.level != "ok":
            record_failure(writer_assessment.message)
        final = writer_output.data if hasattr(writer_output, "data") else FinalAnswer()
        continuity_status = "none"
        continuity_title: str | None = None
        if resolved_case_id:
            continuity_status = "active" if continuity_resolution.reason == "conversation" else "matched"
            try:
                continuity_title = case_store.get_case(
                    resolved_case_id,
                    workspace_id=workspace_id,
                )["case"]["title"]
            except KeyError:
                resolved_case_id = None
        case_state = getattr(final, "case_state", None)
        if (
            resolved_case_id is None
            and continuity_mode == "auto"
            and should_create_case(
                query,
                attachment_count=len(attachments),
                case_state=case_state,
            )
        ):
            title = " ".join(query.strip().split()[:8])[:120] or "Nuevo seguimiento"
            created_case = case_store.create_case(
                workspace_id=workspace_id,
                title=title,
                objective=(getattr(case_state, "case_summary", "") or final.executive_summary)[:500],
            )
            resolved_case_id = created_case["case_id"]
            continuity_title = created_case["title"]
            continuity_status = "created"
            conversation_store.link_case(resolved_conversation_id, resolved_case_id)
            case_store.rebind_conversation(
                case_id=resolved_case_id,
                conversation_id=resolved_conversation_id,
                workspace_id=workspace_id,
            )
        final.continuity = ContinuitySummary(
            case_id=resolved_case_id,
            title=continuity_title,
            status=("ambiguous" if continuity_resolution.reason == "ambiguous" else continuity_status),
            next_step=(
                case_state.open_tasks[0].title
                if case_state is not None and getattr(case_state, "open_tasks", None)
                else (final.next_actions[0] if final.next_actions else None)
            ),
            created=continuity_status == "created",
            candidates=continuity_resolution.candidates,
        )
        final.cost_summary = CostSummary(
            **cost_store.summarize_conversation(resolved_conversation_id)
        )
        if hasattr(final, "execution"):
            final.execution = execution_report
        final.memory = MemoryUsage(
            enabled=memory_enabled and bool(user_id),
            user_id=user_id,
            memory_id=getattr(memory_meta, "memory_id", None),
            memory_name=getattr(memory_meta, "memory_name", None),
            used_sections=memory_used_sections,
        )
        final.case_id = resolved_case_id
        final.context_usage = ContextUsage.model_validate(
            {
                "case_id": resolved_case_id,
                "context_run_id": case_context_usage.get("context_run_id"),
                "items": case_context_usage.get("items") or [],
            }
        )
        if hasattr(final, "limitations"):
            merged_limitations = list(final.limitations or [])
            for failure in agent_failures:
                item = professional_failure_limitation(failure)
                if item and item not in merged_limitations:
                    merged_limitations.append(item)
            final.limitations = merged_limitations

        for field in ("message_md", "report_md"):
            raw_md = getattr(final, field, None)
            if raw_md:
                clean_md, parsed_blocks = parse_block_markers(raw_md)
                setattr(final, field, clean_md)
                final.content_blocks.extend(parsed_blocks)
        for field in ("message_md", "report_md"):
            raw_md = getattr(final, field, None)
            if raw_md:
                final.citations_resolved.extend(
                    resolve_citations(raw_md, final.references)
                )
        if isinstance(execution_report.get("visual_evidence"), dict):
            visible_lower = (getattr(final, "message_md", "") or "").lower()
            evidence = execution_report["visual_evidence"]
            evidence_terms = list(evidence.get("signals") or []) + list(evidence.get("used_for") or [])
            evidence["used_in_final"] = any(
                len(term) >= 8 and term.lower() in visible_lower
                for term in evidence_terms
            )
            # FinalAnswer validates/copies assigned dictionaries. Re-assign
            # after enriching the report so visual evidence metadata reaches
            # answer.execution and the evaluation artifact.
            if hasattr(final, "execution"):
                final.execution = execution_report
        auto_blocks = auto_generate_blocks(final)
        existing_ids = {b.ref_id for b in final.content_blocks}
        for block in auto_blocks:
            if block.ref_id not in existing_ids:
                final.content_blocks.append(block)
        if resolved_case_id:
            case_store.record_assistant_result(
                case_id=resolved_case_id,
                workspace_id=workspace_id,
                conversation_id=resolved_conversation_id,
                query=query,
                executive_summary=final.executive_summary,
                case_state=getattr(final, "case_state", None),
                attachment_ids=[
                    getattr(attachment, "attachment_id", "")
                    for attachment in attachments
                    if getattr(attachment, "attachment_id", "")
                ],
            )
            artifact = build_remote_sensing_artifact(
                query=query,
                decision_mode=decision_mode,
                memory_id=getattr(memory_meta, "memory_id", None),
                memory_name=getattr(memory_meta, "memory_name", None),
                observations=[],
                stac=getattr(latest_result("stac"), "data", None),
                remote_sensing=getattr(latest_result("rs_analyst"), "data", None),
            )
            if artifact is not None:
                case_store.append_event(
                    resolved_case_id,
                    event_type="remote_sensing_evidence_recorded",
                    actor_type="system",
                    source_type="remote_sensing",
                    payload=artifact.model_dump(exclude_none=True),
                )
        if resolved_conversation_id:
            answer_json = final.model_dump_json() if hasattr(final, "model_dump_json") else "{}"
            conversation_store.save_message(
                resolved_conversation_id,
                role="assistant",
                response_mode=response_mode,
                answer_summary=final.executive_summary[:200] if final.executive_summary else "",
                answer_json=answer_json,
            )
        record_result(writer_agent_name, 1, final)
        record_result("writer", 1, final)
        await emit_status(writer_agent_name, 1, 1, 1, 1, status="done", message="Respuesta preparada")
        await log_event("Respuesta preparada", agent=writer_agent_name)
        log.bind(agent=writer_agent_name).info("Writer finished")

        payload = {
            "plan": {
                "steps": steps,
                "runs": {name: 1 for name in plan_runs},
                "dependencies": dependencies,
                "allow_replan": plan.allow_replan,
                "writer_mode": writer_mode,
                "writer_agent": writer_agent_name,
                "response_mode": response_mode,
                "policy": plan.policy.model_dump(),
                "diagnostics": plan.diagnostics.model_dump(),
                "replan": replan_result,
            },
            "answer": final.model_dump() if hasattr(final, "model_dump") else final,
            "conversation_id": resolved_conversation_id,
        }
        await emit(
            {
                "type": "status",
                "stage": "completed",
                "message": "Conversacion finalizada",
                "actor": "system",
            }
        )
        await log_event("Conversacion finalizada")
        write_trace_report(resolved_conversation_id)
        return payload
