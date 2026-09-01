"""Ejecución de evaluación: compara modelos como cerebros del sistema multi-agente.

run_batch() ejecuta casos × modelos, recopila métricas deterministas,
del juez LLM, y persiste el routing de agentes para análisis.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evaluation.baselines import run_system
from evaluation.concurrency import AdaptiveConcurrencyPool
from evaluation.config import EvalConfig, JudgeConfig
from evaluation.export import build_export_rows, export_csv
from evaluation.llm_metrics import evaluate_multi_metrics
from evaluation.llm_support import LLMCallMetrics, LLMCallTracker, llm_enabled, provider_enabled
from evaluation.metrics import (
    compute_execution_metrics,
    compute_routing_score,
    compute_routing_metrics,
    check_routing_assertion,
)
from evaluation.reporting import generate_report
from evaluation.persistence import (
    atomic_write_json,
    create_manifest,
    load_manifest,
    load_persisted_artifacts,
    load_persisted_tracker,
    stable_task_key,
    update_manifest_task,
    write_task_record,
)
from evaluation.stats import compute_full_correlations
from evaluation.schemas import (
    CaseSpec,
    ConversationTurn,
    ExecutionMetrics,
    JudgeMultiMetrics,
    NormalizedOutput,
    RunArtifact,
    SCHEMA_VERSION,
)
from loguru import logger


def _batch_task_timeout_seconds() -> float:
    """Timeout holgado por caso completo, incluidos sistema y jueces.

    El timeout HTTP protege cada llamada; este segundo guardarraíl evita que
    una tarea quede viva indefinidamente si una librería/proveedor no propaga
    correctamente la cancelación. Es deliberadamente mucho mayor que el
    timeout de una petición individual.
    """
    raw = os.environ.get("EVALUATION_TASK_TIMEOUT_SECONDS", "7200").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("EVALUATION_TASK_TIMEOUT_SECONDS must be positive") from exc
    if value <= 0:
        raise ValueError("EVALUATION_TASK_TIMEOUT_SECONDS must be positive")
    return value


def _canonical_agent_name(name: Any) -> str:
    value = str(name or "").strip()
    if value in {"direct_writer", "report_writer"}:
        return "writer"
    return value


def _normalize_agent_names(names: list[Any] | tuple[Any, ...] | None) -> list[str]:
    normalized: list[str] = []
    for name in names or []:
        canonical = _canonical_agent_name(name)
        if canonical and canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _required_agents(case: CaseSpec) -> set[str]:
    optional = set(_normalize_agent_names(getattr(case, "optional_route", [])))
    required = set(_normalize_agent_names(case.expected_route)) - optional
    # Un adjunto no implica automáticamente visión: PDF/HTML y CSV tienen
    # analizadores especializados distintos. Solo las imágenes requieren
    # vision_ocr por modalidad.
    for attachment in case.attachments:
        content_type = str(attachment.get("content_type") or "").lower()
        filename = str(attachment.get("filename") or "").lower()
        if content_type.startswith("image/") or filename.endswith(
            (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
        ):
            required.add("vision_ocr")
    return required


def _execution_agents(report: dict[str, Any]) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return observed, successful and failed agents plus the planned route."""
    raw_agents = report.get("agents") if isinstance(report, dict) else {}
    agents = raw_agents if isinstance(raw_agents, dict) else {}
    observed = _normalize_agent_names(list(agents))
    failed = [
        _canonical_agent_name(name) for name, state in agents.items()
        if isinstance(state, dict)
        and state.get("final_level") in {"hard_error", "soft_error"}
    ]
    successful = [name for name in observed if name not in failed]
    plan = report.get("plan") if isinstance(report, dict) else {}
    planned = plan.get("steps", []) if isinstance(plan, dict) else []
    planned = _normalize_agent_names(planned)
    return observed, successful, failed, planned


def _execution_has_failure(report: dict[str, Any]) -> bool:
    if not isinstance(report, dict):
        return True
    if report.get("error"):
        return True
    _, _, failed, _ = _execution_agents(report)
    return bool(failed)


def _judge_case_for(case: CaseSpec) -> CaseSpec:
    """Use the last turn's expectations while retaining the full case context."""
    if not case.turns:
        return case
    last_turn = case.turns[-1]
    return CaseSpec(
        case_id=case.case_id,
        family=case.family,
        difficulty=case.difficulty,
        query=last_turn.query,
        context=case.context,
        gold_expectations=last_turn.gold_expectations or case.gold_expectations,
        judge_rubric_notes=case.judge_rubric_notes,
        is_multiturn=True,
        turns=case.turns,
        expected_route=case.expected_route,
        optional_route=case.optional_route,
        routing_assertion=case.routing_assertion,
    )


def _write_execution_checkpoint(
    checkpoint_dir: str | Path | None,
    *,
    artifact: RunArtifact,
    execution_report: dict[str, Any],
) -> None:
    if not checkpoint_dir:
        return
    root = Path(checkpoint_dir)
    judges = {
        name: result.model_dump() if result is not None else None
        for name, result in artifact.judge_results.items()
    }
    atomic_write_json(root / "execution.json", {
        "phase": "execution_completed",
        "run_id": artifact.run_id,
        "case_id": artifact.case_id,
        "model": artifact.model,
        "input_query": artifact.input_query,
        "normalized_output": artifact.normalized_output.model_dump(),
        "metrics": asdict(artifact.metrics),
        "execution_report": execution_report,
        "judge_results": judges,
        "errors": artifact.errors,
        "family": artifact.family,
        "difficulty": artifact.difficulty,
        "run_idx": artifact.run_idx,
        "updated_at": time.time(),
    })


def _load_execution_checkpoint(checkpoint_dir: str | Path | None) -> RunArtifact | None:
    if not checkpoint_dir:
        return None
    path = Path(checkpoint_dir) / "execution.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("phase") != "execution_completed":
        return None
    judges = {
        name: JudgeMultiMetrics.model_validate(value) if value is not None else None
        for name, value in (raw.get("judge_results") or {}).items()
    }
    return RunArtifact(
        run_id=str(raw.get("run_id") or "checkpoint"),
        case_id=str(raw.get("case_id") or ""),
        model=str(raw.get("model") or ""),
        input_query=str(raw.get("input_query") or ""),
        normalized_output=NormalizedOutput.model_validate(raw["normalized_output"]),
        metrics=ExecutionMetrics(**raw["metrics"]),
        judge_results=judges,
        agent_routing=raw.get("execution_report") or {},
        errors=list(raw.get("errors") or []),
        family=str(raw.get("family") or ""),
        difficulty=str(raw.get("difficulty") or ""),
        run_idx=int(raw.get("run_idx") or 0),
    )


def _apply_call_breakdown(metrics: ExecutionMetrics, calls: list[LLMCallMetrics]) -> None:
    metrics.estimated_cost_usd = sum(call.cost_usd for call in calls)
    metrics.system_cost_usd = sum(call.cost_usd for call in calls if call.component == "system")
    metrics.vision_cost_usd = sum(call.cost_usd for call in calls if call.component == "vision")
    metrics.judge_cost_usd = sum(call.cost_usd for call in calls if call.component == "judge")


async def _resume_missing_judges(
    artifact: RunArtifact,
    case: CaseSpec,
    *,
    judges: list[JudgeConfig],
    tracker: LLMCallTracker,
    concurrency_pool: AdaptiveConcurrencyPool,
) -> RunArtifact:
    """Evaluate only missing/failed judge slots for a persisted execution."""
    judge_case = _judge_case_for(case)
    execution_context = artifact.agent_routing or {}
    tracker_start = len(tracker.calls)
    for judge in judges:
        if judge.name in artifact.judge_results and artifact.judge_results[judge.name] is not None:
            continue
        try:
            artifact.judge_results[judge.name] = await _evaluate_judge_with_pool(
                judge_case,
                artifact.normalized_output,
                model=judge.model,
                provider=judge.provider,
                max_tokens=judge.max_tokens,
                reasoning_enabled=judge.reasoning_enabled,
                tracker=tracker,
                pool=concurrency_pool,
                execution_context=execution_context,
            )
        except Exception as exc:
            artifact.judge_results[judge.name] = None
            artifact.errors.append(f"Judge {judge.name} resume error: {exc}")
    artifact.judge_metrics = next(
        (result for result in artifact.judge_results.values() if result is not None),
        None,
    )
    new_calls = tracker.calls[tracker_start:]
    existing_cost = artifact.metrics.estimated_cost_usd
    artifact.metrics.system_cost_usd = artifact.metrics.system_cost_usd
    artifact.metrics.vision_cost_usd = artifact.metrics.vision_cost_usd
    artifact.metrics.judge_cost_usd += sum(call.cost_usd for call in new_calls if call.component == "judge")
    artifact.metrics.estimated_cost_usd = existing_cost + sum(call.cost_usd for call in new_calls)
    artifact.metrics.model_calls += len(new_calls)
    artifact.metrics.token_prompt_total += sum(call.prompt_tokens for call in new_calls)
    artifact.metrics.token_completion_total += sum(call.completion_tokens for call in new_calls)
    return artifact


# ── Ejecución de un caso ─────────────────────────────────────────────


async def run_single_case(
    case: CaseSpec,
    model: str,
    *,
    model_provider: str | None = None,
    model_temperature: float | None = None,
    model_max_tokens: int | None = None,
    vision_model_id: str | None = None,
    vision_provider: str | None = None,
    judges: list[JudgeConfig] | None = None,
    judge_model: str | None = None,
    judge_provider: str | None = None,
    tracker: LLMCallTracker | None = None,
    concurrency_pool: AdaptiveConcurrencyPool | None = None,
    run_idx: int = 0,
    checkpoint_dir: str | Path | None = None,
    queue_latency_ms: float = 0.0,
) -> RunArtifact:
    """Ejecuta un caso con un modelo dado usando el sistema multi-agente.

    Args:
        case: Caso de evaluación.
        model: ID del modelo a usar (ej: 'gpt-5-mini').
        model_provider: Proveedor del modelo.
        vision_model_id: ID del modelo de visión (opcional).
        vision_provider: Proveedor del modelo de visión.
        judges: Lista de jueces para evaluación multi-judge.
        judge_model: LEGACY - modelo del juez (si judges es None).
        judge_provider: LEGACY - proveedor del juez.
        tracker: Tracker de métricas LLM.

    Returns:
        RunArtifact con output, métricas, jueces y routing.
    """
    run_id = str(uuid.uuid4())[:8]
    start_time = time.monotonic()
    errors: list[str] = []

    logger.info(f"  [{run_id}] Ejecutando {case.case_id} con modelo={model}")

    tracker_start = len(tracker.calls) if tracker else 0

    # 1. Ejecutar el sistema
    output = NormalizedOutput(parse_status="failed")
    execution_report: dict[str, Any] = {}
    try:
        output, execution_report = await run_system(
            case,
            model_id=model,
            model_provider=model_provider,
            vision_model_id=vision_model_id,
            vision_provider=vision_provider,
            model_temperature=model_temperature,
            model_max_tokens=model_max_tokens,
            tracker=tracker,
        )
    except Exception as exc:
        logger.error(f"  [{run_id}] Error ejecutando: {exc}")
        errors.append(str(exc))

    latency_ms = (time.monotonic() - start_time) * 1000

    # Calcular coste y llamadas del tracker
    cost_usd = 0.0
    model_calls = 0
    token_prompt_total = 0
    token_completion_total = 0
    if tracker is not None:
        recent_calls = tracker.calls[tracker_start:]
        cost_usd = sum(c.cost_usd for c in recent_calls)
        model_calls = len(recent_calls)
        token_prompt_total = sum(c.prompt_tokens for c in recent_calls)
        token_completion_total = sum(c.completion_tokens for c in recent_calls)

    # Extraer routing del execution_report
    agents_invoked, agents_ok_names, agents_failed, agents_planned = _execution_agents(execution_report)
    agents_ok = len(agents_ok_names)
    agents_error = len(agents_failed)

    # 2. Calcular métricas deterministas
    metrics = compute_execution_metrics(
        case,
        output,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        model_calls=model_calls,
        token_prompt_total=token_prompt_total,
        token_completion_total=token_completion_total,
        agents_invoked=agents_invoked,
        agents_ok=agents_ok,
        agents_error=agents_error,
        route_observed=agents_invoked,
    )
    metrics.agents_planned = agents_planned
    metrics.system_latency_ms = latency_ms
    metrics.queue_latency_ms = queue_latency_ms
    metrics.agents_failed = agents_failed
    metrics.agents_extra = [name for name in agents_invoked if name not in agents_planned]

    required_agents = _required_agents(case)
    required_missing = sorted(required_agents - set(agents_invoked))
    required_failed = sorted(
        name for name in required_agents
        if name in agents_failed
        or any(
            _canonical_agent_name(agent_name) == name
            and isinstance(state, dict)
            and state.get("final_level") == "insufficient_data"
            for agent_name, state in (execution_report.get("agents") or {}).items()
        )
    )
    metrics.required_agents_missing = required_missing
    if case.attachments:
        vision_state = (execution_report.get("agents") or {}).get("vision_ocr", {})
        if "vision_ocr" in required_missing or not vision_state:
            metrics.visual_evidence_status = "missing"
        elif "vision_ocr" in required_failed:
            metrics.visual_evidence_status = "insufficient"
        else:
            metrics.visual_evidence_status = str(
                (execution_report.get("visual_evidence") or {}).get("status") or "available"
            )
        metrics.visual_evidence_used = bool(
            (execution_report.get("visual_evidence") or {}).get("used_in_final")
        )

    # 2b. Routing score y assertion
    metrics.routing_score = compute_routing_score(case, agents_invoked)
    metrics.routing_precision, metrics.routing_recall, metrics.routing_order_score = compute_routing_metrics(case, agents_invoked)
    metrics.routing_assertion_pass = (
        check_routing_assertion(case, agents_invoked)
        if isinstance(execution_report.get("agents"), dict)
        else False
    )
    has_required_issue = bool(required_missing or required_failed)
    visual_partial = metrics.visual_evidence_status == "insufficient"
    metrics.success = bool(
        metrics.success
        and not errors
        and not _execution_has_failure(execution_report)
        and not has_required_issue
        and not visual_partial
    )
    metrics.execution_status = (
        "ok" if metrics.success
        else "partial" if output.parse_status in {"ok", "partial"} and visual_partial
        else "failed"
    )
    metrics.failure_reason = str(execution_report.get("error") or (", ".join(agents_failed)) or (errors[0] if errors else ""))
    if required_missing:
        missing_reason = f"required agents missing: {', '.join(required_missing)}"
        metrics.failure_reason = f"{metrics.failure_reason}; {missing_reason}".strip("; ")
    if required_failed:
        failed_reason = f"required agents failed: {', '.join(required_failed)}"
        metrics.failure_reason = f"{metrics.failure_reason}; {failed_reason}".strip("; ")

    _write_execution_checkpoint(
        checkpoint_dir,
        artifact=RunArtifact(
            run_id=run_id,
            case_id=case.case_id,
            model=model,
            input_query=case.query,
            normalized_output=output,
            metrics=metrics,
            agent_routing=execution_report,
            errors=errors,
            family=case.family,
            difficulty=case.difficulty,
            run_idx=run_idx,
        ),
        execution_report=execution_report,
    )

    # 3. Ejecutar jueces LLM
    judge_results: dict[str, JudgeMultiMetrics | None] = {}
    judge_start_time = time.monotonic()

    system_execution_failed = output.parse_status == "failed" or _execution_has_failure(execution_report)
    if judges:
        # Multi-judge: ejecutar cada juez secuencialmente
        for jcfg in judges:
            try:
                jmetrics = await _evaluate_judge_with_pool(
                    case, output, model=jcfg.model, provider=jcfg.provider,
                    max_tokens=jcfg.max_tokens, reasoning_enabled=jcfg.reasoning_enabled,
                    tracker=tracker, pool=concurrency_pool,
                    execution_context=execution_report,
                )
                judge_results[jcfg.name] = jmetrics
                logger.info(
                    f"  [{run_id}] Juez {jcfg.name}: overall={jmetrics.overall_quality.score}/5, "
                    f"confidence={jmetrics.judge_confidence:.2f}"
                )
            except Exception as exc:
                logger.warning(f"  [{run_id}] Error en juez {jcfg.name}: {exc}")
                judge_results[jcfg.name] = None
                errors.append(f"Judge {jcfg.name} error: {exc}")
            _write_execution_checkpoint(
                checkpoint_dir,
                artifact=RunArtifact(
                    run_id=run_id, case_id=case.case_id, model=model,
                    input_query=case.query, normalized_output=output,
                    metrics=metrics, judge_results=judge_results,
                    agent_routing=execution_report, errors=errors,
                    family=case.family, difficulty=case.difficulty, run_idx=run_idx,
                ),
                execution_report=execution_report,
            )
    elif judge_model:
        # Legacy: un solo juez
        try:
            jmetrics = await _evaluate_judge_with_pool(
                case, output, model=judge_model, provider=judge_provider or "openrouter",
                tracker=tracker, pool=concurrency_pool,
                execution_context=execution_report,
            )
            judge_results[judge_model.split("/")[-1]] = jmetrics
            logger.info(
                f"  [{run_id}] Juez: overall={jmetrics.overall_quality.score}/5, "
                f"confidence={jmetrics.judge_confidence:.2f}"
            )
        except Exception as exc:
            logger.warning(f"  [{run_id}] Error en juez: {exc}")
            errors.append(f"Judge error: {exc}")

    if tracker is not None:
        recent_calls = tracker.calls[tracker_start:]
        _apply_call_breakdown(metrics, recent_calls)
        metrics.model_calls = len(recent_calls)
        metrics.token_prompt_total = sum(c.prompt_tokens for c in recent_calls)
        metrics.token_completion_total = sum(c.completion_tokens for c in recent_calls)
    metrics.judge_latency_ms = max(0.0, (time.monotonic() - judge_start_time) * 1000)
    metrics.task_wall_latency_ms = (time.monotonic() - start_time) * 1000

    # Primer juez como judge_metrics legacy
    first_judge = next(iter(judge_results.values()), None)

    import datetime as _dt
    timestamp_iso = _dt.datetime.now(_dt.UTC).isoformat(timespec="milliseconds")

    return RunArtifact(
        run_id=run_id,
        case_id=case.case_id,
        model=model,
        input_query=case.query,
        normalized_output=output,
        metrics=metrics,
        judge_results=judge_results,
        judge_metrics=first_judge,
        agent_routing=execution_report,
        errors=errors,
        timestamp_iso=timestamp_iso,
        family=case.family,
        difficulty=case.difficulty,
        run_idx=run_idx,
    )


async def run_multiturn_case(
    case: CaseSpec,
    model: str,
    *,
    model_provider: str | None = None,
    model_temperature: float | None = None,
    model_max_tokens: int | None = None,
    vision_model_id: str | None = None,
    vision_provider: str | None = None,
    judges: list[JudgeConfig] | None = None,
    judge_model: str | None = None,
    judge_provider: str | None = None,
    tracker: LLMCallTracker | None = None,
    concurrency_pool: AdaptiveConcurrencyPool | None = None,
    run_idx: int = 0,
    checkpoint_dir: str | Path | None = None,
    queue_latency_ms: float = 0.0,
) -> RunArtifact:
    """Ejecuta un caso multi-turn con un modelo dado.

    Para cada turno, ejecuta el sistema con el query del turno y un
    conversation_id compartido para mantener contexto.

    Args:
        case: Caso de evaluación con is_multiturn=True y turns definidos.
        model: ID del modelo a usar.
        model_provider: Proveedor del modelo.
        vision_model_id: ID del modelo de visión (opcional).
        vision_provider: Proveedor del modelo de visión.
        judges: Lista de jueces para evaluación multi-judge.
        judge_model: LEGACY - modelo del juez.
        judge_provider: LEGACY - proveedor del juez.
        tracker: Tracker de métricas LLM.

    Returns:
        RunArtifact con el output del último turno, métricas acumuladas y routing.
    """
    run_id = str(uuid.uuid4())[:8]
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else None
    conversation_id = str(uuid.uuid4())
    start_time = time.monotonic()
    errors: list[str] = []

    logger.info(f"  [{run_id}] Ejecutando multi-turn {case.case_id} ({len(case.turns)} turns) con modelo={model}")

    tracker_start = len(tracker.calls) if tracker else 0
    all_agents_invoked: list[str] = []
    all_agent_statuses: list[str] = []
    agent_status_history: dict[str, list[str]] = {}
    last_output = NormalizedOutput(parse_status="failed")
    last_execution_report: dict[str, Any] = {}

    start_turn_index = 0
    if checkpoint_root and checkpoint_root.exists():
        checkpoints = sorted(checkpoint_root.glob("turn_*.json"))
        if checkpoints:
            checkpoint = json.loads(checkpoints[-1].read_text(encoding="utf-8"))
            start_turn_index = int(checkpoint.get("next_turn_index", 0))
            conversation_id = str(checkpoint.get("conversation_id") or conversation_id)
            all_agents_invoked = list(checkpoint.get("all_agents_invoked") or [])
            all_agent_statuses = list(checkpoint.get("all_agent_statuses") or [])
            agent_status_history = dict(checkpoint.get("agent_status_history") or {})
            errors = list(checkpoint.get("errors") or [])
            if checkpoint.get("last_output"):
                last_output = NormalizedOutput.model_validate(checkpoint["last_output"])
            last_execution_report = dict(checkpoint.get("last_execution_report") or {})
            if tracker is not None:
                tracker.calls.extend(
                    LLMCallMetrics(**call)
                    for call in checkpoint.get("calls", [])
                )

    for turn_index, turn in enumerate(case.turns):
        if turn_index < start_turn_index:
            continue
        logger.info(f"  [{run_id}] Turn {turn.turn}: {turn.query[:60]}...")
        try:
            output, execution_report = await run_system(
                case,
                model_id=model,
                model_provider=model_provider,
                vision_model_id=vision_model_id,
                vision_provider=vision_provider,
                model_temperature=model_temperature,
                model_max_tokens=model_max_tokens,
                tracker=tracker,
                query_override=turn.query,
                conversation_id=conversation_id,
            )
            last_output = output
            last_execution_report = execution_report

            # Acumular agentes invocados
            turn_agents = _normalize_agent_names(list(execution_report.get("agents", {}).keys()))
            all_agents_invoked.extend(turn_agents)
            all_agent_statuses.extend(
                str(value.get("final_level", value.get("status", "unknown")))
                for value in execution_report.get("agents", {}).values()
                if isinstance(value, dict)
            )
            for raw_agent_name, value in execution_report.get("agents", {}).items():
                if isinstance(value, dict):
                    agent_name = _canonical_agent_name(raw_agent_name)
                    agent_status_history.setdefault(agent_name, []).append(str(value.get("final_level", value.get("status", "unknown"))))
            if execution_report.get("error"):
                errors.append(f"Turn {turn.turn} execution error: {execution_report['error']}")

        except Exception as exc:
            logger.error(f"  [{run_id}] Error en turn {turn.turn}: {exc}")
            errors.append(f"Turn {turn.turn} error: {exc}")

        if checkpoint_root:
            atomic_write_json(
                checkpoint_root / f"turn_{turn.turn}.json",
                {
                    "turn": turn.turn,
                    "next_turn_index": turn_index + 1,
                    "conversation_id": conversation_id,
                    "all_agents_invoked": all_agents_invoked,
                    "all_agent_statuses": all_agent_statuses,
                    "agent_status_history": agent_status_history,
                    "errors": errors,
                    "last_output": last_output.model_dump(),
                    "last_execution_report": last_execution_report,
                    "calls": [asdict(call) for call in (tracker.calls if tracker else [])],
                    "updated_at": time.time(),
                },
            )

    latency_ms = (time.monotonic() - start_time) * 1000

    # Calcular coste y llamadas del tracker
    cost_usd = 0.0
    model_calls = 0
    token_prompt_total = 0
    token_completion_total = 0
    if tracker is not None:
        recent_calls = tracker.calls[tracker_start:]
        cost_usd = sum(c.cost_usd for c in recent_calls)
        model_calls = len(recent_calls)
        token_prompt_total = sum(c.prompt_tokens for c in recent_calls)
        token_completion_total = sum(c.completion_tokens for c in recent_calls)

    # Extraer routing acumulado
    agents_ok = sum(1 for status in all_agent_statuses if status == "ok")
    agents_error = sum(1 for status in all_agent_statuses if status != "ok")

    # 2. Calcular métricas deterministas (usando el output del último turno)
    metrics = compute_execution_metrics(
        case,
        last_output,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        model_calls=model_calls,
        token_prompt_total=token_prompt_total,
        token_completion_total=token_completion_total,
        agents_invoked=list(set(all_agents_invoked)),
        agents_ok=agents_ok,
        agents_error=agents_error,
        route_observed=list(set(all_agents_invoked)),
    )
    metrics.system_latency_ms = latency_ms
    metrics.queue_latency_ms = queue_latency_ms

    # 2b. Routing score y assertion
    all_agents_invoked = _normalize_agent_names(all_agents_invoked)
    required_agents = _required_agents(case)
    required_missing = sorted(required_agents - set(all_agents_invoked))
    required_failed = sorted(
        name for name in required_agents
        if any(status in {"hard_error", "soft_error", "insufficient_data"} for status in agent_status_history.get(name, []))
    )
    metrics.required_agents_missing = required_missing
    if case.attachments:
        visual_report = last_execution_report.get("visual_evidence") or {}
        vision_state = (last_execution_report.get("agents") or {}).get("vision_ocr", {})
        if "vision_ocr" in required_missing or not vision_state:
            metrics.visual_evidence_status = "missing"
        elif "vision_ocr" in required_failed:
            metrics.visual_evidence_status = "insufficient"
        else:
            metrics.visual_evidence_status = str(visual_report.get("status") or "available")
        metrics.visual_evidence_used = bool(visual_report.get("used_in_final"))

    has_required_issue = bool(required_missing or required_failed)
    visual_partial = metrics.visual_evidence_status == "insufficient"
    metrics.routing_score = compute_routing_score(case, list(set(all_agents_invoked)))
    metrics.routing_precision, metrics.routing_recall, metrics.routing_order_score = compute_routing_metrics(case, all_agents_invoked)
    metrics.routing_assertion_pass = check_routing_assertion(case, list(set(all_agents_invoked)))
    metrics.execution_status = (
        "ok" if metrics.success and not errors and not last_execution_report.get("error")
        else "partial" if last_output.parse_status in {"ok", "partial"} and visual_partial
        else "failed"
    )
    metrics.failure_reason = "; ".join(errors) or str(last_execution_report.get("error") or "")
    metrics.success = metrics.success and not errors
    metrics.success = bool(
        metrics.success
        and not _execution_has_failure(last_execution_report)
        and not has_required_issue
        and not visual_partial
    )
    metrics.execution_status = (
        "ok" if metrics.success
        else "partial" if last_output.parse_status in {"ok", "partial"} and visual_partial
        else "failed"
    )
    if required_missing:
        metrics.failure_reason = f"{metrics.failure_reason}; required agents missing: {', '.join(required_missing)}".strip("; ")
    if required_failed:
        metrics.failure_reason = f"{metrics.failure_reason}; required agents failed: {', '.join(required_failed)}".strip("; ")
    metrics.agents_failed = [
        name for name, statuses in agent_status_history.items()
        if any(status in {"hard_error", "soft_error"} for status in statuses)
    ]
    metrics.agents_extra = [
        name for name in set(all_agents_invoked)
        if name not in set(_normalize_agent_names(case.expected_route))
    ]

    # 3. Ejecutar jueces LLM (evalúa el output final contra las expectations del último turno)
    judge_results: dict[str, JudgeMultiMetrics | None] = {}

    # Preparar caso para juez
    last_turn = case.turns[-1] if case.turns else None
    judge_case = case
    if last_turn:
        from evaluation.schemas import CaseSpec, GoldExpectations
        judge_case = CaseSpec(
            case_id=case.case_id,
            family=case.family,
            difficulty=case.difficulty,
            query=last_turn.query,
            context=case.context,
            gold_expectations=last_turn.gold_expectations or case.gold_expectations,
            judge_rubric_notes=case.judge_rubric_notes,
            is_multiturn=True,
            turns=case.turns,
            expected_route=case.expected_route,
            optional_route=case.optional_route,
            routing_assertion=case.routing_assertion,
        )

    system_execution_failed = last_output.parse_status == "failed" or _execution_has_failure(last_execution_report)
    judge_start_time = time.monotonic()
    if judges:
        # Multi-judge
        for jcfg in judges:
            try:
                jmetrics = await _evaluate_judge_with_pool(
                    judge_case, last_output, model=jcfg.model, provider=jcfg.provider,
                    max_tokens=jcfg.max_tokens, reasoning_enabled=jcfg.reasoning_enabled,
                    tracker=tracker, pool=concurrency_pool,
                    execution_context=last_execution_report,
                )
                judge_results[jcfg.name] = jmetrics
                logger.info(
                    f"  [{run_id}] Juez {jcfg.name}: overall={jmetrics.overall_quality.score}/5, "
                    f"confidence={jmetrics.judge_confidence:.2f}"
                )
            except Exception as exc:
                logger.warning(f"  [{run_id}] Error en juez {jcfg.name}: {exc}")
                judge_results[jcfg.name] = None
                errors.append(f"Judge {jcfg.name} error: {exc}")
    elif judge_model:
        # Legacy
        try:
            jmetrics = await _evaluate_judge_with_pool(
                judge_case, last_output, model=judge_model, provider=judge_provider or "openrouter",
                tracker=tracker, pool=concurrency_pool,
                execution_context=last_execution_report,
            )
            judge_results[judge_model.split("/")[-1]] = jmetrics
            logger.info(
                f"  [{run_id}] Juez: overall={jmetrics.overall_quality.score}/5, "
                f"confidence={jmetrics.judge_confidence:.2f}"
            )
        except Exception as exc:
            logger.warning(f"  [{run_id}] Error en juez: {exc}")
            errors.append(f"Judge error: {exc}")

    if tracker is not None:
        recent_calls = tracker.calls[tracker_start:]
        _apply_call_breakdown(metrics, recent_calls)
        metrics.model_calls = len(recent_calls)
        metrics.token_prompt_total = sum(c.prompt_tokens for c in recent_calls)
        metrics.token_completion_total = sum(c.completion_tokens for c in recent_calls)
    metrics.judge_latency_ms = max(0.0, (time.monotonic() - judge_start_time) * 1000)
    metrics.task_wall_latency_ms = (time.monotonic() - start_time) * 1000

    # Construir execution_report acumulado
    accumulated_report = {
        "agents": {},
        "turns_executed": len(case.turns),
        "conversation_id": conversation_id,
    }
    for agent_name in set(all_agents_invoked):
        statuses = agent_status_history.get(agent_name, [])
        accumulated_report["agents"][agent_name] = {
            "invocations": all_agents_invoked.count(agent_name),
            "status": "ok" if statuses and all(status == "ok" for status in statuses) else "error",
        }

    # Primer juez como judge_metrics legacy
    first_judge = next(iter(judge_results.values()), None)

    import datetime as _dt
    timestamp_iso = _dt.datetime.now(_dt.UTC).isoformat(timespec="milliseconds")

    return RunArtifact(
        run_id=run_id,
        case_id=case.case_id,
        model=model,
        input_query=case.query,
        normalized_output=last_output,
        metrics=metrics,
        judge_results=judge_results,
        judge_metrics=first_judge,
        agent_routing=accumulated_report,
        errors=errors,
        timestamp_iso=timestamp_iso,
        family=case.family,
        difficulty=case.difficulty,
        run_idx=run_idx,
    )


# ── Batch ────────────────────────────────────────────────────────────


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _is_saturation_error(error: str) -> bool:
    value = error.lower()
    return any(
        token in value
        for token in (
            "429",
            "rate limit",
            "too many requests",
            "timeout",
            "timed out",
            "connection reset",
            "connection error",
            "connecterror",
            "remoteprotocolerror",
            "readtimeout",
            "502",
            "503",
            "504",
            "service unavailable",
        )
    )


async def _evaluate_judge_with_pool(
    case: CaseSpec,
    output: NormalizedOutput,
    *,
    model: str,
    provider: str,
    max_tokens: int | None = None,
    reasoning_enabled: bool | None = None,
    tracker: LLMCallTracker | None,
    pool: AdaptiveConcurrencyPool | None,
    execution_context: dict[str, Any] | None = None,
) -> JudgeMultiMetrics:
    if pool is not None:
        await pool.acquire(provider, role="judge")
    saturated = False
    try:
        return await evaluate_multi_metrics(
            case, output, model=model, provider=provider, tracker=tracker,
            max_tokens=max_tokens, reasoning_enabled=reasoning_enabled,
            execution_context=execution_context,
        )
    except Exception as exc:
        saturated = _is_saturation_error(str(exc))
        raise
    finally:
        if pool is not None:
            await pool.release(provider, role="judge", saturated=saturated)


def _write_batch_progress(output_dir: str | Path, progress: dict[str, Any]) -> None:
    """Escribe el estado observable de un batch sin dejar JSON parcial."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "batch_progress.json"
    temporary = target_dir / "batch_progress.json.tmp"
    temporary.write_text(
        json.dumps(progress, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(target)


@dataclass
class BatchResult:
    """Resultado de una ejecución de batch completa."""

    batch_id: str
    config: EvalConfig
    cases: list[CaseSpec] = field(default_factory=list)
    artifacts: list[RunArtifact] = field(default_factory=list)
    tracker: LLMCallTracker = field(default_factory=LLMCallTracker)
    concurrency_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    wall_time_ms: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return self.tracker.total_cost_usd

    @property
    def total_latency_ms(self) -> float:
        return sum(a.metrics.latency_ms for a in self.artifacts)

    @property
    def task_wall_latency_ms(self) -> float:
        return sum(a.metrics.task_wall_latency_ms for a in self.artifacts)

    def save(self, output_dir: str) -> Path:
        from evaluation.reporting import BatchLock
        out = Path(output_dir) / self.batch_id
        with BatchLock(out):
            return self._save_unlocked(output_dir)

    def _save_unlocked(self, output_dir: str) -> Path:
        """Guarda los resultados del batch en disco."""
        out = Path(output_dir) / self.batch_id
        out.mkdir(parents=True, exist_ok=True)

        # Schema version
        (out / "schema_version.json").write_text(
            json.dumps({"version": SCHEMA_VERSION}, indent=2),
            encoding="utf-8",
        )

        # EvalConfig
        self.config.to_json(str(out / "eval_config.json"))

        # CaseSpec completo
        if self.cases:
            (out / "case_specs.json").write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "cases": [case.model_dump() for case in self.cases],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        # LLM call details
        calls_data = [asdict(call) for call in self.tracker.calls]
        (out / "llm_calls.json").write_text(
            json.dumps(calls_data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        case_specs_by_id = {case.case_id: case for case in self.cases}

        for artifact in self.artifacts:
            dir_name = stable_task_key(artifact.case_id, artifact.model, artifact.run_idx)
            artifact_dir = out / dir_name
            artifact_dir.mkdir(exist_ok=True)

            case_spec = case_specs_by_id.get(artifact.case_id)
            if case_spec is not None:
                (artifact_dir / "case_spec.json").write_text(
                    case_spec.model_dump_json(indent=2),
                    encoding="utf-8",
                )

            # Output
            (artifact_dir / "output.json").write_text(
                artifact.normalized_output.model_dump_json(indent=2),
                encoding="utf-8",
            )

            # Metrics
            (artifact_dir / "metrics.json").write_text(
                json.dumps(asdict(artifact.metrics), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Judge metrics (multi-judge o legacy)
            judges_dir = artifact_dir / "judges"
            if artifact.judge_results:
                for jname, jmetrics in artifact.judge_results.items():
                    jdir = judges_dir / jname
                    jdir.mkdir(parents=True, exist_ok=True)
                    if jmetrics is not None:
                        (jdir / "judge_metrics.json").write_text(
                            jmetrics.model_dump_json(indent=2),
                            encoding="utf-8",
                        )
                    else:
                        (jdir / "judge_metrics.json").write_text(
                            json.dumps({"error": "judge_failed"}, indent=2),
                            encoding="utf-8",
                        )
            elif artifact.judge_metrics is not None:
                # Fallback legacy
                judges_dir.mkdir(parents=True, exist_ok=True)
                default_judge_dir = judges_dir / "default"
                default_judge_dir.mkdir(parents=True, exist_ok=True)
                (default_judge_dir / "judge_metrics.json").write_text(
                    artifact.judge_metrics.model_dump_json(indent=2),
                    encoding="utf-8",
                )

            # Artifact metadata
            artifact_meta = {
                "schema_version": artifact.schema_version,
                "run_id": artifact.run_id,
                "case_id": artifact.case_id,
                "model": artifact.model,
                "input_query": artifact.input_query,
                "timestamp_iso": artifact.timestamp_iso,
                "family": artifact.family,
                "difficulty": artifact.difficulty,
                "run_idx": artifact.run_idx,
                "errors": artifact.errors,
            }
            (artifact_dir / "artifact.json").write_text(
                json.dumps(artifact_meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Agent routing (nuevo)
            if artifact.agent_routing is not None:
                (artifact_dir / "routing.json").write_text(
                    json.dumps(artifact.agent_routing, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        # Batch summary
        judge_failure_counts: dict[str, int] = {}
        quality_scored_artifacts = 0
        for artifact in self.artifacts:
            scored = False
            for judge_name, judge_result in artifact.judge_results.items():
                if judge_result is None:
                    judge_failure_counts[judge_name] = judge_failure_counts.get(judge_name, 0) + 1
                else:
                    scored = True
            if scored:
                quality_scored_artifacts += 1

        summary = {
            "schema_version": SCHEMA_VERSION,
            "batch_id": self.batch_id,
            "total_artifacts": len(self.artifacts),
            "quality_scored_artifacts": quality_scored_artifacts,
            "total_cost_usd": self.total_cost_usd,
            "cost_estimate_complete": self.tracker.cost_complete,
            "unknown_cost_calls": self.tracker.unknown_cost_calls,
            "unknown_cost_models": self.tracker.unknown_cost_models,
            "total_latency_ms": self.total_latency_ms,
            "system_latency_sum_ms": self.total_latency_ms,
            "sum_task_wall_latency_ms": self.task_wall_latency_ms,
            "batch_wall_time_ms": self.wall_time_ms,
            "latency_by_provider_component": self.tracker.breakdown("latency_ms"),
            "cost_by_provider_component": self.tracker.breakdown("cost_usd"),
            "latency_by_component": self.tracker.component_breakdown("latency_ms"),
            "cost_by_component": self.tracker.component_breakdown("cost_usd"),
            "judges_used": list({
                j for a in self.artifacts for j, result in a.judge_results.items() if result is not None
            }),
            "judge_failure_counts": judge_failure_counts,
            "by_model": self._aggregate_by_model(),
            "by_case": self._aggregate_by_case(),
            "routing_summary": self._aggregate_routing(),
            "concurrency": self.concurrency_stats,
            "parser_summary": self.tracker.parse_summary(),
        }
        manifest_path = out / "manifest.json"
        if manifest_path.exists():
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = int(manifest_data.get("expected_tasks", len(self.artifacts)))
            completed = sum(1 for task in manifest_data.get("tasks", []) if task.get("status") == "completed")
            summary["partial"] = completed < expected
            summary["completed_tasks"] = completed
            summary["expected_tasks"] = expected
        else:
            summary["partial"] = False
        (out / "batch_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        export_csv(out, out / "aggregate.csv")
        csv_rows = build_export_rows(out)
        correlations = compute_full_correlations(csv_rows)
        (out / "correlations.json").write_text(
            json.dumps(correlations, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        generate_report(out, str(out), fmt="both")
        if summary.get("partial"):
            (out / "partial_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            report_path = out / "report.html"
            if report_path.exists():
                (out / "partial_report.html").write_text(
                    report_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )

        return out

    def _aggregate_by_model(self) -> dict[str, Any]:
        """Agrega métricas por modelo, con desglose por juez."""
        by_model: dict[str, list[RunArtifact]] = {}
        for a in self.artifacts:
            by_model.setdefault(a.model, []).append(a)

        routing_summary = self._aggregate_routing()
        result = {}
        for model, artifacts in by_model.items():
            latencies = [a.metrics.latency_ms for a in artifacts]
            system_latencies = [a.metrics.system_latency_ms for a in artifacts]
            judge_latencies = [a.metrics.judge_latency_ms for a in artifacts]
            task_wall_latencies = [a.metrics.task_wall_latency_ms for a in artifacts]
            costs = [a.metrics.estimated_cost_usd for a in artifacts]
            system_costs = [a.metrics.system_cost_usd for a in artifacts]
            vision_costs = [a.metrics.vision_cost_usd for a in artifacts]
            judge_costs = [a.metrics.judge_cost_usd for a in artifacts]
            forbidden_rates = [a.metrics.forbidden_claim_rate for a in artifacts]
            overclaim_counts = [a.metrics.overclaim_count for a in artifacts]
            actionabilities = [a.metrics.actionability for a in artifacts]
            clarifications = [1.0 if a.metrics.clarification_detected else 0.0 for a in artifacts]
            agents_ok = [a.metrics.agents_ok for a in artifacts]
            agents_error = [a.metrics.agents_error for a in artifacts]
            token_prompt_total = sum(a.metrics.token_prompt_total for a in artifacts)
            token_completion_total = sum(a.metrics.token_completion_total for a in artifacts)
            pooled_overall_scores: list[float] = []

            # Agregación por juez
            judge_summary: dict[str, Any] = {}
            all_judge_names: set[str] = set()
            for a in artifacts:
                all_judge_names.update(a.judge_results.keys())

            for jname in sorted(all_judge_names):
                scores = []
                dims: dict[str, list[float]] = {}
                judge_confidence: list[float] = []
                perceived_difficulty: list[float] = []
                gold_concepts_coverage: list[float] = []
                gold_actions_coverage: list[float] = []
                gold_facts_coverage: list[float] = []
                for a in artifacts:
                    jm = a.judge_results.get(jname)
                    if jm is not None:
                        scores.append(jm.overall_quality.score)
                        pooled_overall_scores.append(jm.overall_quality.score)
                        judge_confidence.append(jm.judge_confidence)
                        perceived_difficulty.append(float(jm.perceived_difficulty))
                        gold_concepts_coverage.append(jm.gold_concepts_coverage)
                        gold_actions_coverage.append(jm.gold_actions_coverage)
                        gold_facts_coverage.append(jm.gold_facts_coverage)
                        for dim_name in [
                            "factual_correctness", "domain_accuracy",
                            "responsible_action_quality", "actionability",
                            "decision_support_quality", "evidence_utilization",
                            "transparent_confidence", "case_personalization",
                            "practical_value", "overall_quality",
                        ]:
                            dims.setdefault(dim_name, []).append(
                                getattr(jm, dim_name).score
                            )
                judge_summary[jname] = {
                    "overall_mean": round(_mean(scores), 2) if scores else 0.0,
                    "overall_std": round(_std(scores), 2) if len(scores) > 1 else 0.0,
                    "dimensions": {
                        dim: {
                            "mean": round(_mean(vals), 2),
                            "std": round(_std(vals), 2) if len(vals) > 1 else 0.0,
                        }
                        for dim, vals in dims.items()
                        if vals
                    },
                    "judge_confidence_mean": round(_mean(judge_confidence), 2) if judge_confidence else 0.0,
                    "perceived_difficulty_mean": round(_mean(perceived_difficulty), 2) if perceived_difficulty else 0.0,
                    "gold_concepts_coverage_mean": round(_mean(gold_concepts_coverage), 2) if gold_concepts_coverage else 0.0,
                    "gold_actions_coverage_mean": round(_mean(gold_actions_coverage), 2) if gold_actions_coverage else 0.0,
                    "gold_facts_coverage_mean": round(_mean(gold_facts_coverage), 2) if gold_facts_coverage else 0.0,
                    "n_scored": len(scores),
                }

            result[model] = {
                "n": len(artifacts),
                "judges": judge_summary,
                "judge_overall_mean": round(_mean(pooled_overall_scores), 2) if pooled_overall_scores else 0.0,
                "judge_overall_std": round(_std(pooled_overall_scores), 2) if len(pooled_overall_scores) > 1 else 0.0,
                "forbidden_claim_rate_mean": round(_mean(forbidden_rates), 4) if forbidden_rates else 0.0,
                "overclaim_count_mean": round(_mean(overclaim_counts), 2) if overclaim_counts else 0.0,
                "actionability_mean": round(_mean(actionabilities), 4) if actionabilities else 0.0,
                "clarification_rate": round(_mean(clarifications), 4) if clarifications else 0.0,
                "latency_mean_ms": round(_mean(latencies), 1) if latencies else 0.0,
                "system_latency_mean_ms": round(_mean(system_latencies), 1) if system_latencies else 0.0,
                "judge_latency_mean_ms": round(_mean(judge_latencies), 1) if judge_latencies else 0.0,
                "task_wall_latency_mean_ms": round(_mean(task_wall_latencies), 1) if task_wall_latencies else 0.0,
                "latency_std_ms": round(_std(latencies), 1) if len(latencies) > 1 else 0.0,
                "cost_mean_usd": round(_mean(costs), 6) if costs else 0.0,
                "cost_total_usd": round(sum(costs), 6),
                "system_cost_total_usd": round(sum(system_costs), 6),
                "vision_cost_total_usd": round(sum(vision_costs), 6),
                "judge_cost_total_usd": round(sum(judge_costs), 6),
                "success_rate": sum(1 for a in artifacts if a.metrics.success) / len(artifacts) if artifacts else 0.0,
                "agents_ok_mean": round(_mean(agents_ok), 2) if agents_ok else 0.0,
                "agents_error_mean": round(_mean(agents_error), 2) if agents_error else 0.0,
                "token_prompt_total": token_prompt_total,
                "token_completion_total": token_completion_total,
                "routing_summary": routing_summary.get(model, {}),
            }
        return result

    def _aggregate_by_case(self) -> dict[str, Any]:
        """Agrega resultados por caso, con scores por juez y por run."""
        by_case: dict[str, list[RunArtifact]] = {}
        for a in self.artifacts:
            by_case.setdefault(a.case_id, []).append(a)

        result = {}
        for case_id, artifacts in by_case.items():
            models: dict[str, Any] = {}
            first = artifacts[0]
            for a in artifacts:
                judge_scores = {
                    jname: jm.overall_quality.score
                    for jname, jm in a.judge_results.items()
                    if jm is not None
                }
                model_entry = models.setdefault(
                    a.model,
                    {
                        "runs": [],
                        "_judge_scores": {},
                    },
                )
                model_entry["runs"].append(
                    {
                        "run_idx": a.run_idx,
                        "judge_scores": judge_scores,
                        "judge_overall": round(_mean(list(judge_scores.values())), 2) if judge_scores else None,
                        "forbidden_claim_rate": a.metrics.forbidden_claim_rate,
                        "overclaim_count": a.metrics.overclaim_count,
                        "actionability": a.metrics.actionability,
                        "clarification_detected": a.metrics.clarification_detected,
                        "latency_ms": a.metrics.latency_ms,
                        "cost_usd": a.metrics.estimated_cost_usd,
                        "success": a.metrics.success,
                        "agents_invoked": a.metrics.agents_invoked,
                        "gold_concepts_coverage": a.judge_metrics.gold_concepts_coverage if a.judge_metrics else None,
                        "gold_actions_coverage": a.judge_metrics.gold_actions_coverage if a.judge_metrics else None,
                        "gold_facts_coverage": a.judge_metrics.gold_facts_coverage if a.judge_metrics else None,
                    }
                )
                for jname, score in judge_scores.items():
                    model_entry["_judge_scores"].setdefault(jname, []).append(score)
            for model_entry in models.values():
                judge_scores_list = model_entry.pop("_judge_scores", {})
                model_entry["judge_scores"] = {
                    jname: round(_mean(scores), 2) if scores else None
                    for jname, scores in judge_scores_list.items()
                }
                model_entry["judge_overall"] = (
                    round(
                        _mean(
                            [
                                score
                                for scores in judge_scores_list.values()
                                for score in scores
                            ]
                        ),
                        2,
                    )
                    if judge_scores_list
                    else None
                )
            result[case_id] = {
                "family": first.family,
                "difficulty": first.difficulty,
                "query": first.input_query,
                "n_models": len(models),
                "models": models,
            }
        return result

    def _aggregate_routing(self) -> dict[str, Any]:
        """Agrega patrones de routing por modelo."""
        by_model: dict[str, list[RunArtifact]] = {}
        for a in self.artifacts:
            by_model.setdefault(a.model, []).append(a)

        result = {}
        for model, artifacts in by_model.items():
            agent_counts: dict[str, int] = {}
            total = 0
            for a in artifacts:
                if a.agent_routing and "agents" in a.agent_routing:
                    for agent_name in a.agent_routing["agents"]:
                        agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1
                        total += 1

            result[model] = {
                "total_agent_invocations": total,
                "by_agent": agent_counts,
                "unique_agents_used": list(agent_counts.keys()),
            }
        return result


async def run_batch(
    cases: list[CaseSpec],
    config: EvalConfig,
    *,
    max_concurrent: int | None = None,
    case_range: dict[str, int] | None = None,
    resume_batch_dir: str | Path | None = None,
) -> BatchResult:
    """Ejecuta un batch completo de comparación de modelos.

    Para cada caso × modelo, ejecuta el sistema multi-agente y el juez.

    Args:
        cases: Lista de casos a evaluar.
        config: Configuración de evaluación.
        max_concurrent: Máximo de ejecuciones concurrentes.

    Returns:
        BatchResult con todos los artefactos.
    """
    if not llm_enabled():
        raise RuntimeError("La evaluación requiere EVALUATION_ENABLE_LLM=1 y credenciales configuradas.")
    if not config.models:
        raise RuntimeError("La evaluación requiere al menos un modelo configurado.")
    unavailable = [
        f"{item.name} ({item.provider})"
        for item in config.models.values()
        if not provider_enabled(item.provider)
    ]
    if unavailable:
        raise RuntimeError("Faltan credenciales para: " + ", ".join(unavailable))

    # Verificar credenciales de jueces
    judge_providers = {j.provider for j in config.judges}
    for jp in judge_providers:
        if not provider_enabled(jp):
            judge_names = [j.name for j in config.judges if j.provider == jp]
            raise RuntimeError(f"Faltan credenciales para juez ({jp}): {', '.join(judge_names)}")

    resume_dir = Path(resume_batch_dir) if resume_batch_dir else None
    if resume_dir is not None:
        manifest = load_manifest(resume_dir)
        from evaluation.persistence import validate_manifest
        validate_manifest(manifest, config, cases)
        batch_id = str(manifest["batch_id"])
        tracker = load_persisted_tracker(resume_dir)
        existing_artifacts = load_persisted_artifacts(resume_dir)
    else:
        batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        tracker = LLMCallTracker()
        existing_artifacts = {}
    initial_concurrency = max_concurrent or config.max_concurrent
    if initial_concurrency < 1:
        raise ValueError("max_concurrent must be >= 1")
    concurrency = AdaptiveConcurrencyPool(initial_concurrency)
    task_timeout_seconds = _batch_task_timeout_seconds()
    budget_lock = asyncio.Lock()
    progress_dir = Path(config.output_path)
    progress_started = time.monotonic()

    # Construir tareas: caso × modelo
    tasks = []
    model_configs = list(config.models.values())
    model_ids = [item.model_id for item in model_configs]
    for case in cases:
        for model_config in model_configs:
            for run_idx in range(config.runs_per_case):
                tasks.append((case, model_config, run_idx))
    total_tasks = len(tasks)

    batch_dir = resume_dir or (Path(config.output_path) / batch_id)
    batch_dir.mkdir(parents=True, exist_ok=True)
    if resume_dir is None:
        manifest = create_manifest(
            batch_id=batch_id,
            config=config,
            cases=cases,
            tasks=tasks,
            case_range=case_range,
        )
        atomic_write_json(batch_dir / "manifest.json", manifest)
    else:
        manifest = load_manifest(batch_dir)
        completed_keys = {
            task.get("task_key") for task in manifest.get("tasks", [])
            if task.get("status") == "completed" and task.get("task_key") in existing_artifacts
        }
        expected_judge_names = {judge.name for judge in config.judges}
        completed_keys.update(
            key for key, artifact in existing_artifacts.items()
            if expected_judge_names.issubset(artifact.judge_results)
            and all(artifact.judge_results[name] is not None for name in expected_judge_names)
        )
        tasks = [
            task for task in tasks
            if stable_task_key(task[0].case_id, task[1].model_id, task[2]) not in completed_keys
        ]
    atomic_write_json(
        batch_dir / "state.json",
        {
            "batch_id": batch_id,
            "status": "running",
            "completed": 0,
            "expected_tasks": total_tasks,
            "updated_at": time.time(),
        },
    )

    logger.info(
        f"Iniciando batch {batch_id}: {len(tasks)} ejecuciones "
        f"({len(cases)} casos x {len(model_ids)} modelos x {config.runs_per_case} runs)"
    )

    budget_usd = config.budget_usd
    budget_exceeded = False
    reserved_budget_usd = 0.0
    task_count = max(1, total_tasks)

    artifacts = [
        existing_artifacts[key]
        for key in sorted(existing_artifacts)
        if key in {
            task.get("task_key") for task in manifest.get("tasks", [])
            if task.get("status") == "completed"
        }
        or key in {
            candidate for candidate, artifact in existing_artifacts.items()
            if {judge.name for judge in config.judges}.issubset(artifact.judge_results)
            and all(artifact.judge_results[name] is not None for name in {judge.name for judge in config.judges})
        }
    ]
    completed_tasks = len(artifacts)

    def _publish_progress(status: str = "running") -> None:
        elapsed_seconds = time.monotonic() - progress_started
        average_seconds = elapsed_seconds / completed_tasks if completed_tasks else 0.0
        remaining = max(0, total_tasks - completed_tasks)
        judge_valid = sum(
            1
            for artifact in artifacts
            for result in artifact.judge_results.values()
            if result is not None
        )
        judge_slots = completed_tasks * len(config.judges)
        _write_batch_progress(
            progress_dir,
            {
                "status": status,
                "batch_id": batch_id,
                "completed": completed_tasks,
                "total": total_tasks,
                "remaining": remaining,
                "elapsed_seconds": round(elapsed_seconds, 2),
                "eta_seconds": round(average_seconds * remaining, 2) if completed_tasks else None,
                "artifacts_ok": sum(1 for artifact in artifacts if not artifact.errors),
                "artifacts_failed": sum(1 for artifact in artifacts if artifact.errors),
                "skipped_budget": sum(
                    1 for artifact in artifacts
                    if artifact.metrics.execution_status == "skipped_budget"
                ),
                "judge_valid": judge_valid,
                "judge_failures": max(0, judge_slots - judge_valid),
                "judge_slots": judge_slots,
                "cost_usd": round(tracker.total_cost_usd, 6),
                "initial_concurrency": initial_concurrency,
                "concurrency": concurrency.snapshot(),
                "models": model_ids,
                "judges": [judge.name for judge in config.judges],
                "updated_at_epoch": time.time(),
            },
        )
        atomic_write_json(batch_dir / "ledger.json", {
            "budget_usd": budget_usd,
            "confirmed_cost_usd": tracker.total_cost_usd,
            "reserved_cost_usd": reserved_budget_usd,
            "remaining_budget_usd": max(0.0, budget_usd - tracker.total_cost_usd - reserved_budget_usd),
            "updated_at": time.time(),
        })
        atomic_write_json(batch_dir / "state.json", {
            "batch_id": batch_id,
            "status": status,
            "completed": completed_tasks,
            "expected_tasks": total_tasks,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "eta_seconds": round(average_seconds * remaining, 2) if completed_tasks else None,
            "updated_at": time.time(),
        })

    _publish_progress()

    async def _run_with_semaphore(
        case: CaseSpec, model_config, run_idx: int
    ) -> RunArtifact:
        nonlocal budget_exceeded, reserved_budget_usd
        provider = model_config.provider
        queue_started = time.monotonic()
        await concurrency.acquire(provider)
        queue_latency_ms = (time.monotonic() - queue_started) * 1000
        saturated = False
        reservation = 0.0
        task_tracker = LLMCallTracker()
        try:
            async with budget_lock:
                completed = len(artifacts)
                observed_mean = tracker.total_cost_usd / completed if completed and tracker.total_cost_usd > 0 else 0.0
                reservation = max(budget_usd / task_count, observed_mean * 1.5, 0.001)
                if budget_exceeded or tracker.total_cost_usd + reserved_budget_usd + reservation > budget_usd:
                    budget_exceeded = True
                    skipped = RunArtifact(
                        run_id="skipped",
                        case_id=case.case_id,
                        model=model_config.model_id,
                        input_query=case.query,
                        normalized_output=NormalizedOutput(parse_status="failed"),
                        metrics=ExecutionMetrics(
                            execution_status="skipped_budget",
                            failure_reason="budget exceeded before launch",
                        ),
                        errors=["Skipped: budget exceeded"],
                        family=case.family,
                        difficulty=case.difficulty,
                        run_idx=run_idx,
                    )
                    return skipped
                reserved_budget_usd += reservation
            update_manifest_task(
                batch_dir,
                stable_task_key(case.case_id, model_config.model_id, run_idx),
                status="running",
                heartbeat_at=time.time(),
            )
            task_key = stable_task_key(case.case_id, model_config.model_id, run_idx)
            persisted = existing_artifacts.get(task_key)
            checkpoint_artifact = None if persisted is not None else _load_execution_checkpoint(
                batch_dir / "checkpoints" / task_key
            )
            resumable = persisted or checkpoint_artifact
            if resumable is not None and resumable.normalized_output.parse_status != "failed":
                missing_judges = [
                    judge for judge in config.judges
                    if resumable.judge_results.get(judge.name) is None
                ]
                if missing_judges:
                    return await _resume_missing_judges(
                        resumable,
                        case,
                        judges=missing_judges,
                        tracker=task_tracker,
                        concurrency_pool=concurrency,
                    )
            if case.is_multiturn and case.turns:
                execution = run_multiturn_case(
                    case,
                    model_config.model_id,
                    model_provider=model_config.provider,
                    model_temperature=model_config.temperature,
                    model_max_tokens=model_config.max_tokens,
                    vision_model_id=model_config.vision_model_id,
                    vision_provider=model_config.vision_provider,
                    judges=config.judges,
                    judge_model=config.judge_model,
                    judge_provider=config.judge_provider,
                    tracker=task_tracker,
                    concurrency_pool=concurrency,
                    run_idx=run_idx,
                    checkpoint_dir=batch_dir / "checkpoints" / stable_task_key(case.case_id, model_config.model_id, run_idx),
                    queue_latency_ms=queue_latency_ms,
                )
            else:
                execution = run_single_case(
                    case,
                    model_config.model_id,
                    model_provider=model_config.provider,
                    model_temperature=model_config.temperature,
                    model_max_tokens=model_config.max_tokens,
                    vision_model_id=model_config.vision_model_id,
                    vision_provider=model_config.vision_provider,
                    judges=config.judges,
                    judge_model=config.judge_model,
                    judge_provider=config.judge_provider,
                    tracker=task_tracker,
                    concurrency_pool=concurrency,
                    run_idx=run_idx,
                    checkpoint_dir=batch_dir / "checkpoints" / stable_task_key(case.case_id, model_config.model_id, run_idx),
                    queue_latency_ms=queue_latency_ms,
                )
            artifact = await asyncio.wait_for(execution, timeout=task_timeout_seconds)
            saturated = any(_is_saturation_error(error) for error in artifact.errors)
            return artifact
        except asyncio.TimeoutError:
            saturated = True
            return RunArtifact(
                run_id=f"timeout-{uuid.uuid4().hex[:8]}",
                case_id=case.case_id,
                model=model_config.model_id,
                input_query=case.query,
                normalized_output=NormalizedOutput(parse_status="failed"),
                metrics=ExecutionMetrics(
                    latency_ms=task_timeout_seconds * 1000,
                    execution_status="failed",
                    failure_reason=(
                        f"task timeout after {task_timeout_seconds:.0f}s"
                    ),
                ),
                errors=[
                    f"Task timeout after {task_timeout_seconds:.0f}s; no further calls launched"
                ],
                family=case.family,
                difficulty=case.difficulty,
                run_idx=run_idx,
            )
        finally:
            # A shared tracker cannot be sliced by start/end indices when
            # adaptive concurrency overlaps tasks. Keep per-artifact metrics
            # on the task-local tracker and merge calls only after completion
            # for batch totals and budget accounting.
            if task_tracker.calls:
                async with budget_lock:
                    tracker.calls.extend(task_tracker.calls)
            if reservation > 0:
                async with budget_lock:
                    reserved_budget_usd = max(0.0, reserved_budget_usd - reservation)
            await concurrency.release(provider, saturated=saturated)

    async def _persist_completed_artifact(artifact: RunArtifact) -> None:
        """Persist one task before allowing the next task to be reported."""
        task_key = stable_task_key(artifact.case_id, artifact.model, artifact.run_idx)
        judge_states = {
            name: "completed" if result is not None else "failed"
            for name, result in artifact.judge_results.items()
        }
        expected_judges = {judge.name for judge in config.judges}
        missing_judges = expected_judges - set(judge_states)
        for name in missing_judges:
            judge_states[name] = "pending"
        task_status = "completed" if (
            not expected_judges
            or (not missing_judges and all(state == "completed" for state in judge_states.values()))
        ) else "judges_partial"
        if artifact.metrics.execution_status == "skipped_budget":
            task_status = "skipped_budget"
        elif artifact.metrics.execution_status == "failed" and not artifact.normalized_output.message_md.strip():
            task_status = "failed"
        record = {
            "task_key": task_key,
            "case_id": artifact.case_id,
            "model_id": artifact.model,
            "run_idx": artifact.run_idx,
            "status": task_status,
            "judges": judge_states,
            "artifact_dir": task_key,
            "errors": artifact.errors,
            "cost_usd": artifact.metrics.estimated_cost_usd,
            "updated_at": time.time(),
        }
        atomic_write_json(batch_dir / "tasks" / f"{task_key}.json", record)
        # Keep the manifest and task record consistent under the batch lock.
        from evaluation.reporting import BatchLock
        with BatchLock(batch_dir):
            current = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
            for task in current.get("tasks", []):
                if task.get("task_key") == task_key:
                    task.update(record)
                    break
            atomic_write_json(batch_dir / "manifest.json", current)
            atomic_write_json(batch_dir / "state.json", {
                "batch_id": batch_id,
                "status": "running",
                "completed": len(artifacts),
                "expected_tasks": total_tasks,
                "updated_at": time.time(),
            })

    for coro in asyncio.as_completed(
        [_run_with_semaphore(c, m, r) for c, m, r in tasks]
    ):
        try:
            artifact = await coro
            artifacts.append(artifact)
            completed_tasks += 1

            # Persist the artifact and partial reports immediately. This is
            # intentionally done before starting the next scheduling cycle so
            # an interruption cannot lose completed work.
            partial_batch = BatchResult(
                batch_id=batch_id,
                config=config,
                cases=cases,
                artifacts=artifacts,
                tracker=tracker,
                concurrency_stats=concurrency.snapshot(),
                wall_time_ms=(time.monotonic() - progress_started) * 1000,
            )
            partial_batch.save(config.output_path)
            await _persist_completed_artifact(artifact)
            # Refresh summaries after the manifest task state is committed.
            partial_batch.save(config.output_path)

            # Budget enforcement
            if tracker.total_cost_usd >= budget_usd and not budget_exceeded:
                budget_exceeded = True
                logger.warning(
                    f"  Presupuesto alcanzado: ${tracker.total_cost_usd:.4f} >= ${budget_usd:.4f}. "
                    f"Deteniendo nuevas ejecuciones."
                )

            status = "OK" if not artifact.errors else "FAIL"
            judge_scores_str = ", ".join(
                f"{jn}={jm.overall_quality.score}" if jm else f"{jn}=N/A"
                for jn, jm in artifact.judge_results.items()
            ) if artifact.judge_results else (
                f"judge={artifact.judge_metrics.overall_quality.score}" if artifact.judge_metrics else "judge=N/A"
            )
            logger.info(
                f"  [{status}] {artifact.case_id} | model={artifact.model} | "
                f"{judge_scores_str}/5 | "
                f"cost=${tracker.total_cost_usd:.4f}"
            )
            _publish_progress()
        except Exception as exc:
            completed_tasks += 1
            logger.error(f"  Error en tarea: {exc}")
            _publish_progress()

    _publish_progress("completed")

    final_manifest = load_manifest(batch_dir)
    final_status = (
        "completed"
        if all(task.get("status") == "completed" for task in final_manifest.get("tasks", []))
        else "completed_partial"
    )
    atomic_write_json(batch_dir / "state.json", {
        "batch_id": batch_id,
        "status": final_status,
        "completed": len(artifacts),
        "expected_tasks": total_tasks,
        "updated_at": time.time(),
    })

    return BatchResult(
        batch_id=batch_id,
        config=config,
        cases=cases,
        artifacts=artifacts,
        tracker=tracker,
        concurrency_stats=concurrency.snapshot(),
        wall_time_ms=(time.monotonic() - progress_started) * 1000,
    )
