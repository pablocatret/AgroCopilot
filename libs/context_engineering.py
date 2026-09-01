from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Iterable, Mapping
import json


def truncate_text(text: str, max_chars: int) -> str:
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    head = max_chars // 2
    tail = max_chars - head - 1
    return f"{value[:head].rstrip()}…{value[-tail:].lstrip()}"


def summarize_attachments(
    attachments: Iterable[Any], *, max_items: int = 6, max_summary_chars: int = 180
) -> str:
    rows: list[str] = []
    for idx, item in enumerate(list(attachments)[:max_items], start=1):
        filename = (
            getattr(item, "filename", None) or getattr(item, "name", None) or f"adjunto-{idx}"
        )
        content_type = getattr(item, "content_type", None) or "desconocido"
        summary = getattr(item, "summary", None) or ""
        extracted = getattr(item, "extracted_text", None) or ""
        text_len = len(extracted) if extracted else 0
        size_hint = f" ({text_len} chars extraídos)" if text_len > 200 else ""
        detail = summary or extracted
        detail = truncate_text(detail, max_summary_chars) if detail else "Sin resumen extraído"
        rows.append(f"- {filename} ({content_type}){size_hint}: {detail}")
    return "\n".join(rows) if rows else "Sin adjuntos relevantes."


def summarize_memory_context(memory_context: str, *, max_chars: int = 1200) -> str:
    value = (memory_context or "").strip()
    if not value:
        return "Sin memoria de usuario disponible."
    if len(value) <= max_chars:
        return value
    lines = value.splitlines()
    header_indices: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if line.startswith("# "):
            header_indices.append((i, line[2:].strip().lower()))
    priority_headers = {"preguntas abiertas", "open_questions", "estado de caso actual", "current_case", "contexto agronómico", "farm_context"}
    priority_lines: list[str] = []
    other_lines: list[str] = []
    if header_indices:
        for idx, (start, header) in enumerate(header_indices):
            end = header_indices[idx + 1][0] if idx + 1 < len(header_indices) else len(lines)
            block = "\n".join(lines[start:end]).strip()
            if not block:
                continue
            if any(p in header for p in priority_headers):
                priority_lines.append(block)
            else:
                other_lines.append(block)
    else:
        other_lines = [value]
    priority_text = "\n\n".join(priority_lines)
    other_text = "\n\n".join(other_lines)
    priority_budget = min(len(priority_text), max_chars // 2)
    other_budget = max_chars - priority_budget
    result_parts: list[str] = []
    if priority_text:
        result_parts.append(truncate_text(priority_text, priority_budget))
    if other_text:
        result_parts.append(truncate_text(other_text, other_budget))
    return "\n\n".join(result_parts) if result_parts else "Sin memoria de usuario disponible."


def summarize_memory_reuse(memory_reuse: Any, *, max_chars: int = 700) -> str:
    if not memory_reuse:
        return "Sin reutilizacion de memoria estructurada."
    try:
        if hasattr(memory_reuse, "model_dump"):
            payload = memory_reuse.model_dump(exclude_none=True)
        elif isinstance(memory_reuse, Mapping):
            payload = dict(memory_reuse)
        else:
            return truncate_text(str(memory_reuse), max_chars)
    except Exception:
        return truncate_text(str(memory_reuse), max_chars)
    rs = payload.get("remote_sensing") if isinstance(payload, Mapping) else None
    if not isinstance(rs, Mapping):
        return truncate_text(json.dumps(payload, ensure_ascii=False), max_chars)
    status = str(rs.get("status") or "miss")
    reason = str(rs.get("reason") or "").strip()
    artifact = rs.get("artifact") if isinstance(rs.get("artifact"), Mapping) else {}
    parts = [f"remote_sensing={status}"]
    if artifact:
        summary = str(artifact.get("summary") or "").strip()
        parcel = str(artifact.get("parcel") or artifact.get("location_hint") or "").strip()
        latest_scene = str(artifact.get("latest_scene_date") or "").strip()
        if parcel:
            parts.append(f"parcela={parcel}")
        if latest_scene:
            parts.append(f"ultima_escena={latest_scene}")
        if summary:
            parts.append(f"resumen={summary}")
    if reason:
        parts.append(f"motivo={reason}")
    return truncate_text(" | ".join(parts), max_chars)


def summarize_conversation_history(
    history: Any,
    *,
    max_turns: int = 6,
    max_item_chars: int = 180,
    max_chars: int = 1200,
) -> str:
    if not isinstance(history, list) or not history:
        return "Sin historial conversacional previo."
    rows: list[str] = []
    for item in history[-max_turns:]:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "desconocido").strip().lower()
        if role == "user":
            text = str(item.get("query") or "").strip()
        else:
            text = str(item.get("answer_summary") or item.get("query") or "").strip()
        if not text:
            continue
        label = "Usuario" if role == "user" else "Asistente"
        rows.append(f"- {label}: {truncate_text(text, max_item_chars)}")
    if not rows:
        return "Sin historial conversacional previo."
    return truncate_text("\n".join(rows), max_chars)


def summarize_case_history(
    case_history: Iterable[Any], *, max_items: int = 4, max_chars: int = 180
) -> str:
    rows: list[str] = []
    for item in list(case_history)[:max_items]:
        title = (
            getattr(item, "title", None)
            or (item.get("title") if isinstance(item, Mapping) else None)
            or "Caso"
        )
        mode = (
            getattr(item, "decision_mode", None)
            or (item.get("decision_mode") if isinstance(item, Mapping) else None)
            or "decision"
        )
        summary = (
            getattr(item, "summary", None)
            or (item.get("summary") if isinstance(item, Mapping) else None)
            or ""
        )
        rows.append(
            f"- {title} [{mode}]: {truncate_text(str(summary), max_chars) if summary else 'Sin resumen'}"
        )
    return "\n".join(rows) if rows else "Sin historial reciente de expedientes."


def summarize_observations(
    observations: Iterable[Any], *, max_items: int = 5, max_chars: int = 160
) -> str:
    rows: list[str] = []
    for item in list(observations)[:max_items]:
        date = (
            getattr(item, "date", None)
            or (item.get("date") if isinstance(item, Mapping) else None)
            or "sin-fecha"
        )
        parcel = (
            getattr(item, "parcel", None)
            or (item.get("parcel") if isinstance(item, Mapping) else None)
            or "parcela no indicada"
        )
        severity = (
            getattr(item, "severity", None)
            or (item.get("severity") if isinstance(item, Mapping) else None)
            or "media"
        )
        note = (
            getattr(item, "note", None)
            or (item.get("note") if isinstance(item, Mapping) else None)
            or ""
        )
        rows.append(
            f"- {date} · {parcel} [{severity}]: {truncate_text(str(note), max_chars) if note else 'Sin nota'}"
        )
    return "\n".join(rows) if rows else "Sin observaciones de campo recientes."


def summarize_temporal_focus(observations: Iterable[Any], *, fallback_days: int = 60) -> str:
    parsed_dates: list[date] = []
    for item in observations:
        raw = getattr(item, "date", None) or (
            item.get("date") if isinstance(item, Mapping) else None
        )
        if not raw:
            continue
        try:
            parsed_dates.append(date.fromisoformat(str(raw)))
        except ValueError:
            continue
    if not parsed_dates:
        today = date.today()
        start = today - timedelta(days=fallback_days)
        return f"Si el usuario no fija fechas, prioriza escenas recientes dentro de {start.isoformat()}/{today.isoformat()}."
    latest = max(parsed_dates)
    start = latest - timedelta(days=30)
    earlier = latest - timedelta(days=90)
    return (
        f"Hay observaciones con fecha relevante alrededor de {latest.isoformat()}. "
        f"Si el usuario no fija periodo, prioriza una comparación reciente entre {start.isoformat()}/{latest.isoformat()} "
        f"y una referencia anterior alrededor de {earlier.isoformat()}/{start.isoformat()}."
    )


_MONITORING_KEYWORDS: dict[str, tuple[str, ...]] = {
    "es": (
        "seguimiento", "evolución", "evolucion", "comparar", "comparación",
        "comparacion", "antes", "ahora", "temporal", "serie", "parcela",
        "campaña", "campana", "revisar", "cambio",
    ),
    "en": (
        "monitoring", "evolution", "compare", "comparison", "before", "after",
        "temporal", "series", "parcel", "campaign", "review", "change",
    ),
}

_MONITORING_PHRASES: dict[str, tuple[str, ...]] = {
    "es": ("revisar de nuevo",),
    "en": ("review again",),
}


def _has_monitoring_signal(query: str, keywords: tuple[str, ...], phrases: tuple[str, ...]) -> bool:
    q = (query or "").lower()
    for phrase in phrases:
        if phrase in q:
            return True
    for token in keywords:
        if re.search(r"\b" + re.escape(token) + r"\b", q):
            return True
    return False


def summarize_monitoring_signal(
    query: str,
    observations: Iterable[Any],
    case_history: Iterable[Any],
    language: str = "es",
) -> str:
    lang = (language or "es").strip().lower()[:2]
    keywords = _MONITORING_KEYWORDS.get(lang, _MONITORING_KEYWORDS["es"])
    phrases = _MONITORING_PHRASES.get(lang, _MONITORING_PHRASES["es"])
    query_has_signal = _has_monitoring_signal(query, keywords, phrases)
    has_observations = any(True for _ in list(observations)[:1])
    has_history = any(True for _ in list(case_history)[:1])
    if query_has_signal and has_observations:
        return "El caso parece de seguimiento parcelario con observaciones previas; prioriza contraste temporal y comparación antes/ahora."
    if has_observations and has_history:
        return "Hay señales acumuladas de campaña y expedientes previos; considera enfoque de monitoring aunque el usuario no lo formule explícitamente."
    if has_observations:
        return "Existen observaciones recientes de campo; puede aportar valor una revisión temporal o comparativa."
    return "Sin señal fuerte de seguimiento temporal."


def summarize_execution_report(execution_report: Mapping[str, Any], *, max_agents: int = 8) -> str:
    rows: list[str] = []
    for agent, info in list(execution_report.items())[:max_agents]:
        level = str(info.get("final_level", "ok"))
        instances = info.get("instances", []) or []
        details = []
        for item in instances[:2]:
            message = str(item.get("message", "")).strip()
            if message:
                details.append(truncate_text(message, 140))
        suffix = f" ({'; '.join(details)})" if details else ""
        rows.append(f"- {agent}: {level}{suffix}")
    return "\n".join(rows) if rows else "Sin incidencias de ejecución registradas."


def summarize_refs(refs: Iterable[Any], *, max_items: int = 6, max_snippet_chars: int = 180) -> str:
    rows: list[str] = []
    for idx, ref in enumerate(list(refs)[:max_items], start=1):
        title = getattr(ref, "title", None) or f"Fuente {idx}"
        source = getattr(ref, "source", None) or "referencia"
        snippet = getattr(ref, "snippet", None) or ""
        snippet = truncate_text(snippet, max_snippet_chars) if snippet else "Sin fragmento"
        rows.append(f"- [{source}] {title}: {snippet}")
    return "\n".join(rows) if rows else "Sin referencias consolidadas."


def summarize_agent_context_blocks(
    context: Mapping[str, Any],
    *,
    max_blocks: int = 8,
    max_block_chars: int = 900,
) -> str:
    def compact_value(value: Any) -> str:
        if value is None:
            return "Sin datos"
        if hasattr(value, "model_dump"):
            try:
                value = value.model_dump(exclude_none=True)
            except Exception:
                value = str(value)
        if isinstance(value, Mapping):
            try:
                return truncate_text(
                    json.dumps(value, ensure_ascii=False, sort_keys=True), max_block_chars
                )
            except Exception:
                return truncate_text(str(value), max_block_chars)
        if isinstance(value, (list, tuple)):
            try:
                return truncate_text(json.dumps(value, ensure_ascii=False), max_block_chars)
            except Exception:
                return truncate_text(str(value), max_block_chars)
        return truncate_text(str(value), max_block_chars)

    rows: list[str] = []
    for name, output in context.items():
        if name.startswith("_") or output is None or name == "rs_analyst":
            continue
        summary = getattr(output, "summary", "") or ""
        data = getattr(output, "data", None)
        block = compact_value(data if data is not None else summary)
        rows.append(
            f"[SOURCE: {name.upper()}]\nResumen: {summary or 'Sin resumen'}\nContexto: {block}"
        )
        if len(rows) >= max_blocks:
            break
    return "\n\n".join(rows) if rows else "Sin contexto de agentes."
