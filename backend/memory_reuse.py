from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Iterable, Sequence

from backend.deps import settings
from libs.schemas import (
    FieldObservation,
    ImageInsights,
    MemoryReuseAssessment,
    MemoryReuseState,
    RemoteSensingMemoryArtifact,
    StacResults,
)


_RS_SIGNAL = re.compile(
    r"\b(sentinel|landsat|stac|ndvi|ndmi|teledeteccion|satelit|satelite|serie temporal|monitor|vigor|escena|imagen satelital|comparar)\b"
)
_FRESHNESS_SIGNAL = re.compile(
    r"\b(hoy|ahora|actual|actualizado|actualizada|esta semana|esta campana|ultimo|ultima|reciente|nueva imagen|nuevas imagenes|volver a analizar|reanaliz|de nuevo)\b"
)
_COMPARISON_SIGNAL = re.compile(r"\b(compar|antes|despues|cambio|evolucion|serie temporal)\b")
_DIAGNOSIS_SIGNAL = re.compile(r"\b(estres|sequia|clorosis|plaga|enfermedad|anomalia|dan[o\u0303]o)\b")
_MONITORING_SIGNAL = re.compile(r"\b(monitor|seguimiento|vigilar|vigor|alerta)\b")
_PARCEL_QUERY = re.compile(
    r"\b(?:parcela|finca|lote|zona)\s+([a-z0-9][a-z0-9\s_-]{1,60})",
    re.IGNORECASE,
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _normalize(text: str | None) -> str:
    raw = unicodedata.normalize("NFKD", str(text or ""))
    return raw.encode("ascii", "ignore").decode().lower().strip()


def _tokenize(text: str | None) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{3,}", _normalize(text))}


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_date_like(value: str | None) -> datetime | None:
    if not value:
        return None
    for candidate in (value, f"{value}T00:00:00Z"):
        parsed = _parse_dt(candidate)
        if parsed is not None:
            return parsed
    return None


def classify_remote_sensing_intent(query: str) -> str:
    normalized = _normalize(query)
    if _COMPARISON_SIGNAL.search(normalized):
        return "comparison"
    if _MONITORING_SIGNAL.search(normalized):
        return "monitoring"
    if _DIAGNOSIS_SIGNAL.search(normalized):
        return "diagnosis"
    return "general"


def query_requires_fresh_remote_sensing(query: str) -> bool:
    return bool(_FRESHNESS_SIGNAL.search(_normalize(query)))


def query_has_remote_sensing_signal(query: str) -> bool:
    return bool(_RS_SIGNAL.search(_normalize(query)))


def _latest_observation(observations: Sequence[FieldObservation | dict] | None) -> FieldObservation | None:
    if not observations:
        return None
    parsed: list[FieldObservation] = []
    for item in observations:
        try:
            parsed.append(item if isinstance(item, FieldObservation) else FieldObservation.model_validate(item))
        except Exception:
            continue
    if not parsed:
        return None
    return max(parsed, key=lambda o: str(getattr(o, "date", None) or ""), default=parsed[0])


def _extract_query_parcel(query: str) -> str | None:
    match = _PARCEL_QUERY.search(_normalize(query))
    if not match:
        return None
    value = " ".join(match.group(1).split()).strip(" -_")
    return value or None


def _coalesce_location(query: str, observations: Sequence[FieldObservation | dict] | None) -> tuple[str | None, str | None]:
    latest = _latest_observation(observations)
    query_parcel = _extract_query_parcel(query)
    if query_parcel:
        return query_parcel, latest.campaign if latest else None
    return None, None


def _observation_context_location(
    observations: Sequence[FieldObservation | dict] | None,
) -> tuple[str | None, str | None]:
    latest = _latest_observation(observations)
    if latest and latest.parcel:
        return latest.parcel, latest.campaign
    return None, None


def build_remote_sensing_artifact(
    *,
    query: str,
    decision_mode: str,
    memory_id: str | None,
    memory_name: str | None,
    observations: Sequence[FieldObservation | dict] | None,
    stac: StacResults | None,
    remote_sensing: ImageInsights | None,
    generated_at: str | None = None,
) -> RemoteSensingMemoryArtifact | None:
    stac = stac if isinstance(stac, StacResults) else None
    remote_sensing = remote_sensing if isinstance(remote_sensing, ImageInsights) else None
    if not stac and not remote_sensing:
        return None

    items = list(stac.items) if stac else []
    datetimes = sorted(item.datetime for item in items if item.datetime)
    bbox = next((item.bbox for item in items if item.bbox and len(item.bbox) == 4), None)
    parcel, campaign = _coalesce_location(query, observations)
    confidence_values = [change.confidence for change in (remote_sensing.temporal_changes if remote_sensing else []) if change.confidence is not None]
    if not confidence_values:
        confidence_values = [insight.confidence for insight in (remote_sensing.insights if remote_sensing else []) if insight.confidence is not None]
    confidence = round(sum(confidence_values) / len(confidence_values), 3) if confidence_values else None
    change_highlights = []
    limitations: list[str] = []
    if remote_sensing:
        for change in remote_sensing.temporal_changes[:4]:
            change_highlights.append(change.detail or change.label)
            for item in change.limitations:
                if item and item not in limitations:
                    limitations.append(item)
        for insight in remote_sensing.insights[:3]:
            for item in insight.limitations:
                if item and item not in limitations:
                    limitations.append(item)
    summary = ""
    if remote_sensing and remote_sensing.overview:
        summary = remote_sensing.overview
    elif items:
        summary = f"{len(items)} escena(s) STAC recuperada(s) para seguimiento remoto."
    else:
        summary = "Evidencia remota disponible en memoria."

    evidence_level = "retrieval_only"
    if remote_sensing:
        if len(remote_sensing.temporal_changes) >= 1:
            evidence_level = "analyzed_temporal"
        elif remote_sensing.insights:
            evidence_level = "analyzed_partial"

    return RemoteSensingMemoryArtifact(
        generated_at=generated_at or _now_utc().isoformat().replace("+00:00", "Z"),
        query=query,
        query_intent=classify_remote_sensing_intent(query),
        evidence_level=evidence_level,
        decision_mode=decision_mode or "case",
        memory_id=memory_id,
        memory_name=memory_name,
        parcel=parcel,
        location_hint=parcel,
        campaign=campaign,
        bbox=bbox,
        time_window_start=datetimes[0][:10] if datetimes else None,
        time_window_end=datetimes[-1][:10] if datetimes else None,
        latest_scene_date=datetimes[-1][:10] if datetimes else None,
        stac_item_ids=[item.id for item in items[:12]],
        scene_count=len(items),
        summary=summary,
        change_highlights=[item for item in change_highlights if item][:4],
        limitations=limitations[:6],
        confidence=confidence,
    )


def _intent_compatible(query_intent: str, artifact_intent: str) -> bool:
    if query_intent == artifact_intent:
        return True
    return query_intent == "general" or artifact_intent == "general"


def _location_score(parcel: str | None, artifact: RemoteSensingMemoryArtifact) -> int:
    if not parcel:
        return 0
    haystack = " ".join([artifact.parcel or "", artifact.location_hint or ""]).strip()
    if not haystack:
        return 0
    normalized_parcel = _normalize(parcel)
    normalized_haystack = _normalize(haystack)
    if normalized_parcel == normalized_haystack:
        return 4
    if normalized_parcel in normalized_haystack or normalized_haystack in normalized_parcel:
        return 3
    parcel_tokens = _tokenize(parcel)
    haystack_tokens = _tokenize(haystack)
    if parcel_tokens and parcel_tokens.intersection(haystack_tokens):
        return 2
    return 0


def _artifact_age_days(artifact: RemoteSensingMemoryArtifact, now: datetime) -> int | None:
    parsed = _parse_dt(artifact.generated_at)
    if parsed is None:
        return None
    return max(0, (now - parsed).days)


def _latest_scene_age_days(artifact: RemoteSensingMemoryArtifact, now: datetime) -> int | None:
    parsed = _parse_date_like(artifact.latest_scene_date)
    if parsed is None:
        return None
    return max(0, (now.date() - parsed.date()).days)


def resolve_remote_sensing_reuse(
    *,
    query: str,
    decision_mode: str,
    observations: Sequence[FieldObservation | dict] | None,
    artifacts: Iterable[RemoteSensingMemoryArtifact | dict] | None,
    ttl_days: int | None = None,
    now: datetime | None = None,
) -> MemoryReuseAssessment:
    now = now or _now_utc()
    ttl = ttl_days if ttl_days is not None else getattr(settings, "MEMORY_REMOTE_SENSING_TTL_DAYS", 21)
    loaded: list[RemoteSensingMemoryArtifact] = []
    for item in artifacts or []:
        try:
            loaded.append(item if isinstance(item, RemoteSensingMemoryArtifact) else RemoteSensingMemoryArtifact.model_validate(item))
        except Exception:
            continue
    if not loaded:
        return MemoryReuseAssessment(status="miss", reason="No hay artefactos de teledeteccion en memoria.")

    query_intent = classify_remote_sensing_intent(query)
    parcel, _campaign = _coalesce_location(query, observations)
    observed_parcel, _observed_campaign = _observation_context_location(observations)
    freshness_required = query_requires_fresh_remote_sensing(query)
    best: tuple[int, RemoteSensingMemoryArtifact] | None = None
    for artifact in loaded:
        score = 0
        if artifact.evidence_level == "retrieval_only":
            continue
        if _intent_compatible(query_intent, artifact.query_intent):
            score += 3
        if artifact.decision_mode == (decision_mode or "case"):
            score += 1
        explicit_location_score = _location_score(parcel, artifact)
        score += explicit_location_score
        if explicit_location_score == 0 and not parcel and observed_parcel:
            score += min(_location_score(observed_parcel, artifact), 1)
        scene_age_days = _latest_scene_age_days(artifact, now)
        if scene_age_days is not None and scene_age_days <= ttl:
            score += 2
        if best is None or score > best[0]:
            best = (score, artifact)
    if best is None:
        return MemoryReuseAssessment(status="miss", reason="No se pudo validar ningun artefacto reutilizable.")

    best_score, artifact = best
    explicit_location_score = _location_score(parcel, artifact)
    scene_age_days = _latest_scene_age_days(artifact, now)
    generated_age_days = _artifact_age_days(artifact, now)
    is_fresh = scene_age_days is not None and scene_age_days <= ttl
    if freshness_required:
        return MemoryReuseAssessment(
            status="stale",
            reason="La consulta pide evidencia remota actualizada; conviene refrescar escenas aunque exista memoria previa.",
            artifact=artifact,
        )
    if not is_fresh:
        freshness_anchor = (
            f"La ultima escena util en memoria tiene {scene_age_days} dias."
            if scene_age_days is not None
            else "La memoria remota no incluye fecha de escena utilizable."
        )
        return MemoryReuseAssessment(
            status="stale",
            reason=f"Existe evidencia remota previa, pero supera la ventana de frescura de {ttl} dias. {freshness_anchor}",
            artifact=artifact,
        )
    if (
        artifact.evidence_level == "analyzed_temporal"
        and best_score >= 6
        and explicit_location_score >= 2
    ):
        return MemoryReuseAssessment(
            status="hit",
            reason="Existe evidencia remota reciente y compatible en memoria; no hace falta relanzar STAC/RS.",
            artifact=artifact,
        )
    if generated_age_days is not None and generated_age_days > ttl * 2:
        return MemoryReuseAssessment(
            status="stale",
            reason="La evidencia remota es analiticamente util, pero el artefacto de memoria no se ha refrescado desde hace demasiado tiempo.",
            artifact=artifact,
        )
    return MemoryReuseAssessment(
        status="miss",
        reason="La memoria remota disponible no coincide con suficiente precision con la consulta actual.",
        artifact=artifact if best_score >= 4 else None,
    )


def resolve_memory_reuse_state(
    *,
    query: str,
    decision_mode: str,
    observations: Sequence[FieldObservation | dict] | None,
    remote_sensing_artifacts: Iterable[RemoteSensingMemoryArtifact | dict] | None,
    ttl_days: int | None = None,
) -> MemoryReuseState:
    return MemoryReuseState(
        remote_sensing=resolve_remote_sensing_reuse(
            query=query,
            decision_mode=decision_mode,
            observations=observations,
            artifacts=remote_sensing_artifacts,
            ttl_days=ttl_days,
        )
    )
