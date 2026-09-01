from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


_CONTINUITY_TERMS = (
    "seguimiento",
    "continuar",
    "evolucion",
    "comparar",
    "parcela",
    "campana",
    "observacion",
    "antes",
    "ahora",
    "pendiente",
    "expediente",
    "revisar de nuevo",
)


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def has_continuity_signal(
    query: str,
    *,
    attachment_count: int = 0,
    observations: int = 0,
    case_state: Any = None,
) -> bool:
    """Return whether the turn contains enough signal to retain a follow-up."""
    if attachment_count > 0 or observations > 0:
        return True
    if case_state is not None:
        if getattr(case_state, "open_tasks", None) or getattr(case_state, "blocked_by", None):
            return True
        if isinstance(case_state, dict) and (
            case_state.get("open_tasks") or case_state.get("blocked_by")
        ):
            return True
    normalized = _normalize(query)
    return any(term in normalized for term in _CONTINUITY_TERMS)


def should_create_case(
    query: str,
    *,
    attachment_count: int = 0,
    observations: int = 0,
    case_state: Any = None,
) -> bool:
    return has_continuity_signal(
        query,
        attachment_count=attachment_count,
        observations=observations,
        case_state=case_state,
    )


@dataclass(frozen=True)
class CaseResolution:
    case_id: str | None = None
    reason: str = "none"
    candidates: list[str] = field(default_factory=list)


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalize(value))
        if len(token) >= 4
    }


def resolve_case(
    cases: list[dict[str, Any]],
    query: str,
    *,
    explicit_case_id: str | None = None,
    linked_case_id: str | None = None,
) -> CaseResolution:
    """Resolve a follow-up without silently selecting an ambiguous case."""
    by_id = {str(item.get("case_id")): item for item in cases if item.get("case_id")}
    if explicit_case_id and explicit_case_id in by_id:
        return CaseResolution(explicit_case_id, "explicit")
    if (
        linked_case_id
        and linked_case_id in by_id
        and by_id[linked_case_id].get("status") in {"active", "on_hold"}
    ):
        return CaseResolution(linked_case_id, "conversation")
    if not has_continuity_signal(query):
        return CaseResolution()

    query_tokens = _tokens(query)
    scored: list[tuple[int, str]] = []
    for item in cases:
        if item.get("status") not in {"active", "on_hold"}:
            continue
        haystack = " ".join(
            str(item.get(field) or "") for field in ("title", "objective", "summary")
        )
        score = len(query_tokens & _tokens(haystack))
        if score:
            scored.append((score, str(item["case_id"])))
    if not scored:
        return CaseResolution()
    scored.sort(reverse=True)
    best_score = scored[0][0]
    best = [case_id for score, case_id in scored if score == best_score]
    if len(best) > 1:
        return CaseResolution(reason="ambiguous", candidates=best)
    return CaseResolution(best[0], "matched")
