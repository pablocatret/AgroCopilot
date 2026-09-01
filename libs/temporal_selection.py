from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List, Optional


PREFERRED_MIN_GAP_DAYS = 30
LONG_TERM_GAP_DAYS = 180
DEFAULT_MONTH_WINDOW_DAYS = 30

_TEMPORAL_KEYWORDS = (
    "evolucion",
    "evolución",
    "compar",
    "seguimiento",
    "antes",
    "despues",
    "después",
    "temporal",
    "histori",
    "monitor",
    "cambio",
    "serie",
    "avance",
)

_RECENT_OVERRIDE_KEYWORDS = (
    "mas recientes",
    "más recientes",
    "ultimas dos",
    "últimas dos",
    "ultimos dias",
    "últimos dias",
    "ultimos días",
    "últimos días",
    "dos mas recientes",
    "dos más recientes",
    "last two",
    "latest two",
    "most recent",
    "seguidas",
    "consecutivas",
)


@dataclass(frozen=True)
class TemporalWindow:
    label: str
    datetime_range: str
    limit: int
    target_month: Optional[int] = None


@dataclass(frozen=True)
class TemporalPairChoice:
    previous: Any
    current: Any
    rationale: str
    actual_gap_days: Optional[int]
    preferred_min_gap_days: int


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def detect_temporal_intent(text: Optional[str]) -> bool:
    normalized = (text or "").lower()
    return any(token in normalized for token in _TEMPORAL_KEYWORDS)


def detect_recent_override(text: Optional[str]) -> bool:
    normalized = (text or "").lower()
    return any(token in normalized for token in _RECENT_OVERRIDE_KEYWORDS)


def build_temporal_windows(
    base_interval: Optional[str],
    *,
    now: Optional[datetime] = None,
    preferred_min_gap_days: int = PREFERRED_MIN_GAP_DAYS,
    force_recent_pair: bool = False,
    per_query_limit: int = 2,
    target_dates: Optional[List[str]] = None,
) -> list[TemporalWindow]:
    effective_now = now or datetime.now(timezone.utc)

    if target_dates:
        return _build_target_date_windows(
            target_dates,
            effective_now=effective_now,
            per_query_limit=per_query_limit,
        )

    start, end = _parse_interval(base_interval, effective_now)
    if not start or not end or start >= end:
        end = effective_now
        fallback_span_days = max(LONG_TERM_GAP_DAYS, preferred_min_gap_days * 2)
        start = effective_now - timedelta(days=fallback_span_days)

    total_days = max((end - start).days, 1)
    if total_days <= preferred_min_gap_days + 7 or force_recent_pair:
        return [TemporalWindow(label="full_range", datetime_range=_format_interval(start, end), limit=per_query_limit)]

    recent_span = min(35, max(14, total_days // 4))
    current_start = max(start, end - timedelta(days=recent_span))
    reference_end = current_start - timedelta(days=preferred_min_gap_days)
    if reference_end <= start:
        reference_end = start + max((end - start) / 2, timedelta(days=1))
    reference_span = min(45, max(14, total_days // 3))
    reference_start = max(start, reference_end - timedelta(days=reference_span))

    windows = [
        TemporalWindow(
            label="current_window",
            datetime_range=_format_interval(current_start, end),
            limit=per_query_limit,
        ),
        TemporalWindow(
            label="reference_window",
            datetime_range=_format_interval(reference_start, reference_end),
            limit=per_query_limit,
        ),
        TemporalWindow(
            label="fallback_window",
            datetime_range=_format_interval(start, end),
            limit=max(2, per_query_limit),
        ),
    ]
    deduped: list[TemporalWindow] = []
    seen: set[str] = set()
    for window in windows:
        if window.datetime_range in seen:
            continue
        seen.add(window.datetime_range)
        deduped.append(window)
    return deduped


def _build_target_date_windows(
    target_dates: List[str],
    *,
    effective_now: datetime,
    per_query_limit: int = 2,
) -> list[TemporalWindow]:
    windows: list[TemporalWindow] = []
    seen: set[str] = set()
    for idx, date_str in enumerate(target_dates):
        target_dt = parse_dt(date_str)
        if not target_dt:
            continue
        month = target_dt.month
        start = target_dt - timedelta(days=DEFAULT_MONTH_WINDOW_DAYS)
        end = target_dt + timedelta(days=DEFAULT_MONTH_WINDOW_DAYS)
        if end > effective_now:
            end = effective_now
        range_str = _format_interval(start, end)
        if range_str in seen:
            continue
        seen.add(range_str)
        windows.append(
            TemporalWindow(
                label=f"target_window_{idx + 1}",
                datetime_range=range_str,
                limit=per_query_limit,
                target_month=month,
            )
        )
    if not windows:
        fallback_start = effective_now - timedelta(days=LONG_TERM_GAP_DAYS)
        windows.append(
            TemporalWindow(
                label="fallback_window",
                datetime_range=_format_interval(fallback_start, effective_now),
                limit=max(2, per_query_limit),
            )
        )
    return windows


def expand_window(window: TemporalWindow, extra_days: int = 15) -> TemporalWindow:
    start, end = _parse_interval(window.datetime_range, datetime.now(timezone.utc))
    if not start or not end:
        return window
    new_start = start - timedelta(days=extra_days)
    new_end = end + timedelta(days=extra_days)
    return TemporalWindow(
        label=window.label,
        datetime_range=_format_interval(new_start, new_end),
        limit=window.limit,
        target_month=window.target_month,
    )


def detect_target_months(query: Optional[str]) -> Optional[List[str]]:
    import re

    normalized = (query or "").lower()
    month_map = {
        "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
        "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
        "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12",
    }
    found_months: list[tuple[int, str]] = []
    for name, num in month_map.items():
        for match in re.finditer(re.escape(name), normalized):
            pos = match.start()
            year_match = re.search(r"\b(20\d{2})\b", normalized[pos:pos + 30])
            year = year_match.group(1) if year_match else None
            if year:
                found_months.append((pos, f"{year}-{num}-15"))
    found_months.sort(key=lambda x: x[0])
    if found_months:
        return [m[1] for m in found_months]
    return None


def select_temporal_pair(
    items: Iterable[Any],
    *,
    preferred_min_gap_days: int = PREFERRED_MIN_GAP_DAYS,
    force_recent_pair: bool = False,
    selected_previous_id: Optional[str] = None,
    selected_current_id: Optional[str] = None,
    target_dates: Optional[List[str]] = None,
) -> Optional[TemporalPairChoice]:
    ordered = sorted(
        [item for item in items if parse_dt(_attr(item, "datetime")) is not None],
        key=lambda item: _attr(item, "datetime") or "",
    )
    if len(ordered) < 2:
        return None

    if selected_previous_id and selected_current_id:
        previous = next((item for item in ordered if _attr(item, "id") == selected_previous_id), None)
        current = next((item for item in ordered if _attr(item, "id") == selected_current_id), None)
        if previous is not None and current is not None:
            gap_days = _gap_days(previous, current)
            return TemporalPairChoice(
                previous=previous,
                current=current,
                rationale="Se reutiliza la pareja temporal seleccionada por STAC.",
                actual_gap_days=gap_days,
                preferred_min_gap_days=preferred_min_gap_days,
            )

    best: tuple[float, TemporalPairChoice] | None = None
    for index, previous in enumerate(ordered[:-1]):
        for current in ordered[index + 1 :]:
            score = _pair_score(
                previous,
                current,
                preferred_min_gap_days=preferred_min_gap_days,
                force_recent_pair=force_recent_pair,
                target_dates=target_dates,
            )
            rationale = _pair_rationale(
                previous,
                current,
                preferred_min_gap_days=preferred_min_gap_days,
            )
            candidate = TemporalPairChoice(
                previous=previous,
                current=current,
                rationale=rationale,
                actual_gap_days=_gap_days(previous, current),
                preferred_min_gap_days=preferred_min_gap_days,
            )
            current_dt = parse_dt(_attr(candidate.current, "datetime"))
            best_current_dt = (
                parse_dt(_attr(best[1].current, "datetime")) if best is not None else None
            )
            if (
                best is None
                or score > best[0]
                or (
                    abs(score - best[0]) < 1e-9
                    and current_dt is not None
                    and best_current_dt is not None
                    and current_dt > best_current_dt
                )
            ):
                best = (score, candidate)
    if best is not None:
        _, choice = best
        prev_collection = _attr(choice.previous, "collection")
        curr_collection = _attr(choice.current, "collection")
        if prev_collection and curr_collection and prev_collection != curr_collection:
            return None
        prev_index = _attr(choice.previous, "index_name")
        curr_index = _attr(choice.current, "index_name")
        if prev_index and curr_index and prev_index != curr_index:
            return None
    return best[1] if best else None


def dedupe_items_by_id(items: Iterable[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[str] = set()
    for item in items:
        item_id = str(_attr(item, "id") or "")
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        deduped.append(item)
    return deduped


def _pair_score(
    previous: Any,
    current: Any,
    *,
    preferred_min_gap_days: int,
    force_recent_pair: bool,
    target_dates: Optional[List[str]] = None,
) -> float:
    score = 0.0
    previous_dt = parse_dt(_attr(previous, "datetime"))
    current_dt = parse_dt(_attr(current, "datetime"))
    if not previous_dt or not current_dt:
        return -999.0

    gap_days = max((current_dt - previous_dt).days, 0)
    same_collection = _attr(previous, "collection") == _attr(current, "collection")
    prev_index = _attr(previous, "index_name")
    curr_index = _attr(current, "index_name")
    same_index = bool(prev_index and curr_index and prev_index == curr_index)
    if same_collection:
        score += 5.0
    else:
        score -= 4.0
    if same_index:
        score += 2.5
    elif prev_index and curr_index and prev_index != curr_index:
        score -= 2.5

    score += _quality_score(previous) + _quality_score(current)
    if _has_metric(previous):
        score += 1.2
    if _has_metric(current):
        score += 1.2

    recency_bonus = min((datetime.now(timezone.utc) - current_dt).days, 120)
    score += max(0.0, 2.4 - recency_bonus / 45)

    if force_recent_pair:
        score += max(0.0, 2.2 - gap_days / 12)
    else:
        if gap_days >= preferred_min_gap_days:
            score += min(4.0, gap_days / 12)
        else:
            score -= (preferred_min_gap_days - gap_days) * 0.45
        if gap_days > 180:
            score -= min(2.0, (gap_days - 180) / 45)

    if target_dates and len(target_dates) >= 2:
        try:
            target_prev = parse_dt(target_dates[0])
            target_curr = parse_dt(target_dates[1])
            if target_prev and target_curr and previous_dt and current_dt:
                prev_doy = previous_dt.timetuple().tm_yday
                curr_doy = current_dt.timetuple().tm_yday
                target_prev_doy = target_prev.timetuple().tm_yday
                target_curr_doy = target_curr.timetuple().tm_yday
                prev_month_gap = min(
                    abs(prev_doy - target_prev_doy),
                    365 - abs(prev_doy - target_prev_doy),
                )
                curr_month_gap = min(
                    abs(curr_doy - target_curr_doy),
                    365 - abs(curr_doy - target_curr_doy),
                )
                if prev_month_gap <= 15:
                    score += 3.0
                elif prev_month_gap <= 30:
                    score += 1.5
                else:
                    score -= min(2.0, prev_month_gap / 30)
                if curr_month_gap <= 15:
                    score += 3.0
                elif curr_month_gap <= 30:
                    score += 1.5
                else:
                    score -= min(2.0, curr_month_gap / 30)
        except Exception:
            pass

    return score


def _pair_rationale(previous: Any, current: Any, *, preferred_min_gap_days: int) -> str:
    gap_days = _gap_days(previous, current)
    current_dt = _attr(current, "datetime") or "fecha reciente"
    previous_dt = _attr(previous, "datetime") or "fecha previa"
    collection = _attr(current, "collection") or _attr(previous, "collection") or "coleccion compatible"
    if gap_days is None:
        return f"Se comparan escenas compatibles de {collection} entre {previous_dt} y {current_dt}."
    if gap_days < preferred_min_gap_days:
        return (
            f"Se comparan escenas de {collection} separadas por {gap_days} dias; "
            "sirve como referencia rapida, pero no como evolucion robusta."
        )
    return (
        f"Se comparan escenas de {collection} entre {previous_dt} y {current_dt}, "
        f"con una separacion de {gap_days} dias."
    )


def _parse_interval(
    value: Optional[str], fallback_end: datetime
) -> tuple[Optional[datetime], Optional[datetime]]:
    if not value or "/" not in value:
        return None, None
    raw_start, raw_end = value.split("/", 1)
    end = parse_dt(raw_end) if raw_end else fallback_end
    start = parse_dt(raw_start) if raw_start else (end - timedelta(days=180) if end else None)
    return start, end


def _format_interval(start: datetime, end: datetime) -> str:
    return f"{start.isoformat()}/{end.isoformat()}"


def _gap_days(previous: Any, current: Any) -> Optional[int]:
    previous_dt = parse_dt(_attr(previous, "datetime"))
    current_dt = parse_dt(_attr(current, "datetime"))
    if not previous_dt or not current_dt:
        return None
    return max((current_dt - previous_dt).days, 0)


def _quality_score(item: Any) -> float:
    label = _attr(_attr(item, "quality"), "label") or "desconocida"
    return {
        "alta": 1.1,
        "media": 0.5,
        "baja": -0.8,
        "desconocida": 0.0,
    }.get(str(label), 0.0)


def _has_metric(item: Any) -> bool:
    stats = _attr(item, "index_stats")
    if stats is None:
        return False
    mean = _attr(stats, "mean")
    valid_pixels = _attr(stats, "valid_pixels") or 0
    return mean is not None and valid_pixels > 0


def _attr(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)
