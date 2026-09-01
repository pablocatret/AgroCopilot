from __future__ import annotations

from collections import defaultdict
from typing import Iterable, TypeVar


T = TypeVar("T")


def _family_key(case: object) -> str:
    return str(
        getattr(case, "family", None)
        or getattr(case, "decision_mode", None)
        or "unknown"
    )


def stratified_case_sample(cases: Iterable[T], limit: int | None) -> list[T]:
    """Return a deterministic round-robin sample balanced by case family."""
    items = list(cases)
    if limit is None or limit >= len(items):
        return items
    if limit <= 0:
        return []

    by_family: dict[str, list[T]] = defaultdict(list)
    for item in items:
        by_family[_family_key(item)].append(item)

    selected: list[T] = []
    families = sorted(by_family)
    while len(selected) < limit and any(by_family.values()):
        for family in families:
            bucket = by_family[family]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            if len(selected) >= limit:
                return selected
    return selected
