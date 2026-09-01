from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from evaluation.schemas import CaseSpec


class EvaluationLoadError(ValueError):
    pass


def load_case(path: str | Path) -> CaseSpec:
    source = Path(path)
    if not source.exists():
        raise EvaluationLoadError(f"Case path does not exist: {source}")
    if not source.is_file():
        raise EvaluationLoadError(f"Case path is not a file: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationLoadError(f"Invalid JSON in {source}: {exc}") from exc
    try:
        return CaseSpec.model_validate(raw)
    except Exception as exc:
        raise EvaluationLoadError(f"Invalid case schema in {source}: {exc}") from exc


def load_cases(path: str | Path) -> List[CaseSpec]:
    raw = str(path)
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if len(parts) > 1:
        cases: List[CaseSpec] = []
        seen: set[str] = set()
        for part in parts:
            for case in load_cases(part):
                if case.case_id in seen:
                    continue
                seen.add(case.case_id)
                cases.append(case)
        if not cases:
            raise EvaluationLoadError(f"No JSON cases found in {raw}")
        return cases
    root = Path(path)
    if not root.exists():
        raise EvaluationLoadError(f"Cases path does not exist: {root}")
    files: Iterable[Path]
    if root.is_file():
        files = [root]
    else:
        files = sorted(root.glob("*.json"))
    files = list(files)
    if not files:
        raise EvaluationLoadError(f"No JSON cases found in {root}")
    return [load_case(file_path) for file_path in files]


def filter_cases(cases: List[CaseSpec], *, family: str | None = None) -> List[CaseSpec]:
    if not family:
        return cases
    filtered = [case for case in cases if case.family == family]
    if not filtered:
        raise EvaluationLoadError(f"No cases found for family={family}")
    return filtered
