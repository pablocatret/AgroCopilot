from __future__ import annotations

import argparse
import json
import re
from typing import Any


MIGRATION_VERSION = 1


def _field(text: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}\s*:\s*(.+)", text or "", re.IGNORECASE)
    return " ".join(match.group(1).split()) if match else ""


def _priority(value: str) -> str:
    return {"alta": "high", "baja": "low"}.get(value, "medium")


def _status(value: str) -> str:
    return {"bloqueada": "blocked", "hecha": "done"}.get(value, "open")


def _memory_context(sections: dict[str, str]) -> dict[str, str]:
    profile = sections.get("profile", "")
    preferences = sections.get("preferences", "")
    farm_context = sections.get("farm_context", "")
    return {
        "name": _field(profile, "Nombre o alias"),
        "zone": _field(profile, "Zona geografica"),
        "crops": _field(farm_context, "Cultivos principales"),
        "infrastructure": _field(farm_context, "Infraestructura relevante"),
        "constraints": _field(preferences, "Restricciones"),
        "preferences": " | ".join(
            value
            for value in (
                _field(preferences, "Objetivo principal"),
                _field(preferences, "Nivel de detalle preferido"),
            )
            if value
        ),
    }


def _import_memory(*, user_id: str, memory: Any, memory_store: Any, case_store: Any, archived: bool) -> str | None:
    memory_id = memory.memory_id
    sections = memory.sections
    case_state = memory_store.load_case_state(user_id, memory_id)
    history = memory_store.load_case_history(user_id, memory_id)
    observations = memory_store.load_observations(user_id, memory_id)
    decisions = memory_store.load_decision_log(user_id, memory_id)
    open_questions = [
        line.strip()[2:].strip()
        for line in sections.get("open_questions", "").splitlines()
        if line.strip().startswith("- ") and line.strip()[2:].strip()
    ]
    has_case_data = bool(
        case_state.case_summary
        or case_state.open_tasks
        or observations
        or history
        or decisions
        or open_questions
    )
    if not has_case_data:
        return None

    title = case_state.case_summary[:120] or memory.memory_name or "Seguimiento importado"
    case = case_store.create_case(
        workspace_id=user_id,
        title=title,
        objective=case_state.case_summary[:500],
    )
    case_id = case["case_id"]
    if archived:
        case_store.set_case_status(case_id, workspace_id=user_id, status="archived", actor_type="migration")
    for task in case_state.open_tasks:
        case_store.create_task(
            case_id=case_id,
            workspace_id=user_id,
            title=task.title,
            rationale=task.rationale,
            priority=_priority(task.priority),
            status=_status(task.status),
            actor_type="migration",
        )
    for question in open_questions:
        case_store.create_task(
            case_id=case_id,
            workspace_id=user_id,
            title=question,
            rationale="Pregunta abierta importada de la continuidad anterior.",
            priority="medium",
            status="proposed",
            actor_type="migration",
        )
    for blocker in case_state.blocked_by:
        case_store.append_event(
            case_id,
            event_type="legacy_blocker_imported",
            actor_type="migration",
            payload={"text": blocker},
        )
    for item in history:
        case_store.append_event(
            case_id,
            event_type="legacy_case_history_imported",
            actor_type="migration",
            payload=item.model_dump() if hasattr(item, "model_dump") else dict(item),
        )
    for item in decisions:
        case_store.append_event(
            case_id,
            event_type="legacy_decision_imported",
            actor_type="migration",
            payload=item,
        )
    for item in observations:
        case_store.create_observation(
            case_id=case_id,
            workspace_id=user_id,
            date=item.date,
            parcel=item.parcel,
            campaign=item.campaign,
            note=item.note,
            severity=item.severity,
            actor_type="migration",
        )
    return case_id


def migrate_legacy_memory(user_id: str, *, memory_store: Any, case_store: Any) -> dict[str, Any]:
    """Import all legacy memories into compact context plus follow-ups once."""
    if case_store.migration_version(user_id) >= MIGRATION_VERSION:
        return {"migrated": False, "case_id": None, "version": MIGRATION_VERSION}

    memories = memory_store.list_memories(user_id)
    if not memories:
        memories = [memory_store.load(user_id)]
    current_id = memory_store.get_current_memory_id(user_id)
    ordered = sorted(memories, key=lambda item: item.memory_id != current_id)
    context: dict[str, str] = {}
    case_ids: list[str] = []
    for item in ordered:
        memory = memory_store.load(user_id, item.memory_id)
        for key, value in _memory_context(memory.sections).items():
            if value and not context.get(key):
                context[key] = value
        case_id = _import_memory(
            user_id=user_id,
            memory=memory,
            memory_store=memory_store,
            case_store=case_store,
            archived=item.memory_id != current_id,
        )
        if case_id:
            case_ids.append(case_id)
    case_store.save_workspace_context(user_id, context)
    case_store.mark_migration(user_id, MIGRATION_VERSION)
    return {
        "migrated": True,
        "case_id": case_ids[0] if case_ids else None,
        "case_ids": case_ids,
        "version": MIGRATION_VERSION,
    }


def main() -> None:
    from backend.case_store import case_store
    from backend.memory_store import memory_store

    parser = argparse.ArgumentParser(description="Import legacy memory continuity into CaseStore.")
    parser.add_argument("user_ids", nargs="+", help="Workspace/user IDs to migrate")
    args = parser.parse_args()
    results = [migrate_legacy_memory(user_id, memory_store=memory_store, case_store=case_store) for user_id in args.user_ids]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
