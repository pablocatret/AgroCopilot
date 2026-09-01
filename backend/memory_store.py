from __future__ import annotations

import json
import logging
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import filelock

from backend.deps import settings
from libs.schemas import (
    CaseSnapshot,
    CaseState,
    CaseTask,
    FieldObservation,
    MemoryListItem,
    MemoryMeta,
    RemoteSensingMemoryArtifact,
)


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# File locking and atomic write helpers
# ------------------------------------------------------------------

@contextmanager
def _file_lock(path: Path):
    """Cross-platform file lock using filelock (fcntl on Linux, msvcrt on Windows)."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock = filelock.FileLock(str(lock_path))
    with lock:
        yield


def _atomic_write(path: Path, content: str) -> None:
    """Write to a temp file then rename for crash safety."""
    import os
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _split_by_heading(text: str, template: str) -> List[str]:
    """Split markdown by ## headings, returning complete blocks."""
    if not text or text.strip() == template.strip():
        return []
    blocks: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.startswith("## ") and current:
            blocks.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return blocks


SECTION_FILES = {
    "profile": "profile.md",
    "preferences": "preferences.md",
    "farm_context": "farm_context.md",
    "open_questions": "open_questions.md",
    "decision_log": "decision_log.md",
    "current_case": "current_case.md",
    "case_history": "case_history.md",
    "field_observations": "field_observations.md",
}
SECTION_STRUCTURED = {"current_case", "case_history", "field_observations", "decision_log"}
LEGACY_CASE_SECTIONS = {"current_case", "case_history", "field_observations", "decision_log"}
EDITABLE_SECTIONS = ("profile", "preferences", "farm_context", "open_questions")

SECTION_TEMPLATES = {
    "profile": "# Perfil del usuario\n\n- Nombre o alias:\n- Tipo de explotacion:\n- Zona geografica:\n",
    "preferences": "# Preferencias operativas\n\n- Objetivo principal:\n- Restricciones:\n- Nivel de detalle preferido:\n",
    "farm_context": "# Contexto agronomico\n\n- Cultivos principales:\n- Campana actual:\n- Infraestructura relevante:\n",
    "open_questions": "# Preguntas abiertas\n\n- Sin registrar.\n",
    "decision_log": "# Historial de decisiones\n\n",
    "current_case": "# Estado de caso actual\n\n## Resumen\n\nSin caso activo.\n",
    "case_history": "# Historial de expedientes\n\n",
    "field_observations": "# Observaciones de campo\n\n",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", (value or "").strip()).strip("-").lower()
    cleaned = re.sub(r"\.\.", "-", cleaned)
    return cleaned or "default"


class UserMemory:
    def __init__(
        self,
        user_id: str,
        memory_id: str,
        sections: Dict[str, str],
        used_sections: List[str],
        memory_name: str | None = None,
    ) -> None:
        self.user_id = user_id
        self.memory_id = memory_id
        self.sections = sections
        self.used_sections = used_sections
        self.memory_name = memory_name


class UserMemoryStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path(settings.MEMORY_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _user_dir(self, user_id: str) -> Path:
        return self.base_dir / _slugify(user_id)

    def _current_json_path(self, user_id: str) -> Path:
        return self._user_dir(user_id) / "current.json"

    def _memory_dir(self, user_id: str, memory_id: str) -> Path:
        return self._user_dir(user_id) / memory_id

    def _structured_path(self, user_id: str, memory_id: str, section: str) -> Path:
        filename = SECTION_FILES[section].replace(".md", ".json")
        return self._memory_dir(user_id, memory_id) / filename

    def _remote_sensing_artifacts_path(self, user_id: str, memory_id: str) -> Path:
        return self._memory_dir(user_id, memory_id) / "remote_sensing_artifacts.json"

    def _memory_name(self, user_id: str, memory_id: str) -> str:
        current = self._ensure_current(user_id)
        if current.memory_id == memory_id and current.name:
            return current.name
        meta_path = self._memory_dir(user_id, memory_id) / ".meta.json"
        if meta_path.exists():
            try:
                meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
                return str(meta_data.get("name") or memory_id)
            except Exception:
                logger.warning("memory.meta_json_corrupt", extra={"user_id": user_id, "memory_id": memory_id})
        return "Mi memoria" if memory_id == "default" else memory_id

    # ------------------------------------------------------------------
    # Legacy migration
    # ------------------------------------------------------------------

    def _is_legacy_user(self, user_id: str) -> bool:
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            return False
        has_current_json = (user_dir / "current.json").exists()
        has_any_memory_dir = any(
            (user_dir / item.name).is_dir() and (user_dir / item.name / "profile.md").exists()
            for item in user_dir.iterdir()
            if item.is_dir()
        )
        has_profile_at_root = (user_dir / "profile.md").exists()
        return not has_current_json and not has_any_memory_dir and has_profile_at_root

    def _migrate_legacy(self, user_id: str) -> str:
        user_dir = self._user_dir(user_id)
        default_dir = user_dir / "default"
        default_dir.mkdir(parents=True, exist_ok=True)

        for section, filename in SECTION_FILES.items():
            src = user_dir / filename
            dst = default_dir / filename
            if src.exists() and not dst.exists():
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        structured_files = list(user_dir.glob("*.json"))
        for sf in structured_files:
            dst = default_dir / sf.name
            if not dst.exists():
                dst.write_text(sf.read_text(encoding="utf-8"), encoding="utf-8")

        meta = {
            "memory_id": "default",
            "name": "Mi memoria",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        (user_dir / "current.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info("memory.migrated_legacy", extra={"user_id": user_id})
        return "default"

    # ------------------------------------------------------------------
    # current.json helpers
    # ------------------------------------------------------------------

    def _ensure_current(self, user_id: str) -> MemoryMeta:
        user_dir = self._user_dir(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        path = self._current_json_path(user_id)

        if self._is_legacy_user(user_id):
            self._migrate_legacy(user_id)

        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return MemoryMeta(
                    memory_id=data["memory_id"],
                    name=data.get("name", ""),
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                )
            except Exception:
                logger.warning("memory.current_json_corrupt", extra={"user_id": user_id})

        memory_id = "default"
        meta = MemoryMeta(
            memory_id=memory_id,
            name="Mi memoria",
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        self._ensure_memory_files(user_id, memory_id)
        path.write_text(
            json.dumps(meta.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return meta

    def _write_current(self, user_id: str, meta: MemoryMeta) -> None:
        path = self._current_json_path(user_id)
        data = meta.model_dump()
        data["updated_at"] = _now_iso()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Memory files helpers
    # ------------------------------------------------------------------

    def _ensure_memory_files(self, user_id: str, memory_id: str) -> None:
        mem_dir = self._memory_dir(user_id, memory_id)
        mem_dir.mkdir(parents=True, exist_ok=True)
        for section, filename in SECTION_FILES.items():
            path = mem_dir / filename
            if not path.exists():
                path.write_text(SECTION_TEMPLATES[section], encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_memories(self, user_id: str) -> List[MemoryListItem]:
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            return []

        current = self._ensure_current(user_id)
        items: List[MemoryListItem] = []

        for child in sorted(user_dir.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "profile.md").exists():
                continue
            mem_id = child.name
            meta_path = child.parent / f"{mem_id}" / ".meta.json"
            name = mem_id
            if (child / ".meta.json").exists():
                try:
                    meta_data = json.loads((child / ".meta.json").read_text(encoding="utf-8"))
                    name = meta_data.get("name", mem_id)
                except Exception:
                    pass

            sections = {}
            used = []
            for section, filename in SECTION_FILES.items():
                content = (child / filename).read_text(encoding="utf-8").strip()
                sections[section] = content
                if content and content != SECTION_TEMPLATES[section].strip():
                    used.append(section)

            items.append(MemoryListItem(
                memory_id=mem_id,
                name=name,
                is_current=mem_id == current.memory_id,
                used_sections=used,
            ))

        return items

    def get_current_memory_id(self, user_id: str) -> str:
        meta = self._ensure_current(user_id)
        return meta.memory_id

    def set_current_memory(self, user_id: str, memory_id: str) -> MemoryMeta:
        mem_dir = self._memory_dir(user_id, memory_id)
        if not mem_dir.exists() or not (mem_dir / "profile.md").exists():
            raise ValueError(f"Memory '{memory_id}' not found for user '{user_id}'")

        meta = self._ensure_current(user_id)
        meta.memory_id = memory_id
        meta.name = self._memory_name(user_id, memory_id)
        meta.updated_at = _now_iso()
        self._write_current(user_id, meta)
        return meta

    def create_memory(self, user_id: str, name: str) -> MemoryMeta:
        memory_id = uuid.uuid4().hex[:12]
        self._ensure_memory_files(user_id, memory_id)

        meta = MemoryMeta(
            memory_id=memory_id,
            name=name.strip() or memory_id,
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        mem_dir = self._memory_dir(user_id, memory_id)
        (mem_dir / ".meta.json").write_text(
            json.dumps(meta.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        current = self._ensure_current(user_id)
        current.memory_id = memory_id
        current.name = meta.name
        current.updated_at = _now_iso()
        self._write_current(user_id, current)

        return meta

    def rename_memory(self, user_id: str, memory_id: str, new_name: str) -> MemoryMeta:
        mem_dir = self._memory_dir(user_id, memory_id)
        if not mem_dir.exists():
            raise ValueError(f"Memory '{memory_id}' not found")

        now = _now_iso()
        existing_created = now
        meta_path = mem_dir / ".meta.json"
        if meta_path.exists():
            try:
                old = MemoryMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
                existing_created = old.created_at
            except Exception:
                pass
        meta = MemoryMeta(
            memory_id=memory_id,
            name=new_name.strip() or memory_id,
            created_at=existing_created,
            updated_at=now,
        )
        (mem_dir / ".meta.json").write_text(
            json.dumps(meta.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        current = self._ensure_current(user_id)
        if current.memory_id == memory_id:
            current.name = meta.name
            current.updated_at = now
            self._write_current(user_id, current)

        return meta

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        if memory_id == "default":
            raise ValueError("Cannot delete the default memory")

        mem_dir = self._memory_dir(user_id, memory_id)
        if not mem_dir.exists():
            return False

        import shutil
        shutil.rmtree(mem_dir)

        current = self._ensure_current(user_id)
        if current.memory_id == memory_id:
            remaining = self.list_memories(user_id)
            if remaining:
                self.set_current_memory(user_id, remaining[0].memory_id)
            else:
                fallback = self.create_memory(user_id, "Mi memoria")

        return True

    # ------------------------------------------------------------------
    # Section read/write (scoped to current memory)
    # ------------------------------------------------------------------

    def _resolve_memory(self, user_id: str, memory_id: str | None = None) -> str:
        if memory_id:
            return memory_id
        return self.get_current_memory_id(user_id)

    def ensure_user_files(self, user_id: str, memory_id: str | None = None) -> None:
        mem_id = self._resolve_memory(user_id, memory_id)
        self._ensure_memory_files(user_id, mem_id)

    def load(self, user_id: str, memory_id: str | None = None, *, include_legacy: bool = True) -> UserMemory:
        mem_id = self._resolve_memory(user_id, memory_id)
        self._ensure_memory_files(user_id, mem_id)
        mem_dir = self._memory_dir(user_id, mem_id)
        sections: Dict[str, str] = {}
        used_sections: List[str] = []
        for section, filename in SECTION_FILES.items():
            if not include_legacy and section in LEGACY_CASE_SECTIONS:
                sections[section] = SECTION_TEMPLATES[section]
                continue
            content = (mem_dir / filename).read_text(encoding="utf-8").strip()
            sections[section] = content
            if content and content != SECTION_TEMPLATES[section].strip():
                used_sections.append(section)
        return UserMemory(
            user_id=user_id,
            memory_id=mem_id,
            sections=sections,
            used_sections=used_sections,
            memory_name=self._memory_name(user_id, mem_id),
        )

    def editable_snapshot(self, user_id: str, memory_id: str | None = None) -> Dict[str, str]:
        memory = self.load(user_id, memory_id)
        return {section: memory.sections.get(section, "") for section in EDITABLE_SECTIONS}

    def render_context(self, user_id: str, memory_id: str | None = None) -> str:
        memory = self.load(user_id, memory_id, include_legacy=False)
        blocks: List[str] = []
        for section in ("profile", "preferences", "farm_context", "open_questions"):
            content = memory.sections.get(section, "").strip()
            if not content or content == SECTION_TEMPLATES[section].strip():
                continue
            blocks.append(content)

        return "\n\n".join(block for block in blocks if block).strip()

    def replace_sections(self, user_id: str, sections: Dict[str, str], memory_id: str | None = None) -> UserMemory:
        mem_id = self._resolve_memory(user_id, memory_id)
        self._ensure_memory_files(user_id, mem_id)
        mem_dir = self._memory_dir(user_id, mem_id)
        for section, value in sections.items():
            if section not in EDITABLE_SECTIONS:
                continue
            normalized = (value or "").strip()
            content = normalized or SECTION_TEMPLATES[section].strip()
            if not content.startswith("# "):
                content = f"{SECTION_TEMPLATES[section].splitlines()[0]}\n\n{content}"
            (mem_dir / SECTION_FILES[section]).write_text(
                content.rstrip() + "\n", encoding="utf-8"
            )
        return self.load(user_id, mem_id)

    # ------------------------------------------------------------------
    # Case state, case history, observations (scoped to current memory)
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_case_state(case_state: CaseState) -> CaseState:
        validated = CaseState.model_validate(case_state)
        open_tasks: list[CaseTask] = []
        seen_titles: set[str] = set()
        for task in validated.open_tasks:
            title = " ".join(str(task.title).split()).strip()
            if not title:
                continue
            key = title.lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            open_tasks.append(
                CaseTask(
                    title=title[:140],
                    priority=task.priority,
                    status=task.status,
                    rationale=" ".join(str(task.rationale).split()).strip()[:400],
                    source=task.source,
                )
            )
        return CaseState(
            case_summary=" ".join(str(validated.case_summary).split()).strip()[:500],
            open_tasks=open_tasks[:6],
            blocked_by=[
                " ".join(str(item).split()).strip()[:280]
                for item in validated.blocked_by
                if " ".join(str(item).split()).strip()
            ][:6],
            recommended_next_input=[
                " ".join(str(item).split()).strip()[:280]
                for item in validated.recommended_next_input
                if " ".join(str(item).split()).strip()
            ][:6],
            evidence_ledger=validated.evidence_ledger,
        )

    @staticmethod
    def _sanitize_case_snapshot(snapshot: CaseSnapshot) -> CaseSnapshot:
        validated = CaseSnapshot.model_validate(snapshot)
        return CaseSnapshot(
            title=" ".join(str(validated.title).split()).strip()[:90],
            decision_mode=" ".join(str(validated.decision_mode).split()).strip()[:40] or "decision",
            summary=" ".join(str(validated.summary).split()).strip()[:400],
            next_actions=[
                " ".join(str(item).split()).strip()[:220]
                for item in validated.next_actions
                if " ".join(str(item).split()).strip()
            ][:3],
            blocked_by=[
                " ".join(str(item).split()).strip()[:220]
                for item in validated.blocked_by
                if " ".join(str(item).split()).strip()
            ][:3],
        )

    @staticmethod
    def _sanitize_observation(observation: FieldObservation) -> FieldObservation:
        validated = FieldObservation.model_validate(observation)
        return FieldObservation(
            date=" ".join(str(validated.date).split()).strip()[:40],
            parcel=" ".join(str(validated.parcel).split()).strip()[:120],
            campaign=(
                " ".join(str(validated.campaign).split()).strip()[:60]
                if validated.campaign
                else None
            ),
            note=" ".join(str(validated.note).split()).strip()[:500],
            severity=validated.severity,
        )

    def _load_case_snapshot_payload(self, structured_path: Path) -> List[CaseSnapshot]:
        try:
            payload = json.loads(structured_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("memory.case_history_json_invalid")
            return []
        if not isinstance(payload, list):
            logger.warning("memory.case_history_json_not_list")
            return []
        snapshots: list[CaseSnapshot] = []
        for item in payload[-12:]:
            try:
                snapshots.append(self._sanitize_case_snapshot(CaseSnapshot(**item)))
            except Exception:
                logger.warning("memory.case_history_item_invalid")
                continue
        return snapshots

    def _write_case_history_markdown(self, path: Path, snapshots: List[CaseSnapshot]) -> None:
        lines = ["# Historial de expedientes", ""]
        for snapshot in snapshots:
            lines.extend([
                f"## {snapshot.title}",
                f"- Modo: {snapshot.decision_mode}",
                f"- Resumen: {snapshot.summary}",
            ])
            if snapshot.next_actions:
                lines.append(f"- Proximas acciones: {' | '.join(snapshot.next_actions)}")
            if snapshot.blocked_by:
                lines.append(f"- Bloqueos: {' | '.join(snapshot.blocked_by)}")
            lines.append("")
        _atomic_write(path, "\n".join(lines).rstrip() + "\n")

    def _load_observation_payload(self, structured_path: Path) -> List[FieldObservation]:
        try:
            payload = json.loads(structured_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("memory.field_observations_json_invalid")
            return []
        if not isinstance(payload, list):
            logger.warning("memory.field_observations_json_not_list")
            return []
        observations: list[FieldObservation] = []
        for item in payload[-20:]:
            try:
                observations.append(self._sanitize_observation(FieldObservation(**item)))
            except Exception:
                logger.warning("memory.field_observation_item_invalid")
                continue
        return observations

    def _write_observations_markdown(self, path: Path, observations: List[FieldObservation]) -> None:
        lines = ["# Observaciones de campo", ""]
        for observation in observations:
            campaign = f" | campana: {observation.campaign}" if observation.campaign else ""
            lines.extend([
                f"- fecha: {observation.date} | parcela: {observation.parcel}{campaign} | severidad: {observation.severity}",
                f"  nota: {observation.note}",
                "",
            ])
        _atomic_write(path, "\n".join(lines).rstrip() + "\n")

    def load_decision_log(self, user_id: str, memory_id: str | None = None) -> List[dict]:
        mem_id = self._resolve_memory(user_id, memory_id)
        self._ensure_memory_files(user_id, mem_id)
        structured_path = self._structured_path(user_id, mem_id, "decision_log")
        if not structured_path.exists():
            return []
        try:
            payload = json.loads(structured_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                return []
            return payload[-20:]
        except Exception:
            logger.warning("memory.decision_log_json_invalid")
            return []

    def delete_user(self, user_id: str) -> None:
        import shutil
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            return
        shutil.rmtree(user_dir)

    def append_decision_log(
        self,
        user_id: str,
        *,
        query: str,
        decision_mode: str,
        executive_summary: str,
        next_actions: List[str],
        missing_information: List[str],
    ) -> None:
        mem_id = self.get_current_memory_id(user_id)
        self._ensure_memory_files(user_id, mem_id)
        path = self._memory_dir(user_id, mem_id) / SECTION_FILES["decision_log"]
        structured_path = self._structured_path(user_id, mem_id, "decision_log")
        entry_lines = [
            "",
            f"## Caso: {query.strip()[:80]}",
            f"- Modo: {decision_mode}",
            f"- Resumen: {executive_summary.strip() or 'Sin resumen'}",
        ]
        if next_actions:
            entry_lines.append(f"- Proximas acciones: {' | '.join(next_actions[:3])}")
        if missing_information:
            entry_lines.append(f"- Informacion pendiente: {' | '.join(missing_information[:3])}")
        new_entry = "\n".join(entry_lines).strip() + "\n"
        entry = {
            "query": query[:80],
            "decision_mode": decision_mode,
            "summary": executive_summary[:200],
            "next_actions": next_actions[:3],
            "missing_information": missing_information[:3],
            "timestamp": _now_iso(),
        }
        with _file_lock(structured_path):
            current = path.read_text(encoding="utf-8").rstrip()
            existing_lines = current.splitlines()[-120:]
            current = "\n".join(existing_lines).rstrip()
            _atomic_write(path, f"{current}\n{new_entry}")
            existing = []
            if structured_path.exists():
                try:
                    existing = json.loads(structured_path.read_text(encoding="utf-8"))
                    if not isinstance(existing, list):
                        existing = []
                except Exception:
                    existing = []
            existing.append(entry)
            _atomic_write(structured_path, json.dumps(existing[-50:], ensure_ascii=False, indent=2))

    def save_case_state(self, user_id: str, case_state: CaseState, memory_id: str | None = None) -> None:
        mem_id = self._resolve_memory(user_id, memory_id)
        self._ensure_memory_files(user_id, mem_id)
        case_state = self._sanitize_case_state(case_state)
        path = self._memory_dir(user_id, mem_id) / SECTION_FILES["current_case"]
        structured_path = self._structured_path(user_id, mem_id, "current_case")
        lines = [
            "# Estado de caso actual",
            "",
            "## Resumen",
            "",
            case_state.case_summary.strip() or "Sin caso activo.",
            "",
            "## Tareas abiertas",
            "",
        ]
        if case_state.open_tasks:
            for task in case_state.open_tasks:
                lines.append(f"- [{task.priority}] {task.title} :: {task.status}")
                if task.rationale:
                    lines.append(f"  - Motivo: {task.rationale}")
        else:
            lines.append("- Sin tareas abiertas.")
        lines.extend(["", "## Bloqueos", ""])
        if case_state.blocked_by:
            lines.extend([f"- {item}" for item in case_state.blocked_by])
        else:
            lines.append("- Sin bloqueos relevantes.")
        lines.extend(["", "## Proximo input recomendado", ""])
        if case_state.recommended_next_input:
            lines.extend([f"- {item}" for item in case_state.recommended_next_input])
        else:
            lines.append("- Sin input adicional pendiente.")
        with _file_lock(structured_path):
            _atomic_write(structured_path, case_state.model_dump_json(indent=2))
            _atomic_write(path, "\n".join(lines).rstrip() + "\n")

    def load_case_state(self, user_id: str, memory_id: str | None = None) -> CaseState:
        mem_id = self._resolve_memory(user_id, memory_id)
        self._ensure_memory_files(user_id, mem_id)
        path = self._memory_dir(user_id, mem_id) / SECTION_FILES["current_case"]
        structured_path = self._structured_path(user_id, mem_id, "current_case")
        if structured_path.exists():
            try:
                return self._sanitize_case_state(
                    CaseState.model_validate_json(structured_path.read_text(encoding="utf-8"))
                )
            except Exception:
                logger.warning("memory.current_case_json_invalid")
                pass
        content = path.read_text(encoding="utf-8").strip()
        if not content or content == SECTION_TEMPLATES["current_case"].strip():
            return CaseState()

        sections = {"summary": [], "tasks": [], "blocked": [], "next": []}
        current = None
        for raw_line in content.splitlines():
            line = raw_line.rstrip()
            if line.startswith("## Resumen"):
                current = "summary"
                continue
            if line.startswith("## Tareas abiertas"):
                current = "tasks"
                continue
            if line.startswith("## Bloqueos"):
                current = "blocked"
                continue
            if line.startswith("## Proximo input recomendado"):
                current = "next"
                continue
            if current:
                sections[current].append(line)

        summary = " ".join(
            line.strip()
            for line in sections["summary"]
            if line.strip() and not line.startswith("#")
        ).strip()
        tasks: list[CaseTask] = []
        pending_task: CaseTask | None = None
        for line in sections["tasks"]:
            text = line.strip()
            if not text:
                continue
            if text.startswith("- ["):
                try:
                    header, rest = text[3:].split("] ", 1)
                    title, status = rest.rsplit(" :: ", 1)
                    pending_task = CaseTask(
                        title=title.strip(), priority=header.strip("[] "), status=status.strip()
                    )
                    tasks.append(pending_task)
                except Exception:
                    pending_task = CaseTask(title=text.lstrip("- ").strip())
                    tasks.append(pending_task)
            elif text.startswith("- Motivo:") and pending_task:
                pending_task.rationale = text.replace("- Motivo:", "", 1).strip()
        blocked = [
            line.lstrip("- ").strip()
            for line in sections["blocked"]
            if line.strip().startswith("- ") and "Sin bloqueos" not in line
        ]
        next_input = [
            line.lstrip("- ").strip()
            for line in sections["next"]
            if line.strip().startswith("- ") and "Sin input" not in line
        ]
        return self._sanitize_case_state(
            CaseState(
                case_summary=summary,
                open_tasks=tasks,
                blocked_by=blocked,
                recommended_next_input=next_input,
            )
        )

    def append_case_history(
        self,
        user_id: str,
        *,
        title: str,
        decision_mode: str,
        summary: str,
        next_actions: List[str],
        blocked_by: List[str],
    ) -> None:
        mem_id = self.get_current_memory_id(user_id)
        self._ensure_memory_files(user_id, mem_id)
        path = self._memory_dir(user_id, mem_id) / SECTION_FILES["case_history"]
        structured_path = self._structured_path(user_id, mem_id, "case_history")
        entry_lines = [
            "",
            f"## {title.strip()[:90]}",
            f"- Modo: {decision_mode}",
            f"- Resumen: {summary.strip() or 'Sin resumen'}",
        ]
        if next_actions:
            entry_lines.append(f"- Proximas acciones: {' | '.join(next_actions[:3])}")
        if blocked_by:
            entry_lines.append(f"- Bloqueos: {' | '.join(blocked_by[:3])}")
        new_entry = "\n".join(entry_lines).strip() + "\n"
        sanitized = self._sanitize_case_snapshot(
            CaseSnapshot(
                title=title.strip()[:90],
                decision_mode=decision_mode,
                summary=summary.strip() or "Sin resumen",
                next_actions=next_actions[:3],
                blocked_by=blocked_by[:3],
            )
        ).model_dump()
        with _file_lock(structured_path):
            current = path.read_text(encoding="utf-8").rstrip()
            existing_lines = current.splitlines()[-160:]
            current = "\n".join(existing_lines).rstrip()
            _atomic_write(path, f"{current}\n{new_entry}")
            existing_history = []
            if structured_path.exists():
                existing_history = [item.model_dump() for item in self._load_case_snapshot_payload(structured_path)]
            existing_history.append(sanitized)
            _atomic_write(structured_path, json.dumps(existing_history[-12:], ensure_ascii=False, indent=2))

    def load_case_history(self, user_id: str, memory_id: str | None = None) -> List[CaseSnapshot]:
        mem_id = self._resolve_memory(user_id, memory_id)
        self._ensure_memory_files(user_id, mem_id)
        path = self._memory_dir(user_id, mem_id) / SECTION_FILES["case_history"]
        structured_path = self._structured_path(user_id, mem_id, "case_history")
        if structured_path.exists():
            payload = self._load_case_snapshot_payload(structured_path)
            return list(reversed(payload))
        content = path.read_text(encoding="utf-8").strip()
        if not content or content == SECTION_TEMPLATES["case_history"].strip():
            return []
        snapshots: list[CaseSnapshot] = []
        current: dict[str, object] | None = None
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if line.startswith("## "):
                if current:
                    snapshots.append(self._sanitize_case_snapshot(CaseSnapshot(**current)))
                current = {
                    "title": line.replace("## ", "", 1).strip(),
                    "decision_mode": "decision",
                    "summary": "",
                    "next_actions": [],
                    "blocked_by": [],
                }
                continue
            if not current or not line.startswith("- "):
                continue
            if line.startswith("- Modo: "):
                current["decision_mode"] = line.replace("- Modo: ", "", 1).strip()
            elif line.startswith("- Resumen: "):
                current["summary"] = line.replace("- Resumen: ", "", 1).strip()
            elif line.startswith("- Proximas acciones: "):
                current["next_actions"] = [
                    item.strip()
                    for item in line.replace("- Proximas acciones: ", "", 1).split(" | ")
                    if item.strip()
                ]
            elif line.startswith("- Bloqueos: "):
                current["blocked_by"] = [
                    item.strip()
                    for item in line.replace("- Bloqueos: ", "", 1).split(" | ")
                    if item.strip()
                ]
        if current:
            snapshots.append(self._sanitize_case_snapshot(CaseSnapshot(**current)))
        return list(reversed(snapshots[-12:]))

    def append_observation(self, user_id: str, observation: FieldObservation) -> None:
        mem_id = self.get_current_memory_id(user_id)
        self._ensure_memory_files(user_id, mem_id)
        path = self._memory_dir(user_id, mem_id) / SECTION_FILES["field_observations"]
        structured_path = self._structured_path(user_id, mem_id, "field_observations")
        campaign = f" | campana: {observation.campaign}" if observation.campaign else ""
        entry_lines = [
            "",
            f"- fecha: {observation.date} | parcela: {observation.parcel}{campaign} | severidad: {observation.severity}",
            f"  nota: {observation.note}",
        ]
        new_entry = "\n".join(entry_lines).strip() + "\n"
        sanitized = self._sanitize_observation(observation).model_dump()
        with _file_lock(structured_path):
            current = path.read_text(encoding="utf-8").rstrip()
            existing_lines = current.splitlines()[-200:]
            current = "\n".join(existing_lines).rstrip()
            _atomic_write(path, f"{current}\n{new_entry}")
            existing_observations = []
            if structured_path.exists():
                existing_observations = [
                    item.model_dump() for item in self._load_observation_payload(structured_path)
                ]
            existing_observations.append(sanitized)
            _atomic_write(structured_path, json.dumps(existing_observations[-20:], ensure_ascii=False, indent=2))

    def load_observations(self, user_id: str, memory_id: str | None = None) -> List[FieldObservation]:
        mem_id = self._resolve_memory(user_id, memory_id)
        self._ensure_memory_files(user_id, mem_id)
        path = self._memory_dir(user_id, mem_id) / SECTION_FILES["field_observations"]
        structured_path = self._structured_path(user_id, mem_id, "field_observations")
        if structured_path.exists():
            payload = self._load_observation_payload(structured_path)
            return list(reversed(payload))
        content = path.read_text(encoding="utf-8").strip()
        if not content or content == SECTION_TEMPLATES["field_observations"].strip():
            return []
        observations: list[FieldObservation] = []
        current: dict[str, str] | None = None
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if line.startswith("- fecha: "):
                if (
                    current
                    and current.get("date")
                    and current.get("parcel")
                    and current.get("note")
                ):
                    observations.append(self._sanitize_observation(FieldObservation(**current)))
                current = {
                    "date": "",
                    "parcel": "",
                    "campaign": None,
                    "note": "",
                    "severity": "media",
                }
                parts = [
                    part.strip() for part in line.replace("- ", "", 1).split(" | ") if part.strip()
                ]
                for part in parts:
                    if part.startswith("fecha: "):
                        current["date"] = part.replace("fecha: ", "", 1).strip()
                    elif part.startswith("parcela: "):
                        current["parcel"] = part.replace("parcela: ", "", 1).strip()
                    elif part.startswith("campana: "):
                        current["campaign"] = part.replace("campana: ", "", 1).strip()
                    elif part.startswith("severidad: "):
                        current["severity"] = part.replace("severidad: ", "", 1).strip()
            elif line.startswith("nota: ") and current is not None:
                current["note"] = line.replace("nota: ", "", 1).strip()
        if current and current.get("date") and current.get("parcel") and current.get("note"):
            observations.append(self._sanitize_observation(FieldObservation(**current)))
        return list(reversed(observations[-20:]))

    def delete_observation(self, user_id: str, index: int, memory_id: str | None = None) -> bool:
        mem_id = self._resolve_memory(user_id, memory_id)
        self._ensure_memory_files(user_id, mem_id)
        structured_path = self._structured_path(user_id, mem_id, "field_observations")
        if not structured_path.exists():
            return False
        with _file_lock(structured_path):
            try:
                items = json.loads(structured_path.read_text(encoding="utf-8"))
                if not isinstance(items, list) or index < 0 or index >= len(items):
                    return False
                # Public lists are returned newest-first, so API indices refer to
                # that visible order rather than the append-order JSON payload.
                items.pop(len(items) - 1 - index)
                _atomic_write(structured_path, json.dumps(items, ensure_ascii=False, indent=2))
                parsed = [self._sanitize_observation(FieldObservation(**item)) for item in items]
                self._write_observations_markdown(
                    self._memory_dir(user_id, mem_id) / SECTION_FILES["field_observations"],
                    parsed,
                )
            except Exception:
                return False
        return True

    def delete_case_history(self, user_id: str, index: int, memory_id: str | None = None) -> bool:
        mem_id = self._resolve_memory(user_id, memory_id)
        self._ensure_memory_files(user_id, mem_id)
        structured_path = self._structured_path(user_id, mem_id, "case_history")
        if not structured_path.exists():
            return False
        with _file_lock(structured_path):
            try:
                items = json.loads(structured_path.read_text(encoding="utf-8"))
                if not isinstance(items, list) or index < 0 or index >= len(items):
                    return False
                # Public lists are returned newest-first, so API indices refer to
                # that visible order rather than the append-order JSON payload.
                items.pop(len(items) - 1 - index)
                _atomic_write(structured_path, json.dumps(items, ensure_ascii=False, indent=2))
                parsed = [self._sanitize_case_snapshot(CaseSnapshot(**item)) for item in items]
                self._write_case_history_markdown(
                    self._memory_dir(user_id, mem_id) / SECTION_FILES["case_history"],
                    parsed,
                )
            except Exception:
                return False
        return True

    def save_remote_sensing_artifact(
        self,
        user_id: str,
        artifact: RemoteSensingMemoryArtifact,
        memory_id: str | None = None,
    ) -> None:
        mem_id = self._resolve_memory(user_id, memory_id)
        self._ensure_memory_files(user_id, mem_id)
        path = self._remote_sensing_artifacts_path(user_id, mem_id)
        with _file_lock(path):
            items: list[dict] = []
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, list):
                        items = [item for item in payload if isinstance(item, dict)]
                except Exception:
                    logger.warning("memory.remote_sensing_artifacts_json_invalid")
                    try:
                        backup = path.with_suffix(".json.corrupt")
                        path.rename(backup)
                        logger.info("memory.remote_sensing_artifacts_backup_created", backup=str(backup))
                    except Exception:
                        pass
            items.append(RemoteSensingMemoryArtifact.model_validate(artifact).model_dump())
            _atomic_write(path, json.dumps(items[-8:], ensure_ascii=False, indent=2))

    def load_remote_sensing_artifacts(
        self,
        user_id: str,
        memory_id: str | None = None,
    ) -> List[RemoteSensingMemoryArtifact]:
        mem_id = self._resolve_memory(user_id, memory_id)
        self._ensure_memory_files(user_id, mem_id)
        path = self._remote_sensing_artifacts_path(user_id, mem_id)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("memory.remote_sensing_artifacts_json_invalid")
            return []
        if not isinstance(payload, list):
            logger.warning("memory.remote_sensing_artifacts_json_not_list")
            return []
        artifacts: list[RemoteSensingMemoryArtifact] = []
        for item in payload[-8:]:
            try:
                artifacts.append(RemoteSensingMemoryArtifact.model_validate(item))
            except Exception:
                logger.warning("memory.remote_sensing_artifact_item_invalid")
        return list(reversed(artifacts))


memory_store = UserMemoryStore()
