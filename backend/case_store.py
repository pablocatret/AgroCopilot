"""Transactional casework store.

The legacy Markdown memory files remain readable during migration, but they are
not the source of truth for explicit cases.  This module deliberately uses
sqlite3 so a local installation has no infrastructure dependency while the
repository interface remains portable to a server database later on.
"""
from __future__ import annotations

import atexit
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

CASE_STATUSES = {"active", "on_hold", "closed", "archived", "deleted"}
ASSERTION_STATUSES = {"proposed", "confirmed", "superseded", "retracted", "expired"}
TASK_STATUSES = {"proposed", "open", "blocked", "done", "cancelled"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _decode(value: Any, fallback: Any) -> Any:
    try:
        return json.loads(value) if value else fallback
    except (TypeError, json.JSONDecodeError):
        return fallback


class CaseStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or "./data/cases.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        schema = """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS workspaces (
            workspace_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS workspace_context (
            workspace_id TEXT PRIMARY KEY,
            name TEXT NOT NULL DEFAULT '',
            zone TEXT NOT NULL DEFAULT '',
            crops TEXT NOT NULL DEFAULT '',
            infrastructure TEXT NOT NULL DEFAULT '',
            constraints TEXT NOT NULL DEFAULT '',
            preferences TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
        );
        CREATE TABLE IF NOT EXISTS continuity_migrations (
            workspace_id TEXT PRIMARY KEY,
            version INTEGER NOT NULL,
            migrated_at TEXT NOT NULL,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
        );
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            title TEXT NOT NULL,
            objective TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            summary TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_activity_at TEXT NOT NULL,
            deleted_at TEXT,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
        );
        CREATE INDEX IF NOT EXISTS idx_cases_workspace_updated ON cases(workspace_id, updated_at DESC);
        CREATE TABLE IF NOT EXISTS case_conversations (
            case_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL UNIQUE,
            linked_at TEXT NOT NULL,
            PRIMARY KEY (case_id, conversation_id),
            FOREIGN KEY (case_id) REFERENCES cases(case_id)
        );
        CREATE TABLE IF NOT EXISTS case_events (
            event_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            sequence_no INTEGER NOT NULL DEFAULT 0,
            event_type TEXT NOT NULL,
            actor_type TEXT NOT NULL,
            source_type TEXT,
            source_id TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES cases(case_id)
        );
        CREATE INDEX IF NOT EXISTS idx_case_events_case_created ON case_events(case_id, created_at DESC);
        CREATE TABLE IF NOT EXISTS assertions (
            assertion_id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            case_id TEXT,
            scope TEXT NOT NULL,
            assertion_type TEXT NOT NULL DEFAULT 'fact',
            key TEXT NOT NULL,
            value_text TEXT NOT NULL,
            display_text TEXT NOT NULL DEFAULT '',
            provenance TEXT NOT NULL,
            confidence REAL,
            status TEXT NOT NULL DEFAULT 'proposed',
            valid_from TEXT,
            valid_until TEXT,
            source_event_id TEXT,
            source_message_id TEXT,
            source_document_id TEXT,
            supersedes_assertion_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES cases(case_id),
            FOREIGN KEY (supersedes_assertion_id) REFERENCES assertions(assertion_id)
        );
        CREATE INDEX IF NOT EXISTS idx_assertions_scope_status ON assertions(workspace_id, case_id, scope, status);
        CREATE TABLE IF NOT EXISTS evidence_links (
            evidence_link_id TEXT PRIMARY KEY,
            assertion_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            excerpt TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (assertion_id) REFERENCES assertions(assertion_id)
        );
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            title TEXT NOT NULL,
            rationale TEXT NOT NULL DEFAULT '',
            priority TEXT NOT NULL DEFAULT 'medium',
            status TEXT NOT NULL DEFAULT 'proposed',
            source_assertion_id TEXT,
            created_by TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES cases(case_id)
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_case_status ON tasks(case_id, status);
        CREATE TABLE IF NOT EXISTS decisions (
            decision_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            title TEXT NOT NULL,
            rationale TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'proposed',
            created_by TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES cases(case_id)
        );
        CREATE TABLE IF NOT EXISTS case_documents (
            case_id TEXT NOT NULL,
            attachment_id TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'evidence',
            linked_at TEXT NOT NULL,
            PRIMARY KEY (case_id, attachment_id),
            FOREIGN KEY (case_id) REFERENCES cases(case_id)
        );
        CREATE TABLE IF NOT EXISTS case_observations (
            observation_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            date TEXT NOT NULL,
            parcel TEXT NOT NULL,
            campaign TEXT,
            note TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'media',
            created_at TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES cases(case_id)
        );
        CREATE INDEX IF NOT EXISTS idx_case_observations_case_date
            ON case_observations(case_id, date DESC, created_at DESC);
        CREATE TABLE IF NOT EXISTS case_state_projection (
            case_id TEXT PRIMARY KEY,
            projection_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES cases(case_id)
        );
        CREATE TABLE IF NOT EXISTS context_runs (
            context_run_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            conversation_id TEXT,
            query TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES cases(case_id)
        );
        CREATE TABLE IF NOT EXISTS context_run_items (
            context_run_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            rank INTEGER NOT NULL,
            PRIMARY KEY (context_run_id, source_type, source_id),
            FOREIGN KEY (context_run_id) REFERENCES context_runs(context_run_id)
        );
        """
        with self._lock:
            self.conn.executescript(schema)
            columns = {row[1] for row in self.conn.execute("PRAGMA table_info(case_events)").fetchall()}
            if "sequence_no" not in columns:
                self.conn.execute("ALTER TABLE case_events ADD COLUMN sequence_no INTEGER NOT NULL DEFAULT 0")
            existing = self.conn.execute(
                "SELECT event_id, case_id FROM case_events WHERE sequence_no=0 ORDER BY rowid"
            ).fetchall()
            counters = {
                row["case_id"]: int(row["max_sequence"])
                for row in self.conn.execute(
                    "SELECT case_id, MAX(sequence_no) AS max_sequence FROM case_events GROUP BY case_id"
                ).fetchall()
            }
            for row in existing:
                counters[row["case_id"]] = counters.get(row["case_id"], 0) + 1
                self.conn.execute(
                    "UPDATE case_events SET sequence_no=? WHERE event_id=?",
                    (counters[row["case_id"]], row["event_id"]),
                )
            self.conn.commit()

    def _ensure_workspace(self, workspace_id: str) -> None:
        workspace_id = (workspace_id or "local").strip() or "local"
        self.conn.execute(
            "INSERT OR IGNORE INTO workspaces(workspace_id, name, created_at) VALUES (?, ?, ?)",
            (workspace_id, workspace_id, _now()),
        )

    def get_workspace_context(self, workspace_id: str) -> dict[str, Any]:
        workspace_id = (workspace_id or "local").strip() or "local"
        with self._lock:
            self._ensure_workspace(workspace_id)
            row = self.conn.execute(
                "SELECT * FROM workspace_context WHERE workspace_id=?",
                (workspace_id,),
            ).fetchone()
            if row is None:
                context = {
                    "workspace_id": workspace_id,
                    "name": "",
                    "zone": "",
                    "crops": "",
                    "infrastructure": "",
                    "constraints": "",
                    "preferences": "",
                    "updated_at": _now(),
                }
                self.conn.execute(
                    """INSERT INTO workspace_context
                    (workspace_id, name, zone, crops, infrastructure, constraints, preferences, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    tuple(context.values()),
                )
                self.conn.commit()
                return context
        return dict(row)

    def save_workspace_context(self, workspace_id: str, values: dict[str, Any]) -> dict[str, Any]:
        workspace_id = (workspace_id or "local").strip() or "local"
        keys = ("name", "zone", "crops", "infrastructure", "constraints", "preferences")
        normalized = {
            key: " ".join(str(values.get(key) or "").split())[:1000]
            for key in keys
        }
        now = _now()
        with self._lock:
            self._ensure_workspace(workspace_id)
            self.conn.execute(
                """INSERT INTO workspace_context
                (workspace_id, name, zone, crops, infrastructure, constraints, preferences, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id) DO UPDATE SET
                name=excluded.name, zone=excluded.zone, crops=excluded.crops,
                infrastructure=excluded.infrastructure, constraints=excluded.constraints,
                preferences=excluded.preferences, updated_at=excluded.updated_at""",
                (workspace_id, *[normalized[key] for key in keys], now),
            )
            self.conn.commit()
        return self.get_workspace_context(workspace_id)

    def migration_version(self, workspace_id: str) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT version FROM continuity_migrations WHERE workspace_id=?",
                (workspace_id,),
            ).fetchone()
        return int(row["version"]) if row else 0

    def mark_migration(self, workspace_id: str, version: int) -> None:
        with self._lock:
            self._ensure_workspace(workspace_id)
            self.conn.execute(
                """INSERT INTO continuity_migrations(workspace_id, version, migrated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(workspace_id) DO UPDATE SET version=excluded.version, migrated_at=excluded.migrated_at""",
                (workspace_id, version, _now()),
            )
            self.conn.commit()

    @staticmethod
    def _case_row(row: sqlite3.Row) -> dict[str, Any]:
        return dict(row)

    @staticmethod
    def _assertion_row(row: sqlite3.Row) -> dict[str, Any]:
        return dict(row)

    @staticmethod
    def _event_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["payload"] = _decode(item.pop("payload_json", "{}"), {})
        return item

    def create_case(self, *, workspace_id: str, title: str, objective: str = "") -> dict[str, Any]:
        title = " ".join((title or "Nuevo caso").split())[:160] or "Nuevo caso"
        now = _now()
        case_id = _id("case")
        with self._lock:
            self._ensure_workspace(workspace_id)
            self.conn.execute(
                """INSERT INTO cases(case_id, workspace_id, title, objective, status, created_at, updated_at, last_activity_at)
                VALUES (?, ?, ?, ?, 'active', ?, ?, ?)""",
                (case_id, workspace_id, title, (objective or "").strip()[:500], now, now, now),
            )
            self._append_event_locked(case_id, "case_created", "user", payload={"title": title, "objective": objective})
            self.conn.commit()
        return self.get_case(case_id, workspace_id=workspace_id)["case"]

    def list_cases(self, *, workspace_id: str, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        query = "SELECT * FROM cases WHERE workspace_id = ?"
        params: list[Any] = [workspace_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        else:
            query += " AND status != 'deleted'"
        query += " ORDER BY last_activity_at DESC LIMIT ?"
        params.append(max(1, min(limit, 200)))
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        return [self._case_row(row) for row in rows]

    def get_case(self, case_id: str, *, workspace_id: str | None = None) -> dict[str, Any]:
        query = "SELECT * FROM cases WHERE case_id = ?"
        params: list[Any] = [case_id]
        if workspace_id:
            query += " AND workspace_id = ?"
            params.append(workspace_id)
        with self._lock:
            row = self.conn.execute(query, params).fetchone()
        if row is None:
            raise KeyError("Case not found")
        case = self._case_row(row)
        projection = self.project_case(case_id)
        return {
            "case": case,
            "projection": projection,
            "events": self.list_events(case_id, limit=30),
            "assertions": self.list_assertions(case_id=case_id, workspace_id=case["workspace_id"]),
            "tasks": self.list_tasks(case_id),
            "decisions": self.list_decisions(case_id),
            "observations": self.list_observations(case_id),
        }

    def _require_mutable_case(self, case_id: str, *, workspace_id: str | None = None) -> dict[str, Any]:
        detail = self.get_case(case_id, workspace_id=workspace_id)
        if detail["case"]["status"] == "deleted":
            raise ValueError("Deleted cases cannot be modified")
        return detail

    def update_case(self, case_id: str, *, workspace_id: str, title: str | None = None, objective: str | None = None) -> dict[str, Any]:
        case = self._require_mutable_case(case_id, workspace_id=workspace_id)["case"]
        next_title = " ".join((title if title is not None else case["title"]).split())[:160] or case["title"]
        next_objective = (objective if objective is not None else case["objective"] or "").strip()[:500]
        now = _now()
        with self._lock:
            self.conn.execute("UPDATE cases SET title=?, objective=?, updated_at=?, last_activity_at=? WHERE case_id=?", (next_title, next_objective, now, now, case_id))
            self._append_event_locked(case_id, "case_updated", "user", payload={"title": next_title, "objective": next_objective})
            self.conn.commit()
        return self.get_case(case_id, workspace_id=workspace_id)["case"]

    def set_case_status(self, case_id: str, *, workspace_id: str, status: str, actor_type: str = "user") -> dict[str, Any]:
        if status not in CASE_STATUSES:
            raise ValueError("Invalid case status")
        case = self.get_case(case_id, workspace_id=workspace_id)["case"]
        if case["status"] == "deleted" and status != "deleted":
            raise ValueError("Deleted cases cannot be reopened")
        now = _now()
        with self._lock:
            self.conn.execute("UPDATE cases SET status=?, updated_at=?, last_activity_at=?, deleted_at=? WHERE case_id=?", (status, now, now, now if status == "deleted" else None, case_id))
            self._append_event_locked(case_id, f"case_{status}", actor_type, payload={"status": status})
            self.conn.commit()
        return self.get_case(case_id, workspace_id=workspace_id)["case"]

    def _append_event_locked(self, case_id: str, event_type: str, actor_type: str, *, source_type: str | None = None, source_id: str | None = None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        row = self.conn.execute("SELECT COALESCE(MAX(sequence_no), 0) + 1 AS next_sequence FROM case_events WHERE case_id=?", (case_id,)).fetchone()
        sequence_no = int(row["next_sequence"])
        event = {
            "event_id": _id("evt"), "case_id": case_id, "sequence_no": sequence_no, "event_type": event_type,
            "actor_type": actor_type, "source_type": source_type, "source_id": source_id,
            "payload": payload or {}, "created_at": _now(),
        }
        self.conn.execute(
            "INSERT INTO case_events(event_id, case_id, sequence_no, event_type, actor_type, source_type, source_id, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (event["event_id"], case_id, sequence_no, event_type, actor_type, source_type, source_id, _json(event["payload"]), event["created_at"]),
        )
        self.conn.execute("UPDATE cases SET updated_at=?, last_activity_at=? WHERE case_id=?", (event["created_at"], event["created_at"], case_id))
        return event

    def append_event(self, case_id: str, *, event_type: str, actor_type: str, source_type: str | None = None, source_id: str | None = None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            event = self._append_event_locked(case_id, event_type, actor_type, source_type=source_type, source_id=source_id, payload=payload)
            self.conn.commit()
        return event

    def list_events(self, case_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM case_events WHERE case_id=? ORDER BY sequence_no DESC, created_at DESC LIMIT ?", (case_id, max(1, min(limit, 300)))).fetchall()
        return [self._event_row(row) for row in rows]

    def link_conversation(self, *, case_id: str, conversation_id: str, workspace_id: str | None = None) -> None:
        self.rebind_conversation(case_id=case_id, conversation_id=conversation_id, workspace_id=workspace_id)

    def rebind_conversation(self, *, case_id: str, conversation_id: str, workspace_id: str | None = None) -> str | None:
        with self._lock:
            self.get_case(case_id, workspace_id=workspace_id)
            current = self.conn.execute(
                "SELECT case_id FROM case_conversations WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
            previous_case_id = current["case_id"] if current else None
            if previous_case_id == case_id:
                return previous_case_id
            if previous_case_id:
                self.conn.execute(
                    "DELETE FROM case_conversations WHERE conversation_id=?",
                    (conversation_id,),
                )
                self._append_event_locked(
                    previous_case_id,
                    "conversation_unlinked",
                    "system",
                    source_type="conversation",
                    source_id=conversation_id,
                )
            self.conn.execute(
                "INSERT INTO case_conversations(case_id, conversation_id, linked_at) VALUES (?, ?, ?)",
                (case_id, conversation_id, _now()),
            )
            self._append_event_locked(case_id, "conversation_linked", "system", source_type="conversation", source_id=conversation_id)
            self.conn.commit()
            return previous_case_id

    def link_document(self, *, case_id: str, attachment_id: str, workspace_id: str | None = None, role: str = "evidence") -> None:
        self.get_case(case_id, workspace_id=workspace_id)
        with self._lock:
            cursor = self.conn.execute("INSERT OR IGNORE INTO case_documents(case_id, attachment_id, role, linked_at) VALUES (?, ?, ?, ?)", (case_id, attachment_id, role, _now()))
            if cursor.rowcount == 0:
                return
            self._append_event_locked(case_id, "document_linked", "user", source_type="attachment", source_id=attachment_id, payload={"role": role})
            self.conn.commit()

    def create_observation(
        self,
        *,
        case_id: str,
        workspace_id: str | None = None,
        date: str,
        parcel: str,
        note: str,
        severity: str = "media",
        campaign: str | None = None,
        actor_type: str = "user",
    ) -> dict[str, Any]:
        if severity not in {"baja", "media", "alta"}:
            raise ValueError("Invalid observation severity")
        self._require_mutable_case(case_id, workspace_id=workspace_id)
        observation = {
            "observation_id": _id("obs"),
            "case_id": case_id,
            "date": " ".join(str(date or "").split())[:40],
            "parcel": " ".join(str(parcel or "").split())[:160],
            "campaign": " ".join(str(campaign or "").split())[:60] or None,
            "note": " ".join(str(note or "").split())[:2000],
            "severity": severity,
            "created_at": _now(),
        }
        if not observation["date"] or not observation["parcel"] or not observation["note"]:
            raise ValueError("Observation date, parcel and note are required")
        with self._lock:
            self.conn.execute(
                """INSERT INTO case_observations
                (observation_id, case_id, date, parcel, campaign, note, severity, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(observation.values()),
            )
            self._append_event_locked(
                case_id,
                "observation_recorded",
                actor_type,
                source_type="observation",
                source_id=observation["observation_id"],
                payload={"parcel": observation["parcel"], "severity": severity},
            )
            self.conn.commit()
        self.refresh_case_projection(case_id)
        return observation

    def list_observations(self, case_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM case_observations WHERE case_id=? ORDER BY date DESC, created_at DESC LIMIT ?",
                (case_id, max(1, min(limit, 300))),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_assertion(self, *, workspace_id: str, case_id: str | None, key: str, value: str, scope: str, provenance: str, status: str = "proposed", actor_type: str = "assistant", assertion_type: str = "fact", confidence: float | None = None, display_text: str = "", valid_from: str | None = None, valid_until: str | None = None, source_event_id: str | None = None, source_message_id: str | None = None, source_document_id: str | None = None) -> dict[str, Any]:
        if scope not in {"case", "global"} or status not in ASSERTION_STATUSES:
            raise ValueError("Invalid assertion scope or status")
        if scope == "case" and not case_id:
            raise ValueError("Case assertions require a case_id")
        if case_id:
            self._require_mutable_case(case_id, workspace_id=workspace_id)
        assertion_id, now = _id("ast"), _now()
        record = {
            "assertion_id": assertion_id, "workspace_id": workspace_id, "case_id": case_id, "scope": scope,
            "assertion_type": assertion_type, "key": " ".join(key.split())[:120], "value": " ".join(value.split())[:1000],
            "display_text": " ".join(display_text.split())[:1000], "provenance": provenance,
            "confidence": confidence, "status": status, "valid_from": valid_from, "valid_until": valid_until,
            "source_event_id": source_event_id, "source_message_id": source_message_id, "source_document_id": source_document_id,
            "supersedes_assertion_id": None, "created_at": now, "updated_at": now,
        }
        with self._lock:
            self._ensure_workspace(workspace_id)
            self.conn.execute(
                """INSERT INTO assertions(assertion_id, workspace_id, case_id, scope, assertion_type, key, value_text, display_text, provenance, confidence, status, valid_from, valid_until, source_event_id, source_message_id, source_document_id, supersedes_assertion_id, created_at, updated_at)
                VALUES (:assertion_id, :workspace_id, :case_id, :scope, :assertion_type, :key, :value, :display_text, :provenance, :confidence, :status, :valid_from, :valid_until, :source_event_id, :source_message_id, :source_document_id, :supersedes_assertion_id, :created_at, :updated_at)""", record,
            )
            if case_id:
                self._append_event_locked(case_id, "assertion_created", actor_type, source_type="assertion", source_id=assertion_id, payload={"key": record["key"], "status": status, "provenance": provenance})
            self.conn.commit()
        if case_id:
            self.refresh_case_projection(case_id)
        return record

    def list_assertions(self, *, workspace_id: str, case_id: str | None = None, scope: str | None = None, statuses: Iterable[str] | None = None) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM assertions WHERE workspace_id = ?", [workspace_id]
        if case_id is not None:
            query += " AND case_id = ?"
            params.append(case_id)
        if scope:
            query += " AND scope = ?"
            params.append(scope)
        if statuses:
            values = list(statuses)
            query += f" AND status IN ({','.join('?' for _ in values)})"
            params.extend(values)
        query += " ORDER BY updated_at DESC"
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        return [self._assertion_row(row) for row in rows]

    def correct_assertion(self, assertion_id: str, *, workspace_id: str | None = None, value: str, actor_type: str = "user", display_text: str | None = None) -> dict[str, Any]:
        with self._lock:
            if workspace_id is None:
                old = self.conn.execute("SELECT * FROM assertions WHERE assertion_id=?", (assertion_id,)).fetchone()
            else:
                old = self.conn.execute("SELECT * FROM assertions WHERE assertion_id=? AND workspace_id=?", (assertion_id, workspace_id)).fetchone()
            if old is None:
                raise KeyError("Assertion not found")
            old_item = self._assertion_row(old)
            now = _now()
            self.conn.execute("UPDATE assertions SET status='superseded', updated_at=? WHERE assertion_id=?", (now, assertion_id))
            new_item = self.create_assertion(
                workspace_id=old_item["workspace_id"], case_id=old_item["case_id"], key=old_item["key"], value=value,
                scope=old_item["scope"], provenance="user_correction", status="confirmed", actor_type=actor_type,
                assertion_type=old_item["assertion_type"], confidence=old_item["confidence"],
                display_text=display_text if display_text is not None else old_item["display_text"],
                valid_from=old_item["valid_from"], valid_until=old_item["valid_until"],
                source_event_id=old_item["source_event_id"], source_message_id=old_item["source_message_id"], source_document_id=old_item["source_document_id"],
            )
            self.conn.execute("UPDATE assertions SET supersedes_assertion_id=? WHERE assertion_id=?", (assertion_id, new_item["assertion_id"]))
            if old_item["case_id"]:
                self._append_event_locked(old_item["case_id"], "assertion_corrected", actor_type, source_type="assertion", source_id=new_item["assertion_id"], payload={"supersedes": assertion_id})
            self.conn.commit()
        if old_item["case_id"]:
            self.refresh_case_projection(old_item["case_id"])
        new_item["supersedes_assertion_id"] = assertion_id
        return new_item

    def set_assertion_status(self, assertion_id: str, *, workspace_id: str | None = None, status: str, actor_type: str = "user") -> dict[str, Any]:
        if status not in ASSERTION_STATUSES:
            raise ValueError("Invalid assertion status")
        with self._lock:
            if workspace_id is None:
                row = self.conn.execute("SELECT * FROM assertions WHERE assertion_id=?", (assertion_id,)).fetchone()
            else:
                row = self.conn.execute("SELECT * FROM assertions WHERE assertion_id=? AND workspace_id=?", (assertion_id, workspace_id)).fetchone()
            if row is None:
                raise KeyError("Assertion not found")
            item = self._assertion_row(row)
            now = _now()
            self.conn.execute("UPDATE assertions SET status=?, updated_at=? WHERE assertion_id=?", (status, now, assertion_id))
            if item["case_id"]:
                self._append_event_locked(item["case_id"], f"assertion_{status}", actor_type, source_type="assertion", source_id=assertion_id)
            self.conn.commit()
        if item["case_id"]:
            self.refresh_case_projection(item["case_id"])
        item["status"], item["updated_at"] = status, now
        return item

    def create_task(self, *, case_id: str, title: str, workspace_id: str | None = None, rationale: str = "", priority: str = "medium", status: str = "proposed", actor_type: str = "assistant", source_assertion_id: str | None = None) -> dict[str, Any]:
        if status not in TASK_STATUSES:
            raise ValueError("Invalid task status")
        self._require_mutable_case(case_id, workspace_id=workspace_id)
        record = {"task_id": _id("tsk"), "case_id": case_id, "title": " ".join(title.split())[:240], "rationale": " ".join(rationale.split())[:1000], "priority": priority, "status": status, "source_assertion_id": source_assertion_id, "created_by": actor_type, "created_at": _now(), "updated_at": _now()}
        with self._lock:
            self.conn.execute("INSERT INTO tasks(task_id, case_id, title, rationale, priority, status, source_assertion_id, created_by, created_at, updated_at) VALUES (:task_id,:case_id,:title,:rationale,:priority,:status,:source_assertion_id,:created_by,:created_at,:updated_at)", record)
            self._append_event_locked(case_id, "task_created", actor_type, source_type="task", source_id=record["task_id"], payload={"title": record["title"], "status": status})
            self.conn.commit()
        self.refresh_case_projection(case_id)
        return record

    def list_tasks(self, case_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM tasks WHERE case_id=? ORDER BY CASE status WHEN 'open' THEN 0 WHEN 'blocked' THEN 1 WHEN 'proposed' THEN 2 ELSE 3 END, updated_at DESC", (case_id,)).fetchall()
        return [dict(row) for row in rows]

    def update_task(self, task_id: str, *, workspace_id: str | None = None, status: str, actor_type: str = "user") -> dict[str, Any]:
        if status not in TASK_STATUSES:
            raise ValueError("Invalid task status")
        with self._lock:
            if workspace_id is None:
                row = self.conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            else:
                row = self.conn.execute("SELECT tasks.* FROM tasks JOIN cases ON cases.case_id=tasks.case_id WHERE tasks.task_id=? AND cases.workspace_id=?", (task_id, workspace_id)).fetchone()
            if row is None:
                raise KeyError("Task not found")
            record, now = dict(row), _now()
            self.conn.execute("UPDATE tasks SET status=?, updated_at=? WHERE task_id=?", (status, now, task_id))
            self._append_event_locked(record["case_id"], "task_status_changed", actor_type, source_type="task", source_id=task_id, payload={"status": status})
            self.conn.commit()
        self.refresh_case_projection(record["case_id"])
        record["status"], record["updated_at"] = status, now
        return record

    def list_decisions(self, case_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM decisions WHERE case_id=? ORDER BY updated_at DESC", (case_id,)).fetchall()
        return [dict(row) for row in rows]

    def project_case(self, case_id: str) -> dict[str, Any]:
        with self._lock:
            case = self.conn.execute("SELECT workspace_id, updated_at FROM cases WHERE case_id=?", (case_id,)).fetchone()
        if case is None:
            raise KeyError("Case not found")
        facts = self.list_assertions(workspace_id=case["workspace_id"], case_id=case_id, statuses=["confirmed"])
        proposed = self.list_assertions(workspace_id=case["workspace_id"], case_id=case_id, statuses=["proposed"])
        active_tasks = [task for task in self.list_tasks(case_id) if task["status"] in {"open", "blocked", "proposed"}]
        conflicts = self._find_conflicts(case_id, case["workspace_id"])
        summary = " ".join(item["display_text"] or f"{item['key']}: {item['value_text']}" for item in facts[:3])
        projection = {"case_id": case_id, "summary": summary, "confirmed_facts": facts, "proposed_assertions": proposed, "active_tasks": active_tasks, "conflicts": conflicts, "review_count": len(proposed) + len(conflicts), "updated_at": None}
        projection["updated_at"] = case["updated_at"]
        return projection

    def refresh_case_projection(self, case_id: str) -> dict[str, Any]:
        projection = self.project_case(case_id)
        projection["updated_at"] = _now()
        with self._lock:
            self.conn.execute("INSERT INTO case_state_projection(case_id, projection_json, updated_at) VALUES (?, ?, ?) ON CONFLICT(case_id) DO UPDATE SET projection_json=excluded.projection_json, updated_at=excluded.updated_at", (case_id, _json(projection), projection["updated_at"]))
            self.conn.execute("UPDATE cases SET summary=?, updated_at=? WHERE case_id=?", (projection["summary"][:1000], projection["updated_at"], case_id))
            self.conn.commit()
        return projection

    def _find_conflicts(self, case_id: str, workspace_id: str) -> list[dict[str, Any]]:
        confirmed = self.list_assertions(workspace_id=workspace_id, case_id=case_id, statuses=["confirmed"])
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in confirmed:
            grouped.setdefault(item["key"].lower(), []).append(item)
        return [{"key": key, "assertions": items} for key, items in grouped.items() if len({item["value_text"].lower() for item in items}) > 1]

    def build_context(self, *, case_id: str, workspace_id: str, query: str, conversation_id: str | None = None, max_items: int = 12) -> dict[str, Any]:
        case = self.get_case(case_id, workspace_id=workspace_id)["case"]
        query_tokens = {token.lower() for token in query.split() if len(token) >= 3}
        candidates: list[tuple[int, dict[str, Any], str]] = []
        for assertion in self.list_assertions(workspace_id=workspace_id, case_id=case_id, statuses=["confirmed"]):
            text = f"{assertion['key']} {assertion['value_text']} {assertion['display_text']}".lower()
            matches = sum(token in text for token in query_tokens)
            candidates.append((20 + matches * 10, {"source_type": "assertion", "source_id": assertion["assertion_id"], "label": assertion["display_text"] or f"{assertion['key']}: {assertion['value_text']}"}, "hecho confirmado del caso" if matches else "hecho vigente del caso"))
        for assertion in self.list_assertions(workspace_id=workspace_id, scope="global", statuses=["confirmed"]):
            text = f"{assertion['key']} {assertion['value_text']} {assertion['display_text']}".lower()
            matches = sum(token in text for token in query_tokens)
            if matches:
                candidates.append((10 + matches * 8, {"source_type": "assertion", "source_id": assertion["assertion_id"], "label": assertion["display_text"] or f"{assertion['key']}: {assertion['value_text']}"}, "memoria global relevante"))
        for task in self.list_tasks(case_id):
            if task["status"] in {"open", "blocked"}:
                candidates.append((18, {"source_type": "task", "source_id": task["task_id"], "label": task["title"]}, "tarea pendiente del caso"))
        for event in self.list_events(case_id, limit=4):
            candidates.append((5, {"source_type": "event", "source_id": event["event_id"], "label": event["event_type"]}, "evento reciente del caso"))
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = []
        seen = set()
        for rank, (_score, item, reason) in enumerate(candidates, start=1):
            key = (item["source_type"], item["source_id"])
            if key in seen or len(selected) >= max_items:
                continue
            seen.add(key)
            selected.append({**item, "reason": reason, "rank": len(selected) + 1})
        run_id = _id("ctx")
        with self._lock:
            self.conn.execute("INSERT INTO context_runs(context_run_id, case_id, conversation_id, query, created_at) VALUES (?, ?, ?, ?, ?)", (run_id, case_id, conversation_id, query[:2000], _now()))
            self.conn.executemany("INSERT INTO context_run_items(context_run_id, source_type, source_id, reason, rank) VALUES (?, ?, ?, ?, ?)", [(run_id, item["source_type"], item["source_id"], item["reason"], item["rank"]) for item in selected])
            self.conn.commit()
        lines = [f"Caso activo: {case['title']}"]
        for item in selected:
            lines.append(f"- {item['label']} ({item['reason']})")
        return {"context_run_id": run_id, "text": "\n".join(lines), "items": selected}

    def record_assistant_result(self, *, case_id: str, workspace_id: str | None = None, conversation_id: str | None, query: str, executive_summary: str, case_state: Any = None, attachment_ids: Iterable[str] = ()) -> None:
        self.get_case(case_id, workspace_id=workspace_id)
        with self._lock:
            self._append_event_locked(case_id, "user_message_recorded", "user", source_type="conversation", source_id=conversation_id, payload={"query": query[:1000]})
            self._append_event_locked(case_id, "assistant_response_recorded", "assistant", source_type="conversation", source_id=conversation_id, payload={"summary": (executive_summary or "")[:1000]})
            self.conn.commit()
        for attachment_id in attachment_ids:
            self.link_document(case_id=case_id, workspace_id=workspace_id, attachment_id=attachment_id)
        known_task_titles = {item["title"].strip().lower() for item in self.list_tasks(case_id)}
        for task in list(getattr(case_state, "open_tasks", []) or []):
            title = str(getattr(task, "title", "")).strip()
            if not title or title.lower() in known_task_titles:
                continue
            self.create_task(case_id=case_id, workspace_id=workspace_id, title=title, rationale=str(getattr(task, "rationale", "")), priority=str(getattr(task, "priority", "medium")), status="proposed", actor_type="assistant")
            known_task_titles.add(title.lower())
        self.refresh_case_projection(case_id)

    def close(self) -> None:
        with self._lock:
            self.conn.close()


case_store = CaseStore()
atexit.register(case_store.close)
