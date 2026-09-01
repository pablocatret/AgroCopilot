from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from backend.deps import settings


logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_title(query: str) -> str:
    words = query.strip().split()[:8]
    title = " ".join(words)
    if len(query.strip()) > len(title):
        title += "..."
    return title


class ConversationStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or settings.CONVERSATIONS_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT DEFAULT '',
                title TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                metadata_json TEXT DEFAULT '{}',
                case_id TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                query TEXT DEFAULT '',
                response_mode TEXT DEFAULT 'conversation',
                answer_summary TEXT DEFAULT '',
                answer_json TEXT DEFAULT '{}',
                file_names TEXT DEFAULT '[]',
                timestamp TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at)"
        )
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(conversations)").fetchall()}
        if "case_id" not in columns:
            self.conn.execute("ALTER TABLE conversations ADD COLUMN case_id TEXT")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_case ON conversations(case_id)")
        self.conn.commit()

    def save_conversation(
        self,
        conversation_id: str,
        user_id: str = "",
        title: str = "",
        case_id: str | None = None,
    ) -> None:
        now = _now_iso()
        self.conn.execute(
            """
            INSERT INTO conversations
                (conversation_id, user_id, title, created_at, updated_at, message_count, case_id)
            VALUES (?, ?, ?, ?, ?, 0, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                user_id = CASE WHEN excluded.user_id != '' THEN excluded.user_id ELSE conversations.user_id END,
                title = CASE WHEN excluded.title != '' THEN excluded.title ELSE conversations.title END,
                updated_at = excluded.updated_at,
                case_id = COALESCE(excluded.case_id, conversations.case_id)
            """,
            (conversation_id, user_id, title, now, now, case_id),
        )
        self.conn.commit()

    def update_title(self, conversation_id: str, title: str) -> None:
        self.conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE conversation_id = ?",
            (title, _now_iso(), conversation_id),
        )
        self.conn.commit()

    def list_conversations(self, limit: int = 50) -> List[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT conversation_id, user_id, title, created_at, updated_at, message_count, case_id
            FROM conversations
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            """
            SELECT conversation_id, user_id, title, created_at, updated_at, message_count, case_id
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        return dict(row) if row else None

    def save_message(
        self,
        conversation_id: str,
        role: str,
        query: str = "",
        response_mode: str = "conversation",
        answer_summary: str = "",
        answer_json: str = "{}",
        file_names: List[str] | None = None,
    ) -> None:
        now = _now_iso()
        self.conn.execute(
            """
            INSERT INTO messages
                (conversation_id, role, query, response_mode, answer_summary, answer_json, file_names, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                query,
                response_mode,
                answer_summary,
                answer_json,
                json.dumps(file_names or []),
                now,
            ),
        )
        self.conn.execute(
            """
            UPDATE conversations
            SET message_count = message_count + 1, updated_at = ?
            WHERE conversation_id = ?
            """,
            (now, conversation_id),
        )
        self.conn.commit()

    def get_messages(self, conversation_id: str, limit: int = 20) -> List[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT id, role, query, response_mode, answer_summary, answer_json, file_names, timestamp
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        messages = [dict(row) for row in reversed(rows)]
        for msg in messages:
            try:
                msg["file_names"] = json.loads(msg.get("file_names", "[]"))
            except Exception:
                msg["file_names"] = []
        return messages

    def delete_conversation(self, conversation_id: str) -> bool:
        self.conn.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        cursor = self.conn.execute(
            "DELETE FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def link_case(self, conversation_id: str, case_id: str) -> None:
        self.conn.execute(
            "UPDATE conversations SET case_id = ?, updated_at = ? WHERE conversation_id = ?",
            (case_id, _now_iso(), conversation_id),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


conversation_store = ConversationStore()
