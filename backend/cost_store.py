from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any

from backend.deps import settings


class CostStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or settings.COST_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cost_events (
                id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                conversation_id TEXT,
                agent TEXT,
                operation TEXT,
                provider TEXT,
                model TEXT,
                pricing_mode TEXT,
                input_tokens INTEGER DEFAULT 0,
                cached_input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                unit_count INTEGER DEFAULT 0,
                estimated INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0,
                metadata_json TEXT DEFAULT '{}'
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cost_events_conversation ON cost_events(conversation_id)"
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cost_events_ts ON cost_events(ts)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cost_events_agent ON cost_events(agent)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cost_events_model ON cost_events(model)")
        self.conn.commit()

    def insert_event(self, event: dict[str, Any]) -> None:
        ts = event.get("ts") or dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        )
        self.conn.execute(
            """
            INSERT OR IGNORE INTO cost_events (
                id, ts, conversation_id, agent, operation, provider, model, pricing_mode,
                input_tokens, cached_input_tokens, output_tokens, total_tokens, unit_count,
                estimated, cost_usd, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["id"],
                ts,
                event.get("conversation_id"),
                event.get("agent"),
                event.get("operation"),
                event.get("provider"),
                event.get("model"),
                event.get("pricing_mode"),
                int(event.get("input_tokens") or 0),
                int(event.get("cached_input_tokens") or 0),
                int(event.get("output_tokens") or 0),
                int(event.get("total_tokens") or 0),
                int(event.get("unit_count") or 0),
                1 if event.get("estimated") else 0,
                float(event.get("cost_usd") or 0.0),
                json.dumps(event.get("metadata") or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def list_conversation(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM cost_events WHERE conversation_id = ? ORDER BY ts ASC",
            (conversation_id,),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def summarize_conversation(self, conversation_id: str) -> dict[str, Any]:
        events = self.list_conversation(conversation_id)
        return self._summarize_events(events, conversation_id=conversation_id)

    def summary(self, days: int = 7) -> dict[str, Any]:
        since = dt.datetime.now(dt.UTC) - dt.timedelta(days=max(1, days))
        rows = self.conn.execute(
            "SELECT * FROM cost_events WHERE ts >= ? ORDER BY ts ASC",
            (since.isoformat(timespec="milliseconds").replace("+00:00", "Z"),),
        ).fetchall()
        events = [self._row_to_event(row) for row in rows]
        summary = self._summarize_events(events, conversation_id=None)
        summary["days"] = days
        summary["by_day"] = self._group(events, lambda event: (event["ts"] or "")[:10])
        return summary

    def _row_to_event(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["estimated"] = bool(data.get("estimated"))
        try:
            data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        except json.JSONDecodeError:
            data["metadata"] = {}
        return data

    def _summarize_events(
        self, events: list[dict[str, Any]], *, conversation_id: str | None
    ) -> dict[str, Any]:
        by_model = self._group(events, lambda event: event.get("model") or "unknown")
        by_agent = self._group(events, lambda event: event.get("agent") or "unknown")
        by_operation = self._group(events, lambda event: event.get("operation") or "unknown")
        top_model = max(
            by_model.items(), key=lambda item: item[1]["cost_usd"], default=(None, None)
        )
        total_cost = sum(float(event.get("cost_usd") or 0.0) for event in events)
        total_tokens = sum(int(event.get("total_tokens") or 0) for event in events)
        warn_limit = (
            settings.COST_WARN_USD_PER_CONVERSATION
            if conversation_id
            else settings.COST_WARN_USD_PER_DAY
        )
        return {
            "conversation_id": conversation_id,
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "input_tokens": sum(int(event.get("input_tokens") or 0) for event in events),
            "cached_input_tokens": sum(
                int(event.get("cached_input_tokens") or 0) for event in events
            ),
            "output_tokens": sum(int(event.get("output_tokens") or 0) for event in events),
            "web_calls": sum(
                int(event.get("unit_count") or 0)
                for event in events
                if "search" in str(event.get("operation") or "")
            ),
            "estimated": any(bool(event.get("estimated")) for event in events),
            "event_count": len(events),
            "top_model": top_model[0],
            "top_model_cost_usd": top_model[1]["cost_usd"] if top_model[1] else 0.0,
            "warning": total_cost >= warn_limit if warn_limit > 0 else False,
            "warning_threshold_usd": warn_limit,
            "by_model": by_model,
            "by_agent": by_agent,
            "by_operation": by_operation,
            "events": events,
        }

    def _group(self, events: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for event in events:
            key = str(key_fn(event) or "unknown")
            bucket = grouped.setdefault(
                key,
                {
                    "cost_usd": 0.0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "unit_count": 0,
                    "events": 0,
                    "estimated": False,
                },
            )
            bucket["cost_usd"] += float(event.get("cost_usd") or 0.0)
            bucket["input_tokens"] += int(event.get("input_tokens") or 0)
            bucket["cached_input_tokens"] += int(event.get("cached_input_tokens") or 0)
            bucket["output_tokens"] += int(event.get("output_tokens") or 0)
            bucket["total_tokens"] += int(event.get("total_tokens") or 0)
            bucket["unit_count"] += int(event.get("unit_count") or 0)
            bucket["events"] += 1
            bucket["estimated"] = bucket["estimated"] or bool(event.get("estimated"))
        return grouped

    def close(self) -> None:
        self.conn.close()


cost_store = CostStore()
