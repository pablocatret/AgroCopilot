# backend/events.py
from __future__ import annotations

import asyncio
import datetime as dt
import json
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Dict, List, Protocol

from backend.deps import settings

LOG_DIR = Path(__file__).resolve().parent / "logs"
SENT_LOG = LOG_DIR / "sse_emitted.log"
DELIVERED_LOG = LOG_DIR / "sse_received.log"
TRACE_ENABLED = settings.SSE_TRACE
CONVERSATION_LOG_DIR = LOG_DIR / "conversations"

_TRACE_STATE: Dict[str, Dict[str, List[dict]]] = {}


@dataclass
class ConversationState:
    history: List[str] = field(default_factory=list)
    subscribers: Dict[str, asyncio.Queue[str]] = field(default_factory=dict)
    finished: bool = False


HISTORY_LIMIT = 100
COMPLETED_SENTINEL = object()


def _append_log(path: Path, payload: dict) -> None:
    mode = settings.SSE_LOG_MODE
    line = json.dumps(payload, ensure_ascii=False)
    if mode == "stdout":
        print(line)
        return
    if mode != "disk":
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _conversation_log_path(conversation_id: str) -> Path:
    safe_id = "".join(ch for ch in str(conversation_id) if ch.isalnum() or ch in {"-", "_"})
    return CONVERSATION_LOG_DIR / f"{safe_id}.jsonl"


def _append_conversation_event(conversation_id: str, event: dict) -> None:
    if settings.SSE_LOG_MODE != "disk":
        return
    CONVERSATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = _conversation_log_path(conversation_id)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def _load_conversation_events(conversation_id: str) -> tuple[list[str], bool]:
    if settings.SSE_LOG_MODE != "disk":
        return [], False
    path = _conversation_log_path(conversation_id)
    if not path.exists():
        return [], False
    payloads: list[str] = []
    finished = False
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            payloads.append(text)
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                event = {}
            if event.get("type") == "status" and event.get("stage") == "completed":
                finished = True
    return payloads[-HISTORY_LIMIT:], finished


class ConversationEventStore(Protocol):
    def load(self, conversation_id: str) -> tuple[list[str], bool]: ...

    def append(self, conversation_id: str, event: dict) -> None: ...

    def has_conversation(self, conversation_id: str) -> bool: ...


class NoopConversationEventStore:
    def load(self, conversation_id: str) -> tuple[list[str], bool]:
        return [], False

    def append(self, conversation_id: str, event: dict) -> None:
        return None

    def has_conversation(self, conversation_id: str) -> bool:
        return False


class DiskConversationEventStore:
    def load(self, conversation_id: str) -> tuple[list[str], bool]:
        return _load_conversation_events(conversation_id)

    def append(self, conversation_id: str, event: dict) -> None:
        _append_conversation_event(conversation_id, event)

    def has_conversation(self, conversation_id: str) -> bool:
        return _conversation_log_path(conversation_id).exists()


def _event_store() -> ConversationEventStore:
    if settings.SSE_LOG_MODE == "disk":
        return DiskConversationEventStore()
    return NoopConversationEventStore()


def _ts() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class EventBroker:
    """Broker en memoria: una asyncio.Queue por conversation_id."""

    def __init__(self, *, event_store: ConversationEventStore | None = None) -> None:
        self.conversations: Dict[str, ConversationState] = {}
        self.event_store = event_store or _event_store()

    def _get_state(self, conversation_id: str) -> ConversationState:
        if conversation_id not in self.conversations:
            history, finished = self.event_store.load(conversation_id)
            self.conversations[conversation_id] = ConversationState(
                history=history,
                finished=finished,
            )
        return self.conversations[conversation_id]

    def _maybe_cleanup(self, conversation_id: str) -> None:
        state = self.conversations.get(conversation_id)
        if not state:
            return
        if state.finished and not state.subscribers:
            self.conversations.pop(conversation_id, None)
            _TRACE_STATE.pop(conversation_id, None)

    async def publish(self, conversation_id: str, event: dict) -> None:
        state = self._get_state(conversation_id)
        payload = json.dumps(event)
        state.history.append(payload)
        if len(state.history) > HISTORY_LIMIT:
            state.history = state.history[-HISTORY_LIMIT:]
        self.event_store.append(conversation_id, event)
        for queue in list(state.subscribers.values()):
            await queue.put(payload)
        if event.get("type") == "status" and event.get("stage") == "completed":
            state.finished = True
        _append_log(
            SENT_LOG,
            {
                "ts": _ts(),
                "conversation_id": conversation_id,
                "event": event,
            },
        )
        if TRACE_ENABLED:
            _trace_event(conversation_id, event, bucket="emitted")

    async def stream(self, conversation_id: str) -> AsyncIterator[str]:
        state = self._get_state(conversation_id)
        stream_id = uuid.uuid4().hex
        queue: asyncio.Queue[str | object] = asyncio.Queue()
        for item in state.history:
            await queue.put(item)
        if state.finished:
            await queue.put(COMPLETED_SENTINEL)
        state.subscribers[stream_id] = queue
        try:
            while True:
                data = await queue.get()
                if data is COMPLETED_SENTINEL:
                    break
                yield data
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    payload = {}
                if payload.get("type") == "status" and payload.get("stage") == "completed":
                    break
        finally:
            state.subscribers.pop(stream_id, None)
            self._maybe_cleanup(conversation_id)

    def has_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self.conversations:
            return True
        return self.event_store.has_conversation(conversation_id)


def record_delivery(conversation_id: str, data: str, stream_id: str) -> None:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        payload = {"raw": data}
    _append_log(
        DELIVERED_LOG,
        {
            "ts": _ts(),
            "conversation_id": conversation_id,
            "stream_id": stream_id,
            "event": payload,
        },
    )
    if TRACE_ENABLED:
        _trace_event(conversation_id, payload, bucket="delivered", extra={"stream_id": stream_id})


broker = EventBroker()


def _trace_event(
    conversation_id: str, payload: dict, *, bucket: str, extra: dict | None = None
) -> None:
    state = _TRACE_STATE.setdefault(conversation_id, {"emitted": [], "delivered": []})
    record = {"ts": _ts(), "event": payload}
    if extra:
        record.update(extra)
    state[bucket].append(record)


def generate_trace_report(conversation_id: str) -> dict | None:
    state = _TRACE_STATE.get(conversation_id)
    if not state:
        return None
    emitted = state.get("emitted", [])
    delivered = state.get("delivered", [])

    def signature(item: dict) -> str:
        return json.dumps(item.get("event"), sort_keys=True)

    emitted_ctr = Counter(signature(item) for item in emitted)
    delivered_ctr = Counter(signature(item) for item in delivered)
    missing = []
    for sig, count in emitted_ctr.items():
        delta = count - delivered_ctr.get(sig, 0)
        if delta > 0:
            missing.append({"event": json.loads(sig), "count": delta})

    latest_status: Dict[str, str] = {}
    plan_agents: List[str] = []
    for item in emitted:
        event = item["event"]
        if event.get("type") == "agent_status":
            actor = event.get("actor") or event.get("agent")
            if actor:
                latest_status[actor] = event.get("status", "unknown")
        if event.get("type") == "plan":
            plan_agents = event.get("steps") or event.get("agents") or []

    return {
        "conversation_id": conversation_id,
        "emitted": len(emitted),
        "delivered": len(delivered),
        "missing_events": missing,
        "latest_status": latest_status,
        "plan_agents": plan_agents,
    }


def write_trace_report(conversation_id: str) -> None:
    if not TRACE_ENABLED or settings.SSE_LOG_MODE != "disk":
        return
    report = generate_trace_report(conversation_id)
    if not report:
        return
    path = LOG_DIR / f"debug_{conversation_id}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
