import asyncio

import pytest

from backend import events
from backend.events import DiskConversationEventStore, EventBroker, NoopConversationEventStore


@pytest.mark.asyncio
async def test_event_broker_cleans_finished_conversation_after_stream_closes():
    broker = EventBroker()
    conversation_id = "conv-finished"

    stream = broker.stream(conversation_id)
    first_item = asyncio.create_task(stream.__anext__())
    await broker.publish(conversation_id, {"type": "status", "stage": "completed"})

    payload = await first_item
    assert "completed" in payload
    assert broker.has_conversation(conversation_id)

    await stream.aclose()

    assert not broker.has_conversation(conversation_id)


@pytest.mark.asyncio
async def test_event_broker_broadcasts_to_multiple_subscribers_and_closes_on_completed():
    broker = EventBroker()
    conversation_id = "conv-multi"
    first_stream = broker.stream(conversation_id)
    second_stream = broker.stream(conversation_id)

    first_item = asyncio.create_task(first_stream.__anext__())
    second_item = asyncio.create_task(second_stream.__anext__())

    await broker.publish(
        conversation_id, {"type": "agent_status", "agent": "legal", "status": "running"}
    )

    assert "legal" in await asyncio.wait_for(first_item, timeout=1)
    assert "legal" in await asyncio.wait_for(second_item, timeout=1)

    first_done = asyncio.create_task(first_stream.__anext__())
    second_done = asyncio.create_task(second_stream.__anext__())
    await broker.publish(conversation_id, {"type": "status", "stage": "completed"})

    assert "completed" in await asyncio.wait_for(first_done, timeout=1)
    assert "completed" in await asyncio.wait_for(second_done, timeout=1)

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(first_stream.__anext__(), timeout=1)
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(second_stream.__anext__(), timeout=1)

    assert not broker.has_conversation(conversation_id)


@pytest.mark.asyncio
async def test_event_broker_replays_history_from_disk_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(events.settings, "SSE_LOG_MODE", "disk")
    monkeypatch.setattr(events, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(events, "SENT_LOG", tmp_path / "logs" / "sse_emitted.log")
    monkeypatch.setattr(events, "DELIVERED_LOG", tmp_path / "logs" / "sse_received.log")
    monkeypatch.setattr(events, "CONVERSATION_LOG_DIR", tmp_path / "logs" / "conversations")

    broker = EventBroker()
    conversation_id = "conv-disk"
    await broker.publish(conversation_id, {"type": "agent_status", "agent": "legal", "status": "running"})
    await broker.publish(conversation_id, {"type": "status", "stage": "completed"})

    reloaded = EventBroker()
    replayed = []
    async for item in reloaded.stream(conversation_id):
        replayed.append(item)

    assert any("legal" in item for item in replayed)
    assert any("completed" in item for item in replayed)
    assert reloaded.has_conversation(conversation_id)


def test_noop_event_store_reports_no_persisted_conversations():
    store = NoopConversationEventStore()

    assert store.load("conv-none") == ([], False)
    assert store.has_conversation("conv-none") is False


def test_disk_event_store_replays_persisted_events(tmp_path, monkeypatch):
    monkeypatch.setattr(events.settings, "SSE_LOG_MODE", "disk")
    monkeypatch.setattr(events, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(events, "CONVERSATION_LOG_DIR", tmp_path / "logs" / "conversations")

    store = DiskConversationEventStore()
    store.append("conv-store", {"type": "agent_status", "agent": "legal", "status": "running"})
    store.append("conv-store", {"type": "status", "stage": "completed"})

    history, finished = store.load("conv-store")

    assert finished is True
    assert any("legal" in item for item in history)
    assert store.has_conversation("conv-store") is True
