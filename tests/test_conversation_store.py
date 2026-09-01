"""Tests for conversation_store.py — multi-conversation persistence with SQLite."""
from __future__ import annotations

import time
import uuid
from typing import Generator

import pytest

from backend.conversation_store import ConversationStore


@pytest.fixture()
def store(tmp_path) -> Generator[ConversationStore, None, None]:
    """Provide a fresh ConversationStore using a temporary SQLite database."""
    db_path = tmp_path / "conversations.db"
    s = ConversationStore(path=str(db_path))
    yield s


def _new_id() -> str:
    return str(uuid.uuid4())


class TestConversationStoreBasicOperations:
    def test_save_and_get_conversation(self, store: ConversationStore) -> None:
        cid = _new_id()
        store.save_conversation(cid, user_id="user-1", title="Test query about maize")
        conv = store.get_conversation(cid)
        assert conv is not None
        assert conv["conversation_id"] == cid
        assert conv["user_id"] == "user-1"
        assert conv["title"] == "Test query about maize"
        assert conv["message_count"] == 0

    def test_list_conversations_default_order(self, store: ConversationStore) -> None:
        c1 = _new_id()
        c2 = _new_id()
        c3 = _new_id()
        store.save_conversation(c1, title="First query")
        time.sleep(0.02)
        store.save_conversation(c2, title="Second query")
        time.sleep(0.02)
        store.save_conversation(c3, title="Third query")

        convs = store.list_conversations()
        assert len(convs) == 3
        # Most recently updated first
        assert convs[0]["conversation_id"] == c3
        assert convs[1]["conversation_id"] == c2
        assert convs[2]["conversation_id"] == c1

    def test_update_title(self, store: ConversationStore) -> None:
        cid = _new_id()
        store.save_conversation(cid, title="Original title")
        store.update_title(cid, "Updated title")
        updated = store.get_conversation(cid)
        assert updated is not None
        assert updated["title"] == "Updated title"

    def test_delete_conversation(self, store: ConversationStore) -> None:
        cid = _new_id()
        store.save_conversation(cid, title="To delete")
        store.save_message(cid, "user", "Hello", "simple", "Hi there", file_names=None)
        assert store.get_conversation(cid) is not None

        store.delete_conversation(cid)
        assert store.get_conversation(cid) is None
        msgs = store.get_messages(cid)
        assert len(msgs) == 0

    def test_delete_nonexistent_is_noop(self, store: ConversationStore) -> None:
        store.delete_conversation("nonexistent-id")

    def test_get_conversation_nonexistent(self, store: ConversationStore) -> None:
        assert store.get_conversation("nonexistent") is None


class TestMessageOperations:
    def test_save_and_get_messages(self, store: ConversationStore) -> None:
        cid = _new_id()
        store.save_conversation(cid, title="Test")
        store.save_message(cid, "user", "What is maize?", "simple", "Maize is corn.", file_names=None)
        store.save_message(cid, "assistant", "Maize is corn.", "simple", "Maize is corn.", file_names=["maize.pdf"])

        msgs = store.get_messages(cid)
        assert len(msgs) == 2
        assert msgs[0]["query"] == "What is maize?"
        assert msgs[0]["role"] == "user"
        assert msgs[1]["answer_summary"] == "Maize is corn."
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["file_names"] == ["maize.pdf"]

    def test_messages_order_by_timestamp(self, store: ConversationStore) -> None:
        cid = _new_id()
        store.save_conversation(cid, title="Test")
        store.save_message(cid, "user", "Q1", "simple", "A1", file_names=None)
        time.sleep(0.02)
        store.save_message(cid, "assistant", "A1", "simple", "A1", file_names=None)
        time.sleep(0.02)
        store.save_message(cid, "user", "Q2", "simple", "A2", file_names=None)

        msgs = store.get_messages(cid)
        assert len(msgs) == 3
        assert msgs[0]["query"] == "Q1"
        assert msgs[1]["query"] == "A1"
        assert msgs[2]["query"] == "Q2"

    def test_message_count_updates(self, store: ConversationStore) -> None:
        cid = _new_id()
        store.save_conversation(cid, title="Test")
        conv = store.get_conversation(cid)
        assert conv is not None
        assert conv["message_count"] == 0

        store.save_message(cid, "user", "Q1", "simple", "A1", file_names=None)
        store.save_message(cid, "assistant", "A1", "simple", "A1", file_names=None)

        updated = store.get_conversation(cid)
        assert updated is not None
        assert updated["message_count"] == 2

    def test_messages_for_nonexistent_conversation(self, store: ConversationStore) -> None:
        msgs = store.get_messages("nonexistent")
        assert len(msgs) == 0

    def test_file_names_default_empty(self, store: ConversationStore) -> None:
        cid = _new_id()
        store.save_conversation(cid, title="Test")
        store.save_message(cid, "user", "Q", "simple", "A", file_names=None)
        msgs = store.get_messages(cid)
        assert msgs[0]["file_names"] == []

    def test_delete_cascades_messages(self, store: ConversationStore) -> None:
        cid = _new_id()
        store.save_conversation(cid, title="Test")
        store.save_message(cid, "user", "Q1", "simple", "A1", file_names=None)
        store.save_message(cid, "assistant", "A1", "simple", "A1", file_names=None)
        store.save_message(cid, "user", "Q2", "simple", "A2", file_names=None)

        store.delete_conversation(cid)
        msgs = store.get_messages(cid)
        assert len(msgs) == 0


class TestMakeTitle:
    def test_short_query(self) -> None:
        from backend.conversation_store import _make_title
        assert _make_title("Hello world") == "Hello world"

    def test_long_query_truncated(self) -> None:
        from backend.conversation_store import _make_title
        title = _make_title("This is a very long query about agricultural topics in Colombia")
        assert len(title) < len("This is a very long query about agricultural topics in Colombia")
        assert title.endswith("...")

    def test_empty_query(self) -> None:
        from backend.conversation_store import _make_title
        assert _make_title("") == ""
