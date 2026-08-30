import json
import sqlite3
import threading
import time
import pytest
from backend.storage.session_store import SessionStore


@pytest.fixture
def temp_store(tmp_path):
    db_file = str(tmp_path / "test_sessions.db")
    store = SessionStore(db_file)
    yield store
    store.close()


@pytest.fixture
def memory_store():
    store = SessionStore(":memory:")
    yield store
    store.close()


def test_database_schema_creation(temp_store):
    conn = temp_store._get_connection()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in cursor.fetchall()}
    assert "sessions" in tables
    assert "messages" in tables

    # Verify indexes
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = {row["name"] for row in cursor.fetchall()}
    assert "idx_messages_session_timestamp" in indexes
    assert "idx_sessions_updated_at" in indexes


def test_wal_mode(temp_store):
    conn = temp_store._get_connection()
    cursor = conn.execute("PRAGMA journal_mode")
    row = cursor.fetchone()
    assert row[0].lower() == "wal"


def test_foreign_key_enforcement(temp_store):
    with pytest.raises(sqlite3.IntegrityError):
        temp_store.add_message(
            session_id="nonexistent-session-id",
            role="user",
            content="Hello world",
        )


def test_session_creation(memory_store):
    session = memory_store.create_session("Test Chat")
    assert session["id"] is not None
    assert session["title"] == "Test Chat"
    assert session["created_at"] > 0
    assert session["updated_at"] == session["created_at"]

    # Custom ID
    custom_session = memory_store.create_session("Custom", session_id="custom-123")
    assert custom_session["id"] == "custom-123"


def test_session_retrieval(memory_store):
    created = memory_store.create_session("Retrieve Me")
    retrieved = memory_store.get_session(created["id"])
    assert retrieved is not None
    assert retrieved["id"] == created["id"]
    assert retrieved["title"] == "Retrieve Me"

    # Non-existent session
    assert memory_store.get_session("unknown-id") is None


def test_session_listing(memory_store):
    s1 = memory_store.create_session("Session 1")
    time.sleep(0.01)
    s2 = memory_store.create_session("Session 2")
    time.sleep(0.01)
    s3 = memory_store.create_session("Session 3")

    sessions = memory_store.list_sessions()
    assert len(sessions) == 3
    # Newest updated first
    assert sessions[0]["id"] == s3["id"]
    assert sessions[1]["id"] == s2["id"]
    assert sessions[2]["id"] == s1["id"]


def test_session_update(memory_store):
    s = memory_store.create_session("Old Title")
    time.sleep(0.01)
    updated = memory_store.update_session_title(s["id"], "New Title")
    assert updated is not None
    assert updated["title"] == "New Title"
    assert updated["updated_at"] > s["updated_at"]

    # Invalid session
    assert memory_store.update_session_title("invalid-id", "Title") is None


def test_session_deletion(memory_store):
    s = memory_store.create_session("To Delete")
    assert memory_store.get_session(s["id"]) is not None

    deleted = memory_store.delete_session(s["id"])
    assert deleted is True
    assert memory_store.get_session(s["id"]) is None

    # Deleting non-existent session
    assert memory_store.delete_session("unknown-id") is False


def test_message_insertion(memory_store):
    s = memory_store.create_session("Chat")
    m1 = memory_store.add_message(s["id"], "user", "What is planning?")
    assert m1["id"] is not None
    assert m1["session_id"] == s["id"]
    assert m1["role"] == "user"
    assert m1["content"] == "What is planning?"
    assert m1["tool_calls_json"] is None
    assert m1["timestamp"] > 0


def test_message_retrieval(memory_store):
    s = memory_store.create_session("Chat")
    memory_store.add_message(s["id"], "user", "Hello")
    memory_store.add_message(s["id"], "assistant", "Hi Sir, how can I assist?")

    msgs = memory_store.get_messages(s["id"])
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_message_ordering(memory_store):
    s = memory_store.create_session("Order Test")
    t0 = time.time()
    memory_store.add_message(s["id"], "user", "Msg 1", timestamp=t0)
    memory_store.add_message(s["id"], "assistant", "Msg 2", timestamp=t0 + 1)
    memory_store.add_message(s["id"], "user", "Msg 3", timestamp=t0 + 2)

    msgs = memory_store.get_messages(s["id"])
    assert len(msgs) == 3
    assert [m["content"] for m in msgs] == ["Msg 1", "Msg 2", "Msg 3"]


def test_session_message_isolation(memory_store):
    s1 = memory_store.create_session("Session A")
    s2 = memory_store.create_session("Session B")

    memory_store.add_message(s1["id"], "user", "Message A")
    memory_store.add_message(s2["id"], "user", "Message B")

    msgs_a = memory_store.get_messages(s1["id"])
    msgs_b = memory_store.get_messages(s2["id"])

    assert len(msgs_a) == 1
    assert msgs_a[0]["content"] == "Message A"

    assert len(msgs_b) == 1
    assert msgs_b[0]["content"] == "Message B"


def test_deletion_cascade(memory_store):
    s = memory_store.create_session("Cascade Test")
    memory_store.add_message(s["id"], "user", "Hello")
    memory_store.add_message(s["id"], "assistant", "World")

    assert len(memory_store.get_messages(s["id"])) == 2

    # Delete parent session
    memory_store.delete_session(s["id"])

    # Messages should be deleted via ON DELETE CASCADE
    assert len(memory_store.get_messages(s["id"])) == 0

    conn = memory_store._get_connection()
    cursor = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (s["id"],))
    assert cursor.fetchone()[0] == 0


def test_unicode_content(memory_store):
    s = memory_store.create_session("Multilingual 🚀")
    unicode_str = "こんにちは世界！ 🤖 P.I.X.I.E. supports Unicode: 🚀, 💬, こんにちは, Hello,Привет."
    m = memory_store.add_message(s["id"], "user", unicode_str)

    msgs = memory_store.get_messages(s["id"])
    assert msgs[0]["content"] == unicode_str


def test_long_content(memory_store):
    s = memory_store.create_session("Long Chat")
    large_text = "A" * 150000  # 150k characters
    m = memory_store.add_message(s["id"], "assistant", large_text)

    msgs = memory_store.get_messages(s["id"])
    assert len(msgs[0]["content"]) == 150000
    assert msgs[0]["content"] == large_text


def test_empty_session(memory_store):
    s = memory_store.create_session("Empty")
    msgs = memory_store.get_messages(s["id"])
    assert msgs == []


def test_invalid_session_handling(memory_store):
    assert memory_store.get_session("nonexistent") is None
    assert memory_store.update_session_title("nonexistent", "Title") is None
    assert memory_store.delete_session("nonexistent") is False
    assert memory_store.get_messages("nonexistent") == []


def test_tool_calls_json_persistence(memory_store):
    s = memory_store.create_session("Tool Chat")
    tool_payload = json.dumps([
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "system_diagnostics", "arguments": "{\"verbose\": true}"}
        }
    ])
    m = memory_store.add_message(s["id"], "assistant", "", tool_calls_json=tool_payload)

    msgs = memory_store.get_messages(s["id"])
    assert msgs[0]["tool_calls_json"] == tool_payload
    parsed = json.loads(msgs[0]["tool_calls_json"])
    assert parsed[0]["function"]["name"] == "system_diagnostics"


def test_repeated_database_reopen(tmp_path):
    db_file = str(tmp_path / "persistent_sessions.db")

    store1 = SessionStore(db_file)
    s = store1.create_session("Reopen Session")
    store1.add_message(s["id"], "user", "Persistent message")
    store1.close()

    # Reopen database with new store instance
    store2 = SessionStore(db_file)
    retrieved_s = store2.get_session(s["id"])
    assert retrieved_s is not None
    assert retrieved_s["title"] == "Reopen Session"

    msgs = store2.get_messages(s["id"])
    assert len(msgs) == 1
    assert msgs[0]["content"] == "Persistent message"
    store2.close()


def test_concurrent_database_access(tmp_path):
    db_file = str(tmp_path / "concurrent_sessions.db")
    store = SessionStore(db_file)
    s = store.create_session("Concurrent Session")

    errors = []

    def worker(worker_id: int):
        try:
            for i in range(10):
                store.add_message(
                    s["id"],
                    "user",
                    f"Worker {worker_id} message {i}",
                )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    msgs = store.get_messages(s["id"])
    assert len(msgs) == 50
    store.close()
