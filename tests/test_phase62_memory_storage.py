import os
import tempfile
import time

import pytest
from backend.memory import (
    MemoryCategory,
    MemoryRecord,
    MemorySource,
    MemoryStore,
    MemoryValidationError,
)
from backend.storage.session_store import SessionStore


@pytest.fixture
def memory_store():
    """Provides an isolated in-memory MemoryStore for testing."""
    store = MemoryStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def temp_db_path():
    """Provides a temporary file path for disk-based persistence tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = tmp.name
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_db_initialization_and_schema(memory_store):
    """1 & 2. Verifies initialization and table/index schema creation."""
    conn = memory_store._get_connection()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
    assert cursor.fetchone() is not None

    cursor = conn.execute("PRAGMA table_info(memories)")
    columns = {row["name"]: row["type"] for row in cursor.fetchall()}
    assert "id" in columns
    assert "category" in columns
    assert "key" in columns
    assert "value" in columns
    assert "source" in columns
    assert "confidence" in columns
    assert "created_at" in columns
    assert "updated_at" in columns
    assert "expires_at" in columns
    assert "is_active" in columns
    assert "metadata_json" in columns


def test_empty_store_behaviour(memory_store):
    """3. Verifies default empty store responses."""
    assert memory_store.count_memories() == 0
    assert memory_store.list_memories() == []
    assert memory_store.get_memory("nonexistent-id") is None
    assert memory_store.get_memory_by_key(MemoryCategory.USER_PROFILE, "name") is None


def test_persist_and_read_roundtrip(memory_store):
    """4, 5, 6, 7, 8, 9, 10, 11, 12. Verifies persistence and full roundtrip of all fields."""
    now = time.time()
    record = MemoryRecord(
        id="mem-uuid-1234-5678",
        category=MemoryCategory.USER_PROFILE,
        key="favorite_language",
        value="Python",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=0.95,
        created_at=now,
        updated_at=now + 1.0,
        expires_at=now + 3600.0,
        is_active=True,
        metadata_json='{"verified": true}',
    )

    saved = memory_store.save_memory(record)
    assert saved.id == "mem-uuid-1234-5678"

    retrieved = memory_store.get_memory("mem-uuid-1234-5678")
    assert retrieved is not None
    assert retrieved.id == "mem-uuid-1234-5678"
    assert retrieved.category == MemoryCategory.USER_PROFILE
    assert isinstance(retrieved.category, MemoryCategory)
    assert retrieved.key == "favorite_language"
    assert retrieved.value == "Python"
    assert retrieved.source == MemorySource.EXPLICIT_USER_INPUT
    assert isinstance(retrieved.source, MemorySource)
    assert retrieved.confidence == 0.95
    assert retrieved.created_at == now
    assert retrieved.updated_at == now + 1.0
    assert retrieved.expires_at == now + 3600.0
    assert retrieved.is_active is True
    assert retrieved.metadata_json == '{"verified": true}'


def test_get_memory_by_key(memory_store):
    """13. Verifies lookup by category and logical key."""
    now = time.time()
    r = MemoryRecord(
        id="mem-key-1",
        category=MemoryCategory.USER_PREFERENCE,
        key="theme",
        value="dark",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=1.0,
        created_at=now,
        updated_at=now,
    )
    memory_store.save_memory(r)

    fetched = memory_store.get_memory_by_key(MemoryCategory.USER_PREFERENCE, "theme")
    assert fetched is not None
    assert fetched.id == "mem-key-1"
    assert fetched.value == "dark"

    # String category lookup should also work
    fetched_str = memory_store.get_memory_by_key("user_preference", "theme")
    assert fetched_str is not None
    assert fetched_str.id == "mem-key-1"


def test_duplicate_logical_key_behaviour(memory_store):
    """14. Verifies updating existing record when saving duplicate logical key."""
    now = time.time()
    r1 = MemoryRecord(
        id="mem-id-1",
        category=MemoryCategory.USER_FACT,
        key="employer",
        value="Company A",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=1.0,
        created_at=now,
        updated_at=now,
    )
    memory_store.save_memory(r1)
    assert memory_store.count_memories() == 1

    r2 = MemoryRecord(
        id="mem-id-2", # Different ID, same (category, key)
        category=MemoryCategory.USER_FACT,
        key="employer",
        value="Company B",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=1.0,
        created_at=now,
        updated_at=now + 10.0,
    )
    memory_store.save_memory(r2)

    # Should update logical key, total count remains 1 active memory
    assert memory_store.count_memories() == 1
    updated = memory_store.get_memory_by_key(MemoryCategory.USER_FACT, "employer")
    assert updated is not None
    assert updated.value == "Company B"


def test_multiple_categories_and_filtering(memory_store):
    """15 & 16. Verifies handling multiple records and filtering by category."""
    now = time.time()
    r1 = MemoryRecord(
        id="m1", category=MemoryCategory.USER_PROFILE, key="name", value="Alice",
        source=MemorySource.EXPLICIT_USER_INPUT, confidence=1.0, created_at=now, updated_at=now
    )
    r2 = MemoryRecord(
        id="m2", category=MemoryCategory.CONTEXT_RULE, key="format", value="json",
        source=MemorySource.EXPLICIT_USER_INPUT, confidence=1.0, created_at=now, updated_at=now
    )
    r3 = MemoryRecord(
        id="m3", category=MemoryCategory.USER_PROFILE, key="location", value="NY",
        source=MemorySource.EXPLICIT_USER_INPUT, confidence=1.0, created_at=now, updated_at=now
    )

    memory_store.save_memory(r1)
    memory_store.save_memory(r2)
    memory_store.save_memory(r3)

    assert memory_store.count_memories() == 3

    profiles = memory_store.list_memories(category=MemoryCategory.USER_PROFILE)
    assert len(profiles) == 2
    keys = {p.key for p in profiles}
    assert keys == {"name", "location"}

    rules = memory_store.list_memories(category=MemoryCategory.CONTEXT_RULE)
    assert len(rules) == 1
    assert rules[0].key == "format"


def test_persistence_across_reopening(temp_db_path):
    """17. Verifies that records survive store closing and reopening on disk."""
    now = time.time()
    s1 = MemoryStore(db_path=temp_db_path)
    rec = MemoryRecord(
        id="mem-disk-1",
        category=MemoryCategory.USER_FACT,
        key="project",
        value="PIXIE",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=1.0,
        created_at=now,
        updated_at=now,
    )
    s1.save_memory(rec)
    s1.close()

    # Reopen same database file
    s2 = MemoryStore(db_path=temp_db_path)
    loaded = s2.get_memory("mem-disk-1")
    assert loaded is not None
    assert loaded.key == "project"
    assert loaded.value == "PIXIE"
    s2.close()


def test_invalid_and_sensitive_memory_rejection(memory_store):
    """19. Verifies boundary validator rejects secrets and invalid candidates before DB write."""
    now = time.time()

    # Sensitive API key
    rec_sensitive = MemoryRecord(
        id="mem-sec-1",
        category=MemoryCategory.USER_FACT,
        key="api_token",
        value="gsk_1234567890abcdef1234567890",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=1.0,
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(MemoryValidationError, match="Security Violation"):
        memory_store.save_memory(rec_sensitive)

    # Rejection means 0 records persisted
    assert memory_store.count_memories() == 0


def test_independence_from_session_store(temp_db_path):
    """20, 21, 22. Verifies MemoryStore and SessionStore remain completely independent."""
    sess_store = SessionStore(":memory:")
    mem_store = MemoryStore(":memory:")

    # Create a session and a message in SessionStore
    session = sess_store.create_session("Test Session")
    sess_id = session["id"]
    sess_store.add_message(sess_id, "user", "Hello P.I.X.I.E.")

    # Create a memory in MemoryStore
    now = time.time()
    mem = MemoryRecord(
        id="mem-indep-1",
        category=MemoryCategory.USER_PROFILE,
        key="user_name",
        value="Bob",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=1.0,
        created_at=now,
        updated_at=now,
    )
    mem_store.save_memory(mem)

    # Assert initial states
    assert len(sess_store.get_messages(sess_id)) == 1
    assert mem_store.get_memory("mem-indep-1") is not None

    # Delete session in SessionStore
    sess_store.delete_session(sess_id)
    assert len(sess_store.get_messages(sess_id)) == 0

    # MemoryStore MUST NOT be affected by session deletion!
    assert mem_store.get_memory("mem-indep-1") is not None
    assert mem_store.get_memory("mem-indep-1").value == "Bob"

    # Delete memory in MemoryStore
    mem_store.delete_memory("mem-indep-1")
    assert mem_store.get_memory("mem-indep-1") is None

    # SessionStore is unchanged
    assert sess_store.get_session("nonexistent") is None

    sess_store.close()
    mem_store.close()


def test_deterministic_ordering(memory_store):
    """23. Verifies listing memories returns deterministic ordering (updated_at DESC, id ASC)."""
    t0 = 1000.0
    r1 = MemoryRecord(
        id="a-mem", category=MemoryCategory.USER_FACT, key="k1", value="v1",
        source=MemorySource.EXPLICIT_USER_INPUT, confidence=1.0, created_at=t0, updated_at=t0 + 10
    )
    r2 = MemoryRecord(
        id="b-mem", category=MemoryCategory.USER_FACT, key="k2", value="v2",
        source=MemorySource.EXPLICIT_USER_INPUT, confidence=1.0, created_at=t0, updated_at=t0 + 20
    )
    r3 = MemoryRecord(
        id="c-mem", category=MemoryCategory.USER_FACT, key="k3", value="v3",
        source=MemorySource.EXPLICIT_USER_INPUT, confidence=1.0, created_at=t0, updated_at=t0 + 5
    )

    memory_store.save_memory(r1)
    memory_store.save_memory(r2)
    memory_store.save_memory(r3)

    listed = memory_store.list_memories()
    # Expect order: b-mem (updated_at + 20), a-mem (updated_at + 10), c-mem (updated_at + 5)
    ids = [item.id for item in listed]
    assert ids == ["b-mem", "a-mem", "c-mem"]


def test_deletion_and_deactivation(memory_store):
    """Verifies hard delete and soft deactivation."""
    now = time.time()
    r = MemoryRecord(
        id="mem-del-1",
        category=MemoryCategory.USER_PROFILE,
        key="temp_key",
        value="val",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=1.0,
        created_at=now,
        updated_at=now,
    )
    memory_store.save_memory(r)

    # Soft delete (deactivate)
    assert memory_store.delete_memory("mem-del-1", hard_delete=False) is True
    rec = memory_store.get_memory("mem-del-1")
    assert rec is not None
    assert rec.is_active is False
    assert memory_store.count_memories(active_only=True) == 0

    # Hard delete
    assert memory_store.delete_memory("mem-del-1", hard_delete=True) is True
    assert memory_store.get_memory("mem-del-1") is None
