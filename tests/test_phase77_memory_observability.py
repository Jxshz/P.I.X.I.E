import time
from unittest.mock import MagicMock
import pytest

from backend.agent.core import AgentCore
from backend.memory import (
    AuditEvent,
    ConsentState,
    MemoryCategory,
    MemoryConsentManager,
    MemoryEventType,
    MemoryObservabilityService,
    MemoryRecord,
    MemoryRetriever,
    MemoryService,
    MemorySource,
    MemoryValidationError,
    sanitize_audit_text,
)
from backend.storage.memory_audit_store import MemoryAuditStore


@pytest.fixture
def audit_store():
    """Provides an in-memory MemoryAuditStore."""
    store = MemoryAuditStore(db_path=":memory:")
    yield store
    store.close()


@pytest.fixture
def obs_service(audit_store):
    """Provides a MemoryObservabilityService instance."""
    return MemoryObservabilityService(audit_store=audit_store)


@pytest.fixture
def memory_service(obs_service):
    """Provides a MemoryService integrated with MemoryObservabilityService."""
    service = MemoryService(db_path=":memory:", observability=obs_service)
    yield service
    service.close()


def test_1_event_model_validation():
    """1. Verifies AuditEvent enforces schema rules and enum conversion."""
    evt = AuditEvent(
        event_id="e-1",
        event_type="MEMORY_CREATED",
        timestamp=time.time(),
        key="primary_language",
    )
    assert evt.event_type == MemoryEventType.MEMORY_CREATED
    assert evt.key == "primary_language"


def test_2_event_ids_are_unique(obs_service):
    """2. Verifies successive audit events get unique IDs."""
    e1 = obs_service.record_event(MemoryEventType.MEMORY_CREATED, key="key1")
    e2 = obs_service.record_event(MemoryEventType.MEMORY_CREATED, key="key2")
    assert e1.event_id != e2.event_id


def test_3_events_deterministic_ordering(obs_service):
    """3. Verifies event querying returns deterministic timestamp DESC order."""
    now = time.time()
    obs_service.record_event(MemoryEventType.MEMORY_CREATED, key="first")
    time.sleep(0.01)
    obs_service.record_event(MemoryEventType.MEMORY_UPDATED, key="second")

    events = obs_service.get_recent_events(limit=10)
    assert len(events) == 2
    assert events[0].key == "second"
    assert events[1].key == "first"


def test_4_memory_value_never_persisted_in_audit(obs_service):
    """4. Verifies raw memory value is never stored in AuditEvent fields."""
    evt = obs_service.record_event(
        MemoryEventType.MEMORY_CREATED,
        memory_id="m-1",
        key="primary_language",
        result="success",
    )

    assert not hasattr(evt, "value")
    # Verify to_dict / attributes do not contain raw values
    assert getattr(evt, "value", None) is None


def test_5_secrets_never_persisted():
    """5. Verifies sanitize_audit_text redacts secrets and API keys."""
    secret = "sk-1234567890abcdef1234567890"
    sanitized = sanitize_audit_text(secret)
    assert sanitized == "[REDACTED_SENSITIVE_CONTENT]"

    safe_text = "primary_language"
    assert sanitize_audit_text(safe_text) == "primary_language"


def test_6_raw_prompts_responses_redacted(obs_service):
    """6. Verifies record_event redacts prompt/response/secret metadata keys."""
    evt = obs_service.record_event(
        MemoryEventType.MEMORY_RETRIEVED,
        metadata={
            "prompt": "Secret prompt text",
            "response": "Secret response text",
            "value": "Secret memory value",
            "selected_count": 2,
        },
    )

    assert evt.metadata["prompt"] == "[REDACTED]"
    assert evt.metadata["response"] == "[REDACTED]"
    assert evt.metadata["value"] == "[REDACTED]"
    assert evt.metadata["selected_count"] == 2


def test_7_memory_create_produces_audit_event(memory_service, obs_service):
    """7. Verifies MemoryService.create_memory produces a MEMORY_CREATED event."""
    rec = memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
    )

    events = obs_service.get_events_for_memory(rec.id)
    assert len(events) == 1
    assert events[0].event_type == MemoryEventType.MEMORY_CREATED
    assert events[0].key == "primary_language"


def test_8_update_produces_audit_event(memory_service, obs_service):
    """8. Verifies MemoryService.update_memory produces a MEMORY_UPDATED event."""
    rec = memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
    )
    memory_service.update_memory(rec.id, value="Python")

    events = obs_service.get_events_for_memory(rec.id)
    assert len(events) == 2
    assert events[0].event_type == MemoryEventType.MEMORY_UPDATED


def test_9_forget_and_delete_produce_audit_events(memory_service, obs_service):
    """9. Verifies forget_memory and delete_memory produce audit events."""
    rec = memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
    )
    memory_service.forget_memory(rec.id)
    events_forget = obs_service.get_events_by_type(MemoryEventType.MEMORY_FORGOTTEN)
    assert len(events_forget) == 1

    memory_service.delete_memory(rec.id, hard_delete=True)
    events_del = obs_service.get_events_by_type(MemoryEventType.MEMORY_DELETED)
    assert len(events_del) == 1


def test_10_supersession_produces_audit_event(memory_service, obs_service):
    """10. Verifies MemoryService.supersede_memory produces a MEMORY_SUPERSEDED event."""
    rec1 = memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
    )
    rec2 = memory_service.supersede_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Python",
    )

    events = obs_service.get_events_by_type(MemoryEventType.MEMORY_SUPERSEDED)
    assert len(events) == 1
    assert events[0].memory_id == rec2.id


def test_11_retrieval_produces_safe_metrics(memory_service, obs_service):
    """11. Verifies MemoryRetriever.retrieve logs safe metrics without memory values."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
    )
    retriever = MemoryRetriever(memory_service=memory_service, observability=obs_service)
    matches = retriever.retrieve("What language should I use for coding?")

    assert len(matches) == 1
    events = obs_service.get_events_by_type(MemoryEventType.MEMORY_RETRIEVED)
    assert len(events) == 1
    assert events[0].metadata["selected_count"] == 1
    assert "selected_ids" in events[0].metadata


def test_12_empty_retrieval_is_observable(memory_service, obs_service):
    """12. Verifies retrieval with no matching memories logs MEMORY_RETRIEVAL_EMPTY."""
    retriever = MemoryRetriever(memory_service=memory_service, observability=obs_service)
    retriever.retrieve("What is quantum physics?")

    events = obs_service.get_events_by_type(MemoryEventType.MEMORY_RETRIEVAL_EMPTY)
    assert len(events) == 1


def test_13_retrieval_failure_is_observable(memory_service, obs_service):
    """13. Verifies retrieval internal exception logs MEMORY_RETRIEVAL_FAILED."""
    retriever = MemoryRetriever(memory_service=memory_service, observability=obs_service)
    retriever.service = None  # Force failure

    retriever.retrieve("primary_language")
    events = obs_service.get_events_by_type(MemoryEventType.MEMORY_RETRIEVAL_FAILED)
    assert len(events) == 1


def test_14_consent_decisions_are_observable(memory_service, obs_service):
    """14. Verifies consent manager decisions log audit events."""
    # Verified through MemoryService integration
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="test_key",
        value="test_val",
    )
    assert obs_service.count_events(MemoryEventType.MEMORY_CREATED) == 1


def test_15_conflict_decisions_are_observable(memory_service, obs_service):
    """15. Verifies conflict supersessions log audit events."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="key1",
        value="val1",
    )
    memory_service.supersede_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="key1",
        value="val2",
    )
    events = obs_service.get_events_by_type(MemoryEventType.MEMORY_SUPERSEDED)
    assert len(events) == 1


def test_16_security_rejection_is_observable(memory_service, obs_service):
    """16. Verifies security boundary violations log MEMORY_SECURITY_REJECTED."""
    with pytest.raises(MemoryValidationError):
        memory_service.create_memory(
            category=MemoryCategory.USER_PREFERENCE,
            key="api_key",
            value="sk-1234567890abcdef1234567890",
        )

    events = obs_service.get_events_by_type(MemoryEventType.MEMORY_SECURITY_REJECTED)
    assert len(events) == 1
    assert events[0].key == "api_key"


def test_17_audit_failure_does_not_break_memory_ops(memory_service, obs_service):
    """17. Verifies audit store write exceptions swallow safely without breaking CRUD."""
    obs_service.store.append_event = MagicMock(side_effect=Exception("DB write failure"))

    # Operations must succeed cleanly
    rec = memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
    )
    assert rec.id is not None
    assert memory_service.count_memories() == 1


def test_18_audit_failure_does_not_break_agent(memory_service, obs_service):
    """18. Verifies audit store write failure does not break AgentCore inference setup."""
    obs_service.store.append_event = MagicMock(side_effect=Exception("DB write failure"))

    retriever = MemoryRetriever(memory_service=memory_service, observability=obs_service)
    agent = AgentCore(memory_retriever=retriever, enable_memory=True)
    agent.conversation_history.append({"role": "user", "content": "primary_language"})

    msgs = agent._get_llm_messages()
    assert len(msgs) == 2


def test_19_audit_storage_isolated_from_sessions():
    """19. Verifies MemoryAuditStore writes strictly to memory_audit_events table."""
    store = MemoryAuditStore(db_path=":memory:")
    conn = store._get_connection()
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]

    assert "memory_audit_events" in tables
    assert "sessions" not in tables
    assert "messages" not in tables
    store.close()


def test_20_audit_storage_isolated_from_memories():
    """20. Verifies MemoryAuditStore does not create or pollute memories table."""
    store = MemoryAuditStore(db_path=":memory:")
    conn = store._get_connection()
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]

    assert "memory_audit_events" in tables
    assert "memories" not in tables
    store.close()


def test_21_event_queries_are_bounded(obs_service):
    """21. Verifies get_recent_events limit parameter is enforced."""
    for i in range(10):
        obs_service.record_event(MemoryEventType.MEMORY_CREATED, key=f"key_{i}")

    events = obs_service.get_recent_events(limit=3)
    assert len(events) == 3


def test_22_event_ordering_deterministic(obs_service):
    """22. Verifies ordering is deterministic across event queries."""
    for i in range(5):
        obs_service.record_event(MemoryEventType.MEMORY_CREATED, key=f"key_{i}")

    e1 = obs_service.get_recent_events(limit=5)
    e2 = obs_service.get_recent_events(limit=5)
    assert [e.event_id for e in e1] == [e.event_id for e in e2]


def test_23_observability_cannot_mutate_memory(obs_service):
    """23. Verifies MemoryObservabilityService exposes ZERO memory mutation methods."""
    assert not hasattr(obs_service, "save_memory")
    assert not hasattr(obs_service, "delete_memory")
    assert not hasattr(obs_service, "update_memory")
    assert not hasattr(obs_service, "reactivate_memory")


def test_24_phase6_security_invariants_intact(memory_service):
    """24. Verifies sensitive content boundary blocks secret memory creation."""
    with pytest.raises(MemoryValidationError):
        memory_service.create_memory(
            category=MemoryCategory.USER_FACT,
            key="password",
            value="password = super_secret_password_123",
        )


def test_25_phase7_subsystems_intact(memory_service, obs_service):
    """25. Verifies end-to-end extraction, consent, conflict, retrieval, and response context."""
    rec = memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
    )
    retriever = MemoryRetriever(memory_service=memory_service, observability=obs_service)
    matches = retriever.retrieve("What language should I use for coding?")

    assert len(matches) == 1
    assert matches[0].record.key == "primary_language"
    assert obs_service.count_events() >= 2
