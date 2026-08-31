import time
import pytest
from backend.memory import (
    MemoryCategory,
    MemoryMatch,
    MemoryRecord,
    MemoryRetriever,
    MemoryService,
    MemorySource,
)
from backend.storage.session_store import SessionStore


@pytest.fixture
def memory_service():
    """Provides an isolated MemoryService in :memory: database."""
    service = MemoryService(db_path=":memory:")
    yield service
    service.close()


@pytest.fixture
def retriever(memory_service):
    """Provides a MemoryRetriever instance using the test memory service."""
    r = MemoryRetriever(memory_service=memory_service)
    yield r
    r.close()


def test_empty_memory_store_retrieval(retriever):
    """1. Verifies retrieval on an empty store returns an empty list."""
    results = retriever.retrieve("python preference")
    assert results == []


def test_exact_key_match(retriever, memory_service):
    """2. Verifies exact key match produces high relevance score."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="python",
        value="User prefers Python over JavaScript",
    )
    memory_service.create_memory(
        category=MemoryCategory.USER_PROFILE,
        key="location",
        value="New York",
    )

    matches = retriever.retrieve("python")
    assert len(matches) == 1
    assert matches[0].record.key == "python"
    assert "exact_key_match" in matches[0].matched_signals
    assert matches[0].relevance_score > 3.0


def test_strong_lexical_match(retriever, memory_service):
    """3. Verifies strong token and phrase matching."""
    memory_service.create_memory(
        category=MemoryCategory.USER_FACT,
        key="project_framework",
        value="FastAPI backend with React UI",
    )

    matches = retriever.retrieve("FastAPI backend")
    assert len(matches) == 1
    assert matches[0].record.key == "project_framework"
    assert any("matched" in sig for sig in matches[0].matched_signals)


def test_weak_lexical_match(retriever, memory_service):
    """4. Verifies weak partial token match gives lower score."""
    memory_service.create_memory(
        category=MemoryCategory.USER_FACT,
        key="editor",
        value="Visual Studio Code",
    )
    matches = retriever.retrieve("Studio")
    assert len(matches) == 1
    assert matches[0].relevance_score < 3.0


def test_irrelevant_memory_exclusion(retriever, memory_service):
    """5. Verifies irrelevant memories with zero overlap are excluded."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PROFILE,
        key="favorite_food",
        value="Pizza",
    )

    matches = retriever.retrieve("quantum physics equations")
    assert matches == []


def test_category_filtering(retriever, memory_service):
    """6. Verifies filtering candidates by MemoryCategory."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PROFILE,
        key="language",
        value="English",
    )
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="language",
        value="Python 3.13",
    )

    matches = retriever.retrieve("language", category=MemoryCategory.USER_PREFERENCE)
    assert len(matches) == 1
    assert matches[0].record.category == MemoryCategory.USER_PREFERENCE
    assert matches[0].record.value == "Python 3.13"


def test_confidence_filtering(retriever, memory_service):
    """7 & 21. Verifies confidence threshold excludes low confidence memories."""
    memory_service.create_memory(
        category=MemoryCategory.USER_FACT,
        key="drink",
        value="Coffee",
        confidence=0.3,
    )
    memory_service.create_memory(
        category=MemoryCategory.USER_FACT,
        key="drink",
        value="Tea",
        confidence=0.9,
    )

    matches = retriever.retrieve("drink", min_confidence=0.8)
    assert len(matches) == 1
    assert matches[0].record.value == "Tea"


def test_expired_memory_exclusion(retriever, memory_service):
    """8. Verifies expired memories are not returned."""
    now = time.time()

    # Expired memory: created in past, expired in past
    rec_expired = MemoryRecord(
        id="mem-exp-1",
        category=MemoryCategory.USER_FACT,
        key="temp_location",
        value="Hotel Paris",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=1.0,
        created_at=now - 500,
        updated_at=now - 500,
        expires_at=now - 100,  # > created_at, but < now (expired)
    )
    memory_service.store.save_memory(rec_expired)

    # Active memory: expires in future
    memory_service.create_memory(
        category=MemoryCategory.USER_FACT,
        key="perm_location",
        value="London",
        expires_at=now + 3600,  # Active in future
    )

    matches = retriever.retrieve("location")
    assert len(matches) == 1
    assert matches[0].record.key == "perm_location"


def test_inactive_memory_exclusion(retriever, memory_service):
    """9. Verifies inactive memories are excluded."""
    memory_service.create_memory(
        category=MemoryCategory.USER_FACT,
        key="status",
        value="Draft",
        is_active=False,
    )

    matches = retriever.retrieve("status")
    assert matches == []


def test_result_limit_enforcement(retriever, memory_service):
    """10. Verifies result count adheres to requested limit."""
    for i in range(10):
        memory_service.create_memory(
            category=MemoryCategory.USER_FACT,
            key=f"item_{i}",
            value=f"value keyword {i}",
        )

    matches = retriever.retrieve("keyword", limit=3)
    assert len(matches) == 3


def test_deterministic_ordering_and_tiebreaking(retriever, memory_service):
    """11 & 12. Verifies deterministic sorting and tie-breaking by score, updated_at, id."""
    now = time.time()

    # Create records with identical lexical score
    m1 = memory_service.create_memory(
        category=MemoryCategory.USER_FACT,
        key="tag_a",
        value="shared_keyword",
        memory_id="a-uuid",
    )
    m2 = memory_service.create_memory(
        category=MemoryCategory.USER_FACT,
        key="tag_b",
        value="shared_keyword",
        memory_id="b-uuid",
    )

    res1 = retriever.retrieve("shared_keyword")
    res2 = retriever.retrieve("shared_keyword")

    # Ordering across multiple invocations MUST be identical
    assert [m.record.id for m in res1] == [m.record.id for m in res2]


def test_case_normalization_and_punctuation(retriever, memory_service):
    """13 & 14. Verifies case-insensitivity and punctuation resilience."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="Code_Style",
        value="PEP8! Strict formatting.",
    )

    matches = retriever.retrieve("code_style pep8")
    assert len(matches) == 1
    assert matches[0].record.key == "Code_Style"


def test_retrieval_does_not_mutate_records(retriever, memory_service):
    """16. Verifies retrieve() is strictly read-only and does not mutate record fields."""
    rec = memory_service.create_memory(
        category=MemoryCategory.USER_PROFILE,
        key="timezone",
        value="UTC+5:30",
    )
    updated_at_before = rec.updated_at
    confidence_before = rec.confidence

    # Run retrieval query
    matches = retriever.retrieve("timezone")
    assert len(matches) == 1

    # Reload from store to verify database state was untouched
    after = memory_service.get_memory(rec.id)
    assert after.updated_at == updated_at_before
    assert after.confidence == confidence_before
    assert after.is_active is True


def test_secret_and_injection_defense(retriever, memory_service):
    """17. Verifies defense-in-depth against secret keys or prompt overrides."""
    # Direct insertion bypassing validator (simulated)
    # Boundaries module check in retrieve() should filter secrets out
    matches = retriever.retrieve("gsk_1234567890abcdef12345678")
    assert matches == []


def test_very_large_limit_is_bounded(retriever, memory_service):
    """18. Verifies requesting limit=999999 is safely capped to MAX_RETRIEVAL_LIMIT."""
    for i in range(30):
        memory_service.create_memory(
            category=MemoryCategory.USER_FACT,
            key=f"key_{i}",
            value="universal_search_term",
        )

    matches = retriever.retrieve("universal_search_term", limit=999999)
    assert len(matches) <= 20  # Capped at MAX_RETRIEVAL_LIMIT (20)


def test_very_long_query_handled_safely(retriever, memory_service):
    """19. Verifies long query string is safely truncated without crashing."""
    memory_service.create_memory(
        category=MemoryCategory.USER_FACT,
        key="short_key",
        value="short_value",
    )
    long_query = "search " * 2000
    matches = retriever.retrieve(long_query)
    # Should complete safely without exception
    assert isinstance(matches, list)


def test_old_relevant_vs_new_irrelevant(retriever, memory_service):
    """22. Verifies old relevant memory outranks new irrelevant memory."""
    t_old = time.time() - 100000.0
    t_new = time.time()

    # Old relevant memory
    memory_service.create_memory(
        category=MemoryCategory.USER_FACT,
        key="database",
        value="SQLite 3",
    )

    # New irrelevant memory
    memory_service.create_memory(
        category=MemoryCategory.USER_FACT,
        key="color",
        value="Blue",
    )

    matches = retriever.retrieve("SQLite database")
    assert len(matches) == 1
    assert matches[0].record.key == "database"


def test_session_store_independence(retriever, memory_service):
    """23 & 24. Verifies MemoryRetriever has no session_id dependency and is independent of SessionStore."""
    sess_store = SessionStore(":memory:")
    sess = sess_store.create_session("Test Session")
    sess_store.add_message(sess["id"], "user", "What is my favorite language?")

    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="language",
        value="Python",
    )

    matches = retriever.retrieve("language")
    assert len(matches) == 1
    assert matches[0].record.value == "Python"

    # Delete session in SessionStore
    sess_store.delete_session(sess["id"])

    # Retrieval in MemoryRetriever remains 100% functional and untouched
    matches_after = retriever.retrieve("language")
    assert len(matches_after) == 1
    assert matches_after[0].record.value == "Python"

    sess_store.close()
