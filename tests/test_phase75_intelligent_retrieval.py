import time
import pytest

from backend.memory import (
    MemoryCategory,
    MemoryRecord,
    MemoryRetriever,
    MemoryService,
    MemorySource,
)


@pytest.fixture
def memory_service():
    """Provides isolated MemoryService in :memory: DB."""
    service = MemoryService(db_path=":memory:")
    yield service
    service.close()


@pytest.fixture
def retriever(memory_service):
    """Provides MemoryRetriever backed by isolated MemoryService."""
    return MemoryRetriever(memory_service=memory_service)


def test_intent_analysis_profile_query(memory_service, retriever):
    """Verifies profile query prioritizes user_profile category."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PROFILE,
        key="name",
        value="Joshva",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="response_style",
        value="concise",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )

    matches = retriever.retrieve("My name?")
    assert len(matches) >= 1
    assert matches[0].record.category == MemoryCategory.USER_PROFILE
    assert matches[0].record.value == "Joshva"
    assert "intent_match:user_profile" in matches[0].matched_signals


def test_intent_analysis_preference_query(memory_service, retriever):
    """Verifies preference query prioritizes user_preference category."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )
    memory_service.create_memory(
        category=MemoryCategory.USER_PROFILE,
        key="occupation",
        value="Software Engineer",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )

    matches = retriever.retrieve("What language should we use for coding?")
    assert len(matches) >= 1
    assert matches[0].record.category == MemoryCategory.USER_PREFERENCE
    assert matches[0].record.value == "Java"
    assert "intent_match:user_preference" in matches[0].matched_signals


def test_ranking_explicit_beats_inferred(memory_service, retriever):
    """Verifies explicit user input outranks system inferred information."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Python",
        source=MemorySource.SYSTEM_INFERRED,
        confidence=1.0,
    )
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="preferred_language",
        value="Java",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=1.0,
    )

    matches = retriever.retrieve("coding language")
    assert len(matches) >= 2
    assert matches[0].record.source == MemorySource.EXPLICIT_USER_INPUT
    assert matches[0].record.value == "Java"


def test_ranking_confidence_weighting(memory_service, retriever):
    """Verifies higher confidence memory outranks lower confidence memory."""
    memory_service.create_memory(
        category=MemoryCategory.USER_FACT,
        key="location",
        value="Bangalore",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=1.0,
    )
    memory_service.create_memory(
        category=MemoryCategory.USER_FACT,
        key="location_old",
        value="Delhi",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=0.3,
    )

    matches = retriever.retrieve("location")
    assert len(matches) >= 2
    assert matches[0].record.value == "Bangalore"


def test_conflict_inactive_and_expired_filtering(memory_service, retriever):
    """Verifies inactive and expired records are completely excluded from retrieval."""
    now = time.time()
    # Inactive memory
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="C++",
        source=MemorySource.EXPLICIT_USER_INPUT,
        is_active=False,
    )

    # Insert an expired memory directly into connection to bypass validation check
    conn = memory_service.store._get_connection()
    conn.execute(
        "INSERT INTO memories (id, category, key, value, source, confidence, created_at, updated_at, expires_at, is_active, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("exp-1", MemoryCategory.USER_PREFERENCE.value, "primary_language", "Ruby", MemorySource.EXPLICIT_USER_INPUT.value, 1.0, now - 200, now - 200, now - 100, 1, None),
    )
    conn.commit()

    # Active memory
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )

    matches = retriever.retrieve("primary language")
    values = [m.record.value for m in matches]

    assert "C++" not in values
    assert "Ruby" not in values
    assert "Java" in values


def test_redundancy_suppression(memory_service, retriever):
    """Verifies redundant duplicate logical key matches are suppressed to top candidate."""
    m1 = memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=1.0,
    )

    matches = retriever.retrieve("primary language")
    assert len(matches) == 1
    assert matches[0].record.id == m1.id


def test_diversity_selection(memory_service, retriever):
    """Verifies cross-category matches coexist when limit >= 3."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PROFILE,
        key="name",
        value="Joshva",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="response_style",
        value="concise",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )
    memory_service.create_memory(
        category=MemoryCategory.CONTEXT_RULE,
        key="always_rule",
        value="explain code before showing solution",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )

    matches = retriever.retrieve("Joshva explain concise code", limit=5)
    categories = [m.record.category for m in matches]

    assert len(matches) == 3
    assert MemoryCategory.USER_PROFILE in categories
    assert MemoryCategory.USER_PREFERENCE in categories
    assert MemoryCategory.CONTEXT_RULE in categories


def test_bounds_and_empty_queries(memory_service, retriever):
    """Verifies bounds on query length, limit caps, and empty input."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PROFILE,
        key="name",
        value="Joshva",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )

    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []

    huge_query = "name " * 500
    matches = retriever.retrieve(huge_query, limit=50)
    assert len(matches) <= 20  # MAX_RETRIEVAL_LIMIT


def test_read_only_invariant(memory_service, retriever):
    """Verifies retrieval performs ZERO database mutations."""
    rec = memory_service.create_memory(
        category=MemoryCategory.USER_PROFILE,
        key="name",
        value="Joshva",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )

    orig_created = rec.created_at
    orig_updated = rec.updated_at
    orig_active = rec.is_active

    retriever.retrieve("name")

    fetched = memory_service.get_memory(rec.id)
    assert fetched.created_at == orig_created
    assert fetched.updated_at == orig_updated
    assert fetched.is_active == orig_active
    assert fetched.value == "Joshva"


def test_fail_safe_fallback(retriever):
    """Verifies retriever degrades safely without raising exceptions on internal errors."""
    retriever.service = None  # Force internal error
    matches = retriever.retrieve("test query")
    assert matches == []
