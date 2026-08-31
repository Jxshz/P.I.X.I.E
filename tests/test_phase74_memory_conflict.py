import time
import pytest

from backend.memory import (
    ConflictResolutionOutcome,
    MemoryCandidate,
    MemoryCandidateExtractor,
    MemoryCategory,
    MemoryConflictDecision,
    MemoryConflictResolver,
    MemoryPolicyDecision,
    MemoryPolicyOutcome,
    MemoryRecord,
    MemoryService,
    MemorySource,
)
from backend.storage.session_store import SessionStore


@pytest.fixture
def memory_service():
    """Provides isolated MemoryService in :memory: DB."""
    service = MemoryService(db_path=":memory:")
    yield service
    service.close()


@pytest.fixture
def conflict_resolver(memory_service):
    """Provides MemoryConflictResolver backed by MemoryService."""
    return MemoryConflictResolver(memory_service=memory_service)


def test_1_no_existing_memory(conflict_resolver):
    """1. Candidate with no existing active memory yields NO_CONFLICT."""
    cands = MemoryCandidateExtractor.extract("Use Java as my primary language.")
    assert len(cands) == 1

    decision = conflict_resolver.resolve(cands[0])
    assert decision.outcome == ConflictResolutionOutcome.NO_CONFLICT
    assert decision.conflicting_record is None


def test_2_identical_value(conflict_resolver, memory_service):
    """2. Candidate with identical value to active memory yields NO_CONFLICT."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )

    cands = MemoryCandidateExtractor.extract("Use Java as my primary language.")
    assert len(cands) == 1

    decision = conflict_resolver.resolve(cands[0])
    assert decision.outcome == ConflictResolutionOutcome.NO_CONFLICT
    assert decision.conflicting_record is not None
    assert decision.conflicting_record.value == "Java"


def test_3_same_key_contradiction(conflict_resolver, memory_service):
    """3. Candidate with new explicit value for existing key yields SUPERSEDE_EXISTING."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )

    cands = MemoryCandidateExtractor.extract("Use Python as my primary language.")
    assert len(cands) == 1

    decision = conflict_resolver.resolve(cands[0])
    assert decision.outcome == ConflictResolutionOutcome.SUPERSEDE_EXISTING
    assert decision.conflicting_record.value == "Java"


def test_4_explicit_to_explicit_update(conflict_resolver, memory_service):
    """4. Newer explicit candidate supersedes older explicit memory."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="response_style",
        value="detailed",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )

    cands = MemoryCandidateExtractor.extract("I prefer concise answers.")
    assert len(cands) == 1

    decision = conflict_resolver.resolve(cands[0])
    assert decision.outcome == ConflictResolutionOutcome.SUPERSEDE_EXISTING
    assert decision.conflicting_record.value == "detailed"


def test_5_explicit_to_inferred_candidate(conflict_resolver, memory_service):
    """5. System-inferred candidate trying to overwrite explicit user memory yields REJECT_CONFLICT."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )

    policy_dec = MemoryPolicyDecision(
        outcome=MemoryPolicyOutcome.ALLOW_CANDIDATE,
        category=MemoryCategory.USER_PREFERENCE,
        confidence=0.8,
        reason="Inferred",
    )
    cand_inferred = MemoryCandidate(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="C++",
        source=MemorySource.SYSTEM_INFERRED,
        confidence=0.8,
        evidence="Inferred from code snippets",
        policy_decision=policy_dec,
        extraction_reason="Inferred",
    )

    decision = conflict_resolver.resolve(cand_inferred)
    assert decision.outcome == ConflictResolutionOutcome.REJECT_CONFLICT
    assert "lower trust" in decision.reason.lower()


def test_6_inferred_to_explicit_candidate(conflict_resolver, memory_service):
    """6. Explicit user candidate supersedes existing system-inferred memory."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="C++",
        source=MemorySource.SYSTEM_INFERRED,
        confidence=0.6,
    )

    cands = MemoryCandidateExtractor.extract("Use Java as my primary language.")
    assert len(cands) == 1

    decision = conflict_resolver.resolve(cands[0])
    assert decision.outcome == ConflictResolutionOutcome.SUPERSEDE_EXISTING
    assert decision.provenance_override is True


def test_7_confidence_advantage(conflict_resolver, memory_service):
    """7. Higher confidence candidate supersedes lower confidence existing memory."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PROFILE,
        key="goal",
        value="2026 placements",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=0.5,
    )

    cands = MemoryCandidateExtractor.extract("I am preparing for 2027 placements.")
    assert len(cands) == 1
    cands[0].confidence = 0.95

    decision = conflict_resolver.resolve(cands[0])
    assert decision.outcome == ConflictResolutionOutcome.SUPERSEDE_EXISTING
    assert decision.confidence_delta > 0.15


def test_8_confidence_ambiguity(conflict_resolver, memory_service):
    """8. Candidate with significantly lower confidence yields REJECT_CONFLICT."""
    memory_service.create_memory(
        category=MemoryCategory.USER_FACT,
        key="user_fact",
        value="I live in Bangalore",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=0.95,
    )

    policy_dec = MemoryPolicyDecision(
        outcome=MemoryPolicyOutcome.ALLOW_CANDIDATE,
        category=MemoryCategory.USER_FACT,
        confidence=0.4,
        reason="Low confidence statement",
    )
    cand_low = MemoryCandidate(
        category=MemoryCategory.USER_FACT,
        key="user_fact",
        value="I live in Delhi",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=0.4,
        evidence="Vague mention",
        policy_decision=policy_dec,
        extraction_reason="Low confidence",
    )

    decision = conflict_resolver.resolve(cand_low)
    assert decision.outcome == ConflictResolutionOutcome.REJECT_CONFLICT
    assert decision.confidence_delta < -0.15


def test_9_recency_tie_break(conflict_resolver, memory_service):
    """9. Equal provenance & confidence candidate supersedes older memory via recency tie-break."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=0.9,
    )

    cands = MemoryCandidateExtractor.extract("Use Python as my primary language.")
    assert len(cands) == 1
    cands[0].confidence = 0.9

    decision = conflict_resolver.resolve(cands[0])
    assert decision.outcome == ConflictResolutionOutcome.SUPERSEDE_EXISTING


def test_10_multiple_existing_records(conflict_resolver):
    """10. Resolver targets active memory and ignores inactive superseded records."""
    inactive_rec = MemoryRecord(
        id="m-old",
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=1.0,
        created_at=time.time() - 100,
        updated_at=time.time() - 100,
        is_active=False,
    )
    active_rec = MemoryRecord(
        id="m-cur",
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Python",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=1.0,
        created_at=time.time() - 50,
        updated_at=time.time() - 50,
        is_active=True,
    )

    cands = MemoryCandidateExtractor.extract("Use C++ as my primary language.")
    decision = conflict_resolver.resolve(cands[0], existing_records=[inactive_rec, active_rec])

    assert decision.outcome == ConflictResolutionOutcome.SUPERSEDE_EXISTING
    assert decision.conflicting_record.id == active_rec.id
    assert decision.conflicting_record.value == "Python"


def test_11_inactive_existing_record(conflict_resolver, memory_service):
    """11. Inactive existing memory is ignored, yielding NO_CONFLICT."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
        source=MemorySource.EXPLICIT_USER_INPUT,
        is_active=False,
    )

    cands = MemoryCandidateExtractor.extract("Use Python as my primary language.")
    decision = conflict_resolver.resolve(cands[0])

    assert decision.outcome == ConflictResolutionOutcome.NO_CONFLICT
    assert decision.conflicting_record is None


def test_12_expired_existing_record(conflict_resolver):
    """12. Expired existing memory is ignored, yielding NO_CONFLICT."""
    now = time.time()
    expired_rec = MemoryRecord(
        id="m-exp",
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=1.0,
        created_at=now - 200,
        updated_at=now - 200,
        expires_at=now - 100,
        is_active=True,
    )

    cands = MemoryCandidateExtractor.extract("Use Python as my primary language.")
    decision = conflict_resolver.resolve(cands[0], existing_records=[expired_rec])

    assert decision.outcome == ConflictResolutionOutcome.NO_CONFLICT
    assert decision.conflicting_record is None


def test_13_rejected_candidate(conflict_resolver):
    """13. Policy rejected candidate yields REJECT_CONFLICT."""
    rej_decision = MemoryPolicyDecision(
        outcome=MemoryPolicyOutcome.REJECT,
        category=None,
        confidence=0.0,
        reason="Rejected by policy",
    )
    cand_rej = MemoryCandidate(
        category=MemoryCategory.USER_FACT,
        key="test",
        value="test",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=0.0,
        evidence="test",
        policy_decision=rej_decision,
        extraction_reason="Test",
    )

    decision = conflict_resolver.resolve(cand_rej)
    assert decision.outcome == ConflictResolutionOutcome.REJECT_CONFLICT


def test_14_confirmation_required_candidate(conflict_resolver, memory_service):
    """14. Candidate requiring confirmation yields REQUIRE_REVIEW."""
    cands = MemoryCandidateExtractor.extract("I might move to Bangalore next year.")
    assert len(cands) == 1

    decision = conflict_resolver.resolve(cands[0])
    assert decision.outcome == ConflictResolutionOutcome.REQUIRE_REVIEW


def test_15_approved_candidate(conflict_resolver, memory_service):
    """15. Approved explicit candidate with contradiction yields SUPERSEDE_EXISTING."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PROFILE,
        key="name",
        value="Bob",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )

    cands = MemoryCandidateExtractor.extract("My name is Joshva.")
    decision = conflict_resolver.resolve(cands[0])

    assert decision.outcome == ConflictResolutionOutcome.SUPERSEDE_EXISTING
    assert decision.conflicting_record.value == "Bob"


def test_16_security_rejection(conflict_resolver):
    """16. Candidate containing sensitive data yields REJECT_CONFLICT."""
    sec_decision = MemoryPolicyDecision(
        outcome=MemoryPolicyOutcome.ALLOW_CANDIDATE,
        category=MemoryCategory.USER_FACT,
        confidence=1.0,
        reason="Test",
    )
    sec_cand = MemoryCandidate(
        category=MemoryCategory.USER_FACT,
        key="secret",
        value="sk-1234567890abcdef1234567890",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=1.0,
        evidence="sk-1234567890abcdef1234567890",
        policy_decision=sec_decision,
        extraction_reason="Test",
    )

    decision = conflict_resolver.resolve(sec_cand)
    assert decision.outcome == ConflictResolutionOutcome.REJECT_CONFLICT
    assert "Security Violation" in decision.reason


def test_17_duplicate_prevention(conflict_resolver, memory_service):
    """17. apply_resolution on NO_CONFLICT does not write to database."""
    cands = MemoryCandidateExtractor.extract("My name is Joshva.")
    decision = conflict_resolver.resolve(cands[0])
    assert decision.outcome == ConflictResolutionOutcome.NO_CONFLICT

    saved = conflict_resolver.apply_resolution(decision)
    assert saved is None
    assert memory_service.count_memories() == 0


def test_18_supersession_correctness(conflict_resolver, memory_service):
    """18. apply_resolution on SUPERSEDE_EXISTING updates logical memory value in MemoryService."""
    old_mem = memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )

    cands = MemoryCandidateExtractor.extract("Use Python as my primary language.")
    decision = conflict_resolver.resolve(cands[0])
    assert decision.outcome == ConflictResolutionOutcome.SUPERSEDE_EXISTING

    new_mem = conflict_resolver.apply_resolution(decision)
    assert new_mem is not None
    assert new_mem.value == "Python"

    # Memory value is now updated to Python in MemoryService
    updated = memory_service.get_memory(old_mem.id)
    assert updated.value == "Python"

    # Only 1 active memory record exists for key
    active_mems = memory_service.list_memories(active_only=True)
    assert len(active_mems) == 1
    assert active_mems[0].value == "Python"


def test_19_unrelated_category_isolation(conflict_resolver, memory_service):
    """19. Candidate in USER_PREFERENCE does not conflict with USER_PROFILE for same key name."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PROFILE,
        key="language",
        value="English",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )

    policy_dec = MemoryPolicyDecision(
        outcome=MemoryPolicyOutcome.ALLOW_CANDIDATE,
        category=MemoryCategory.USER_PREFERENCE,
        confidence=0.95,
        reason="Test",
    )
    pref_cand = MemoryCandidate(
        category=MemoryCategory.USER_PREFERENCE,
        key="language",
        value="Python",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=0.95,
        evidence="Python",
        policy_decision=policy_dec,
        extraction_reason="Test",
    )

    decision = conflict_resolver.resolve(pref_cand)
    assert decision.outcome == ConflictResolutionOutcome.NO_CONFLICT


def test_20_deterministic_resolution(conflict_resolver, memory_service):
    """20. Identical inputs yield identical resolution decisions."""
    memory_service.create_memory(
        category=MemoryCategory.USER_PREFERENCE,
        key="primary_language",
        value="Java",
        source=MemorySource.EXPLICIT_USER_INPUT,
    )

    cands = MemoryCandidateExtractor.extract("Use Python as my primary language.")

    d1 = conflict_resolver.resolve(cands[0])
    d2 = conflict_resolver.resolve(cands[0])

    assert d1.outcome == d2.outcome
    assert d1.reason == d2.reason
    assert d1.conflicting_record.id == d2.conflicting_record.id


def test_system_isolation_during_conflict_analysis(conflict_resolver, memory_service):
    """Verifies resolve() performs ZERO database writes and leaves SessionStore untouched."""
    session_store = SessionStore(db_path=":memory:")
    sess = session_store.create_session("Test")
    sess_id = sess["id"]

    cands = MemoryCandidateExtractor.extract("My name is Joshva.")

    # Run resolve multiple times
    conflict_resolver.resolve(cands[0])
    conflict_resolver.resolve(cands[0])

    # MemoryStore and SessionStore remain completely untouched
    assert memory_service.count_memories() == 0
    assert len(session_store.get_messages(sess_id)) == 0
    session_store.close()
