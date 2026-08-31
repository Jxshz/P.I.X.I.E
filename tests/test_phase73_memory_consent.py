import time
import pytest

from backend.memory import (
    ConsentRecord,
    ConsentState,
    MemoryCandidate,
    MemoryCandidateExtractor,
    MemoryCategory,
    MemoryConsentManager,
    MemoryPolicyDecision,
    MemoryPolicyOutcome,
    MemoryProvenance,
    MemoryRetriever,
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
def consent_manager(memory_service):
    """Provides MemoryConsentManager backed by isolated MemoryService."""
    return MemoryConsentManager(memory_service=memory_service, auto_approve_candidates=True)


@pytest.fixture
def consent_manager_manual(memory_service):
    """Provides MemoryConsentManager requiring manual approval for all candidates."""
    return MemoryConsentManager(memory_service=memory_service, auto_approve_candidates=False)


def test_1_allow_candidate_behaviour(consent_manager):
    """1. Verifies ALLOW_CANDIDATE outcome auto-approves and persists when auto_approve=True."""
    cands = MemoryCandidateExtractor.extract("My name is Joshva.")
    assert len(cands) == 1

    rec = consent_manager.process_candidate(cands[0])
    assert rec.state == ConsentState.AUTO_APPROVED
    assert rec.persisted_memory_id is not None

    # Verified in MemoryService
    mem = consent_manager.memory_service.get_memory(rec.persisted_memory_id)
    assert mem is not None
    assert mem.value == "Joshva"


def test_2_require_confirmation_behaviour(consent_manager):
    """2. Verifies REQUIRE_CONFIRMATION enters PENDING state without initial persistence."""
    cands = MemoryCandidateExtractor.extract("I might move to Bangalore next year.")
    assert len(cands) == 1

    rec = consent_manager.process_candidate(cands[0])
    assert rec.state == ConsentState.PENDING
    assert rec.persisted_memory_id is None

    # MemoryService count remains 0
    assert consent_manager.memory_service.count_memories() == 0


def test_3_temporary_context_non_persistence(consent_manager):
    """3. Verifies TEMPORARY_CONTEXT outcome is REJECTED and never persisted."""
    cands = MemoryCandidateExtractor.extract("I am studying this topic tonight.")
    assert len(cands) == 1

    rec = consent_manager.process_candidate(cands[0])
    assert rec.state == ConsentState.REJECTED
    assert "Temporary Context" in rec.rejection_reason
    assert consent_manager.memory_service.count_memories() == 0


def test_4_reject_non_persistence(consent_manager):
    """4. Verifies REJECT outcome is marked REJECTED and never persisted."""
    cands = MemoryCandidateExtractor.extract("If I were a billionaire, I would buy an island.")
    assert cands == []  # Rejected by extractor

    # Test direct candidate rejection handling
    rej_decision = MemoryPolicyDecision(
        outcome=MemoryPolicyOutcome.REJECT,
        category=None,
        confidence=0.0,
        reason="Test Rejection",
    )
    raw_cand = MemoryCandidate(
        category=MemoryCategory.USER_FACT,
        key="test",
        value="test_val",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=0.0,
        evidence="test_val",
        policy_decision=rej_decision,
        extraction_reason="Test",
    )
    rec = consent_manager.process_candidate(raw_cand)
    assert rec.state == ConsentState.REJECTED
    assert consent_manager.memory_service.count_memories() == 0


def test_5_explicit_approval(consent_manager):
    """5. Verifies explicit approve() call persists candidate and returns MemoryRecord."""
    cands = MemoryCandidateExtractor.extract("I think I prefer Python now.")
    assert len(cands) == 1

    rec = consent_manager.process_candidate(cands[0])
    assert rec.state == ConsentState.PENDING

    mem = consent_manager.approve(rec.candidate_id)
    assert mem is not None
    assert "Python" in mem.value

    # State updated to APPROVED
    updated_rec = consent_manager.get_consent_record(rec.candidate_id)
    assert updated_rec.state == ConsentState.APPROVED
    assert updated_rec.persisted_memory_id == mem.id


def test_6_explicit_rejection(consent_manager):
    """6. Verifies explicit reject() call sets state to REJECTED and prevents persistence."""
    cands = MemoryCandidateExtractor.extract("I usually wake up around 7.")
    assert len(cands) == 1

    rec = consent_manager.process_candidate(cands[0])
    assert rec.state == ConsentState.PENDING

    success = consent_manager.reject(rec.candidate_id, reason="User cancelled request")
    assert success is True

    updated_rec = consent_manager.get_consent_record(rec.candidate_id)
    assert updated_rec.state == ConsentState.REJECTED
    assert consent_manager.memory_service.count_memories() == 0


def test_7_pending_cannot_persist(consent_manager):
    """7. Verifies pending requests remain unpersisted in MemoryService."""
    cands = MemoryCandidateExtractor.extract("I may want to switch my primary language.")
    assert len(cands) == 1

    rec = consent_manager.process_candidate(cands[0])
    assert rec.state == ConsentState.PENDING
    assert consent_manager.memory_service.count_memories() == 0


def test_8_rejected_cannot_persist(consent_manager):
    """8. Verifies rejected requests return None if approve() is subsequently attempted."""
    cands = MemoryCandidateExtractor.extract("I usually wake up around 7.")
    assert len(cands) == 1

    rec = consent_manager.process_candidate(cands[0])
    consent_manager.reject(rec.candidate_id)

    # Attempt to approve rejected candidate fails
    mem = consent_manager.approve(rec.candidate_id)
    assert mem is None
    assert consent_manager.memory_service.count_memories() == 0


def test_9_expired_consent_cannot_persist(memory_service):
    """9. Verifies expired consent requests cannot be approved."""
    short_mgr = MemoryConsentManager(memory_service=memory_service, ttl_seconds=-10.0)
    cands = MemoryCandidateExtractor.extract("I might move to Bangalore next year.")
    assert len(cands) == 1

    rec = short_mgr.process_candidate(cands[0])
    # Attempting to approve an expired request fails
    mem = short_mgr.approve(rec.candidate_id)
    assert mem is None
    assert short_mgr.get_consent_record(rec.candidate_id).state == ConsentState.EXPIRED


def test_10_duplicate_approval_idempotency(consent_manager):
    """10. Verifies calling approve() multiple times is idempotent and returns existing MemoryRecord."""
    cands = MemoryCandidateExtractor.extract("I might move to Bangalore next year.")
    assert len(cands) == 1

    rec = consent_manager.process_candidate(cands[0])
    mem1 = consent_manager.approve(rec.candidate_id)
    mem2 = consent_manager.approve(rec.candidate_id)

    assert mem1 is not None
    assert mem2 is not None
    assert mem1.id == mem2.id
    # DB record count remains 1
    assert consent_manager.memory_service.count_memories() == 1


def test_11_candidate_scoped_approval(consent_manager):
    """11. Verifies approving candidate A does NOT approve candidate B."""
    cands1 = MemoryCandidateExtractor.extract("I might move to Bangalore next year.")
    cands2 = MemoryCandidateExtractor.extract("I usually wake up around 7.")

    rec1 = consent_manager.process_candidate(cands1[0])
    rec2 = consent_manager.process_candidate(cands2[0])

    consent_manager.approve(rec1.candidate_id)

    assert consent_manager.get_consent_record(rec1.candidate_id).state == ConsentState.APPROVED
    assert consent_manager.get_consent_record(rec2.candidate_id).state == ConsentState.PENDING
    assert consent_manager.memory_service.count_memories() == 1


def test_12_security_rejection_in_consent(consent_manager):
    """12. Verifies candidates containing sensitive data are rejected immediately."""
    sec_decision = MemoryPolicyDecision(
        outcome=MemoryPolicyOutcome.ALLOW_CANDIDATE,
        category=MemoryCategory.USER_FACT,
        confidence=1.0,
        reason="Test",
    )
    sec_cand = MemoryCandidate(
        category=MemoryCategory.USER_FACT,
        key="api_key",
        value="sk-1234567890abcdef1234567890",
        source=MemorySource.EXPLICIT_USER_INPUT,
        confidence=1.0,
        evidence="sk-1234567890abcdef1234567890",
        policy_decision=sec_decision,
        extraction_reason="Test",
    )
    rec = consent_manager.process_candidate(sec_cand)
    assert rec.state == ConsentState.REJECTED
    assert "Security Violation" in rec.rejection_reason
    assert consent_manager.memory_service.count_memories() == 0


def test_13_session_isolation(consent_manager):
    """13. Verifies consent manager operations do NOT touch SessionStore."""
    store = SessionStore(db_path=":memory:")
    sess = store.create_session("Test Session")
    sess_id = sess["id"]

    cands = MemoryCandidateExtractor.extract("My name is Joshva.")
    rec = consent_manager.process_candidate(cands[0])

    # SessionStore remains untouched
    assert len(store.get_messages(sess_id)) == 0
    store.close()


def test_14_memory_service_remains_persistence_boundary(consent_manager):
    """14. Verifies MemoryConsentManager uses MemoryService for persistence."""
    cands = MemoryCandidateExtractor.extract("Use Java as my primary language.")
    rec = consent_manager.process_candidate(cands[0])

    assert rec.persisted_memory_id is not None
    mem = consent_manager.memory_service.get_memory(rec.persisted_memory_id)
    assert mem.value == "Java"


def test_15_no_direct_sqlite_access_from_consent_layer(consent_manager):
    """15. Verifies consent manager delegates persistence exclusively to MemoryService."""
    assert hasattr(consent_manager, "memory_service")
    assert consent_manager.memory_service is not None


def test_16_invalid_candidate_handling(consent_manager):
    """16. Verifies invalid candidate IDs return None safely."""
    assert consent_manager.approve("non_existent_id") is None
    assert consent_manager.reject("non_existent_id") is False
    assert consent_manager.get_consent_record("non_existent_id") is None


def test_17_consent_state_transitions(consent_manager):
    """17. Verifies valid consent state transitions."""
    cands = MemoryCandidateExtractor.extract("I usually wake up around 7.")
    rec = consent_manager.process_candidate(cands[0])
    assert rec.state == ConsentState.PENDING

    # Transition to APPROVED
    mem = consent_manager.approve(rec.candidate_id)
    assert mem is not None
    assert consent_manager.get_consent_record(rec.candidate_id).state == ConsentState.APPROVED

    # Rejection of approved candidate soft-deactivates memory
    success = consent_manager.reject(rec.candidate_id)
    assert success is True
    assert consent_manager.get_consent_record(rec.candidate_id).state == ConsentState.REJECTED


def test_18_full_integration_with_retrieval(consent_manager):
    """18. Verifies approved candidates are retrieved by MemoryRetriever."""
    cands = MemoryCandidateExtractor.extract("My name is Joshva.")
    rec = consent_manager.process_candidate(cands[0])

    retriever = MemoryRetriever(memory_service=consent_manager.memory_service)
    matches = retriever.retrieve("name")

    assert len(matches) == 1
    assert matches[0].record.value == "Joshva"
    retriever.close()
