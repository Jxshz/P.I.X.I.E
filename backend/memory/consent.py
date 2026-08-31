import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from backend.memory.boundaries import (
    contains_system_override_attempt,
    is_sensitive_content,
    validate_metadata_json,
)
from backend.memory.extraction import MemoryCandidate
from backend.memory.models import MemoryCategory, MemoryRecord, MemoryValidationError
from backend.memory.policy import MemoryPolicyOutcome, MemoryProvenance
from backend.memory.service import MemoryService


class ConsentState(str, Enum):
    """
    Explicit lifecycle states for candidate memory approval requests.
    """
    PENDING = "pending"          # Awaiting explicit user approval
    APPROVED = "approved"        # Explicitly approved by user action
    AUTO_APPROVED = "auto_approved" # Auto-approved via policy rules for ALLOW_CANDIDATE
    REJECTED = "rejected"        # Explicitly rejected by user or policy
    EXPIRED = "expired"          # Approval request timed out or expired


@dataclass
class ConsentRecord:
    """
    Managed consent record representing an approval request for a MemoryCandidate.
    """
    candidate_id: str
    candidate: MemoryCandidate
    state: ConsentState
    created_at: float
    updated_at: float
    expires_at: Optional[float] = None
    persisted_memory_id: Optional[str] = None
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes consent record to dictionary format."""
        return {
            "candidate_id": self.candidate_id,
            "candidate": self.candidate.to_dict(),
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "persisted_memory_id": self.persisted_memory_id,
            "rejection_reason": self.rejection_reason,
        }


class MemoryConsentManager:
    """
    Coordinates candidate consent workflows, policy outcomes, and MemoryService persistence.
    Does NOT access SQLite directly; delegates all storage operations exclusively to MemoryService.
    """

    def __init__(
        self,
        memory_service: Optional[MemoryService] = None,
        ttl_seconds: float = 86400.0,
        auto_approve_candidates: bool = True,
    ):
        self.memory_service = memory_service or MemoryService(db_path=":memory:")
        self.ttl_seconds = ttl_seconds
        self.auto_approve_candidates = auto_approve_candidates
        self._records: Dict[str, ConsentRecord] = {}

    def process_candidate(self, candidate: MemoryCandidate) -> ConsentRecord:
        """
        Evaluates a MemoryCandidate, applies policy outcomes and security checks,
        and constructs a managed ConsentRecord.
        """
        now = time.time()
        candidate_id = f"c-{uuid.uuid4()}"
        expires_at = now + self.ttl_seconds

        # 1. Phase 6 Security Boundary Scan
        if is_sensitive_content(candidate.value) or contains_system_override_attempt(candidate.value):
            rec = ConsentRecord(
                candidate_id=candidate_id,
                candidate=candidate,
                state=ConsentState.REJECTED,
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
                rejection_reason="Security Violation: Sensitive content or system override detected.",
            )
            self._records[candidate_id] = rec
            return rec

        if candidate.evidence and (is_sensitive_content(candidate.evidence) or contains_system_override_attempt(candidate.evidence)):
            rec = ConsentRecord(
                candidate_id=candidate_id,
                candidate=candidate,
                state=ConsentState.REJECTED,
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
                rejection_reason="Security Violation: Sensitive content detected in evidence.",
            )
            self._records[candidate_id] = rec
            return rec

        outcome = candidate.policy_decision.outcome

        # 2. Handle Policy Outcomes
        if outcome == MemoryPolicyOutcome.REJECT:
            rec = ConsentRecord(
                candidate_id=candidate_id,
                candidate=candidate,
                state=ConsentState.REJECTED,
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
                rejection_reason=candidate.policy_decision.reason or "Policy Rejection",
            )
            self._records[candidate_id] = rec
            return rec

        if outcome == MemoryPolicyOutcome.TEMPORARY_CONTEXT:
            rec = ConsentRecord(
                candidate_id=candidate_id,
                candidate=candidate,
                state=ConsentState.REJECTED,
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
                rejection_reason="Temporary Context: Not eligible for persistent storage.",
            )
            self._records[candidate_id] = rec
            return rec

        if outcome == MemoryPolicyOutcome.REQUIRE_CONFIRMATION:
            rec = ConsentRecord(
                candidate_id=candidate_id,
                candidate=candidate,
                state=ConsentState.PENDING,
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
            )
            self._records[candidate_id] = rec
            return rec

        if outcome == MemoryPolicyOutcome.ALLOW_CANDIDATE:
            if self.auto_approve_candidates:
                # Auto-approve and delegate persistence to MemoryService
                persisted = self._persist_to_service(candidate)
                rec = ConsentRecord(
                    candidate_id=candidate_id,
                    candidate=candidate,
                    state=ConsentState.AUTO_APPROVED,
                    created_at=now,
                    updated_at=now,
                    expires_at=expires_at,
                    persisted_memory_id=persisted.id if persisted else None,
                )
                self._records[candidate_id] = rec
                return rec
            else:
                rec = ConsentRecord(
                    candidate_id=candidate_id,
                    candidate=candidate,
                    state=ConsentState.PENDING,
                    created_at=now,
                    updated_at=now,
                    expires_at=expires_at,
                )
                self._records[candidate_id] = rec
                return rec

        # Default fallback pending
        rec = ConsentRecord(
            candidate_id=candidate_id,
            candidate=candidate,
            state=ConsentState.PENDING,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
        )
        self._records[candidate_id] = rec
        return rec

    def approve(self, candidate_id: str) -> Optional[MemoryRecord]:
        """
        Explicit user action approving a candidate.
        Delegates persistence exclusively to MemoryService.
        Is fully idempotent: approving an already approved candidate returns existing MemoryRecord.
        """
        rec = self._records.get(candidate_id)
        if not rec:
            return None

        # Idempotency check: if already approved and persisted, return existing memory record
        if rec.state in (ConsentState.APPROVED, ConsentState.AUTO_APPROVED) and rec.persisted_memory_id:
            return self.memory_service.get_memory(rec.persisted_memory_id)

        # Rejected or expired candidates cannot be persisted
        if rec.state == ConsentState.REJECTED:
            return None

        now = time.time()
        if rec.expires_at and now > rec.expires_at:
            rec.state = ConsentState.EXPIRED
            rec.updated_at = now
            return None

        # Re-verify Phase 6 security boundary
        if is_sensitive_content(rec.candidate.value) or contains_system_override_attempt(rec.candidate.value):
            rec.state = ConsentState.REJECTED
            rec.updated_at = now
            rec.rejection_reason = "Security Violation on Approval"
            return None

        # Persist through MemoryService
        persisted = self._persist_to_service(rec.candidate)
        if persisted:
            rec.state = ConsentState.APPROVED
            rec.updated_at = now
            rec.persisted_memory_id = persisted.id
            return persisted

        return None

    def reject(self, candidate_id: str, reason: Optional[str] = None) -> bool:
        """
        Explicit user action rejecting a candidate.
        If candidate was previously persisted, soft-deactivates memory via MemoryService.
        """
        rec = self._records.get(candidate_id)
        if not rec:
            return False

        now = time.time()
        if rec.state in (ConsentState.APPROVED, ConsentState.AUTO_APPROVED) and rec.persisted_memory_id:
            self.memory_service.forget_memory(rec.persisted_memory_id)

        rec.state = ConsentState.REJECTED
        rec.updated_at = now
        rec.rejection_reason = reason or "Explicit user rejection."
        return True

    def get_pending_requests(self) -> List[ConsentRecord]:
        """
        Returns active pending consent requests that have not expired.
        """
        now = time.time()
        results: List[ConsentRecord] = []
        for rec in self._records.values():
            if rec.state == ConsentState.PENDING:
                if rec.expires_at and now > rec.expires_at:
                    rec.state = ConsentState.EXPIRED
                    rec.updated_at = now
                else:
                    results.append(rec)
        return results

    def get_consent_record(self, candidate_id: str) -> Optional[ConsentRecord]:
        """Retrieves a managed ConsentRecord by candidate_id."""
        return self._records.get(candidate_id)

    def _persist_to_service(self, candidate: MemoryCandidate) -> Optional[MemoryRecord]:
        """
        Internal helper delegating persistence to MemoryService.supersede_memory.
        """
        try:
            metadata_dict = candidate.metadata or {}
            metadata_dict["extracted_confidence"] = candidate.confidence
            metadata_dict["evidence"] = candidate.evidence
            metadata_json = json.dumps(metadata_dict)

            return self.memory_service.supersede_memory(
                category=candidate.category,
                key=candidate.key,
                value=candidate.value,
                source=candidate.source,
                confidence=candidate.confidence,
                metadata_json=metadata_json,
            )
        except MemoryValidationError:
            return None
