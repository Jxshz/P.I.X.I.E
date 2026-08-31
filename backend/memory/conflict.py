import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from backend.memory.boundaries import (
    contains_system_override_attempt,
    is_sensitive_content,
)
from backend.memory.extraction import MemoryCandidate
from backend.memory.models import MemoryCategory, MemoryRecord, MemorySource
from backend.memory.policy import MemoryPolicyOutcome
from backend.memory.service import MemoryService


class ConflictResolutionOutcome(str, Enum):
    """
    Explicit outcomes produced by the Memory Conflict Resolver.
    """
    NO_CONFLICT = "no_conflict"                  # No active memory or identical value exists
    UPDATE_EXISTING = "update_existing"          # Non-contradictory metadata update
    SUPERSEDE_EXISTING = "supersede_existing"    # Candidate supersedes existing active memory
    REQUIRE_REVIEW = "require_review"            # Ambiguous conflict or candidate requiring confirmation
    REJECT_CONFLICT = "reject_conflict"          # Candidate rejected due to lower trust/security violation


@dataclass
class MemoryConflictDecision:
    """
    Structured resolution output describing how a MemoryCandidate resolves against existing memories.
    """
    outcome: ConflictResolutionOutcome
    candidate: MemoryCandidate
    conflicting_record: Optional[MemoryRecord] = None
    reason: str = ""
    confidence_delta: float = 0.0
    provenance_override: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes conflict decision to dictionary format."""
        return {
            "outcome": self.outcome.value,
            "candidate": self.candidate.to_dict(),
            "conflicting_record": self.conflicting_record.to_dict() if self.conflicting_record else None,
            "reason": self.reason,
            "confidence_delta": self.confidence_delta,
            "provenance_override": self.provenance_override,
        }


# Provenance trust ranking (higher integer means higher trust)
PROVENANCE_RANK = {
    MemorySource.EXPLICIT_USER_INPUT: 2,
    MemorySource.SYSTEM_INFERRED: 1,
}


class MemoryConflictResolver:
    """
    Deterministic, side-effect-free conflict resolution engine.
    Analyzes MemoryCandidates against existing persistent memories for the same logical key (category, key).
    """

    def __init__(self, memory_service: Optional[MemoryService] = None):
        self.memory_service = memory_service

    def resolve(
        self,
        candidate: MemoryCandidate,
        existing_records: Optional[List[MemoryRecord]] = None,
    ) -> MemoryConflictDecision:
        """
        Deterministically evaluates conflict resolution state between a candidate and existing memories.
        This operation is strictly side-effect free and performs ZERO database writes.
        """
        # 1. Security boundary check
        if is_sensitive_content(candidate.value) or contains_system_override_attempt(candidate.value):
            return MemoryConflictDecision(
                outcome=ConflictResolutionOutcome.REJECT_CONFLICT,
                candidate=candidate,
                reason="Security Violation: Sensitive content or system override detected in candidate.",
            )

        # 2. Policy outcome check
        if candidate.policy_decision.outcome == MemoryPolicyOutcome.REJECT:
            return MemoryConflictDecision(
                outcome=ConflictResolutionOutcome.REJECT_CONFLICT,
                candidate=candidate,
                reason=candidate.policy_decision.reason or "Policy Rejection.",
            )

        if candidate.policy_decision.outcome == MemoryPolicyOutcome.TEMPORARY_CONTEXT:
            return MemoryConflictDecision(
                outcome=ConflictResolutionOutcome.REJECT_CONFLICT,
                candidate=candidate,
                reason="Temporary Context: Cannot overwrite or update persistent memory.",
            )

        # 3. Retrieve relevant active memory for (category, key)
        active_target: Optional[MemoryRecord] = None
        now = time.time()

        if existing_records is not None:
            for rec in existing_records:
                if (
                    rec.category == candidate.category
                    and rec.key.lower() == candidate.key.lower()
                    and rec.is_active
                    and (rec.expires_at is None or rec.expires_at > now)
                ):
                    active_target = rec
                    break
        elif self.memory_service is not None:
            active_target = self.memory_service.get_memory_by_key(
                category=candidate.category,
                key=candidate.key,
                active_only=True,
            )

        # 4. Case: No active existing memory for logical key
        if not active_target:
            if candidate.policy_decision.outcome == MemoryPolicyOutcome.REQUIRE_CONFIRMATION:
                return MemoryConflictDecision(
                    outcome=ConflictResolutionOutcome.REQUIRE_REVIEW,
                    candidate=candidate,
                    reason="Confirmation Required: Candidate requires user confirmation before initial persistence.",
                )
            return MemoryConflictDecision(
                outcome=ConflictResolutionOutcome.NO_CONFLICT,
                candidate=candidate,
                reason="No active existing memory for logical key.",
            )

        # 5. Case: Active existing memory present
        cand_val_clean = candidate.value.strip().lower()
        target_val_clean = active_target.value.strip().lower()

        # A. Identical Value -> NO_CONFLICT / UPDATE_EXISTING
        if cand_val_clean == target_val_clean:
            return MemoryConflictDecision(
                outcome=ConflictResolutionOutcome.NO_CONFLICT,
                candidate=candidate,
                conflicting_record=active_target,
                reason="Identical memory value already active in system.",
            )

        # B. Contradictory Value Evaluation
        cand_rank = PROVENANCE_RANK.get(candidate.source, 1)
        target_rank = PROVENANCE_RANK.get(active_target.source, 1)

        # Policy confirmation requirement overrides auto-supersession
        if candidate.policy_decision.outcome == MemoryPolicyOutcome.REQUIRE_CONFIRMATION:
            return MemoryConflictDecision(
                outcome=ConflictResolutionOutcome.REQUIRE_REVIEW,
                candidate=candidate,
                conflicting_record=active_target,
                reason="Contradictory candidate requires explicit user review and confirmation.",
            )

        # Lower trust candidate trying to overwrite higher trust memory -> REJECT_CONFLICT
        if cand_rank < target_rank:
            return MemoryConflictDecision(
                outcome=ConflictResolutionOutcome.REJECT_CONFLICT,
                candidate=candidate,
                conflicting_record=active_target,
                reason="Candidate has lower trust provenance than existing explicit memory.",
            )

        # Higher trust candidate replacing lower trust memory -> SUPERSEDE_EXISTING
        if cand_rank > target_rank:
            return MemoryConflictDecision(
                outcome=ConflictResolutionOutcome.SUPERSEDE_EXISTING,
                candidate=candidate,
                conflicting_record=active_target,
                provenance_override=True,
                reason="Explicit user candidate supersedes lower-trust inferred memory.",
            )

        # Equal trust provenance: compare confidence and recency
        conf_delta = candidate.confidence - active_target.confidence

        if conf_delta < -0.15:
            return MemoryConflictDecision(
                outcome=ConflictResolutionOutcome.REJECT_CONFLICT,
                candidate=candidate,
                conflicting_record=active_target,
                confidence_delta=conf_delta,
                reason="Candidate confidence is significantly lower than existing memory.",
            )

        if conf_delta > 0.15:
            return MemoryConflictDecision(
                outcome=ConflictResolutionOutcome.SUPERSEDE_EXISTING,
                candidate=candidate,
                conflicting_record=active_target,
                confidence_delta=conf_delta,
                reason="Candidate has significantly higher confidence than existing memory.",
            )

        # Equal provenance and comparable confidence -> Recency tie-break: newer explicit statement wins
        return MemoryConflictDecision(
            outcome=ConflictResolutionOutcome.SUPERSEDE_EXISTING,
            candidate=candidate,
            conflicting_record=active_target,
            confidence_delta=conf_delta,
            reason="Newer explicit statement supersedes older memory for logical key.",
        )

    def apply_resolution(self, decision: MemoryConflictDecision) -> Optional[MemoryRecord]:
        """
        Applies an approved conflict resolution decision using MemoryService.
        Only SUPERSEDE_EXISTING or UPDATE_EXISTING decisions result in database persistence.
        """
        if not self.memory_service:
            return None

        if decision.outcome in (ConflictResolutionOutcome.SUPERSEDE_EXISTING, ConflictResolutionOutcome.UPDATE_EXISTING):
            metadata_dict = decision.candidate.metadata or {}
            metadata_dict["conflict_resolution"] = decision.outcome.value
            metadata_dict["resolution_reason"] = decision.reason
            metadata_json = json.dumps(metadata_dict)

            return self.memory_service.supersede_memory(
                category=decision.candidate.category,
                key=decision.candidate.key,
                value=decision.candidate.value,
                source=decision.candidate.source,
                confidence=decision.candidate.confidence,
                metadata_json=metadata_json,
            )

        return None
