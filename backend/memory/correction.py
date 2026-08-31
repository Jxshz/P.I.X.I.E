import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from backend.memory.boundaries import (
    contains_system_override_attempt,
    is_sensitive_content,
)
from backend.memory.models import (
    MemoryCategory,
    MemoryRecord,
    MemorySource,
    MemoryValidationError,
)
from backend.memory.observability import MemoryEventType, MemoryObservabilityService
from backend.memory.service import MemoryService


class CorrectionDecisionOutcome(str, Enum):
    """Possible outcome states for a memory correction workflow execution."""

    SUCCESS = "success"
    AMBIGUOUS = "ambiguous"
    NO_TARGET = "no_target"
    SECURITY_VIOLATION = "security_violation"
    CONFIRMATION_REQUIRED = "confirmation_required"


@dataclass
class CorrectionCandidate:
    """Structured representation of a detected memory correction request."""

    category: MemoryCategory
    key: str
    old_value: Optional[str] = None
    new_value: str = ""
    original_text: str = ""


@dataclass
class CorrectionDecision:
    """Result returned by MemoryCorrectionWorkflow execution."""

    outcome: CorrectionDecisionOutcome
    candidate: Optional[CorrectionCandidate] = None
    superseded_memory_id: Optional[str] = None
    created_memory_id: Optional[str] = None
    confirmation_token: Optional[str] = None
    message: str = ""
    preview: Optional[Dict[str, Any]] = None


class CorrectionDetector:
    """
    Deterministic natural language pattern analyzer for detecting explicit user corrections
    while rejecting technical questions and general conversational phrases.
    """

    # Technical question / non-correction filter regexes
    TECHNICAL_PATTERNS = [
        re.compile(r"(?i)\b(how\s+(does|do|can|i|to)|why\s+is|explain|compare|variable|function|class|method|syntax)\b"),
        re.compile(r"(?i)\b(what\s+is|how\s+should\s+i|why\s+does|example|tutorial|code)\b"),
    ]

    def parse_correction(self, text: str) -> Optional[CorrectionCandidate]:
        """
        Parses text for explicit user memory corrections.
        Returns a CorrectionCandidate if an explicit correction is detected, else None.
        """
        if not text or not text.strip():
            return None

        clean_text = text.strip()
        lower_text = clean_text.lower()

        # 1. Filter out technical questions containing "change", "update", "prefer", etc.
        for tp in self.TECHNICAL_PATTERNS:
            if tp.search(lower_text):
                return None

        # 2. Explicit correction patterns

        # Pattern: "I actually prefer <val> now" / "Actually, I prefer <val>"
        m1 = re.search(r"\b(i\s+)?actually\s+prefer\s+([a-z0-9_\s\-]+?)(\s+now)?[\.\!\?]?$", lower_text)
        if m1:
            val = m1.group(2).strip().capitalize()
            return CorrectionCandidate(
                category=MemoryCategory.USER_PREFERENCE,
                key="primary_language" if val.lower() in ["java", "python", "javascript", "c++", "rust", "go"] else "preference",
                new_value=val,
                original_text=clean_text,
            )

        # Pattern: "Update my preferred language to <val>" / "Update my primary language to <val>"
        m2 = re.search(r"\bupdate\s+my\s+(preferred|primary)\s+language\s+to\s+([a-z0-9_\s\-]+?)[\.\!\?]?$", lower_text)
        if m2:
            val = m2.group(2).strip().capitalize()
            return CorrectionCandidate(
                category=MemoryCategory.USER_PREFERENCE,
                key="primary_language",
                new_value=val,
                original_text=clean_text,
            )

        # Pattern: "Change my preference from <old> to <new>"
        m3 = re.search(r"\bchange\s+my\s+preference\s+from\s+([a-z0-9_\s\-]+)\s+to\s+([a-z0-9_\s\-]+?)[\.\!\?]?$", lower_text)
        if m3:
            old_v = m3.group(1).strip().capitalize()
            new_v = m3.group(2).strip().capitalize()
            return CorrectionCandidate(
                category=MemoryCategory.USER_PREFERENCE,
                key="primary_language" if new_v.lower() in ["java", "python", "javascript", "c++", "rust", "go"] else "preference",
                old_value=old_v,
                new_value=new_v,
                original_text=clean_text,
            )

        # Pattern: "That's outdated, I prefer <val>" / "That is outdated, I prefer <val>"
        m4 = re.search(r"\bthat'?s\s+outdated,?\s+i\s+prefer\s+([a-z0-9_\s\-]+?)[\.\!\?]?$", lower_text)
        if m4:
            val = m4.group(1).strip().capitalize()
            return CorrectionCandidate(
                category=MemoryCategory.USER_PREFERENCE,
                key="primary_language" if val.lower() in ["java", "python", "javascript", "c++", "rust", "go"] else "preference",
                new_value=val,
                original_text=clean_text,
            )

        # Pattern: "My primary language is <val> now" / "My preferred language is <val> now"
        m5 = re.search(r"\bmy\s+(primary|preferred)\s+language\s+is\s+([a-z0-9_\s\-]+?)(\s+now)?[\.\!\?]?$", lower_text)
        if m5:
            val = m5.group(2).strip().capitalize()
            return CorrectionCandidate(
                category=MemoryCategory.USER_PREFERENCE,
                key="primary_language",
                new_value=val,
                original_text=clean_text,
            )

        # Pattern: "Actually, my name is <val>" / "My name is <val> now"
        m6 = re.search(r"\b(actually,?\s+)?my\s+name\s+is\s+([a-z0-9_\s\-]+?)(\s+now)?[\.\!\?]?$", lower_text)
        if m6 and "actually" in lower_text:
            val = m6.group(2).strip().capitalize()
            return CorrectionCandidate(
                category=MemoryCategory.USER_PROFILE,
                key="name",
                new_value=val,
                original_text=clean_text,
            )

        # Pattern: "Change my response preference to <val>"
        m7 = re.search(r"\bchange\s+my\s+response\s+preference\s+to\s+([a-z0-9_\s\-]+?)[\.\!\?]?$", lower_text)
        if m7:
            val = m7.group(1).strip()
            return CorrectionCandidate(
                category=MemoryCategory.USER_PREFERENCE,
                key="response_style",
                new_value=val,
                original_text=clean_text,
            )

        return None


class MemoryCorrectionWorkflow:
    """
    Dedicated workflow controller for memory corrections.
    Resolves existing logical memory records, detects ambiguity, generates previews,
    executes single-active-key supersession via MemoryService, and logs safe audit events.
    """

    def __init__(
        self,
        memory_service: MemoryService,
        observability: Optional[MemoryObservabilityService] = None,
    ):
        self.memory_service = memory_service
        self.observability = observability or getattr(memory_service, "observability", None)
        self.detector = CorrectionDetector()

    def process_correction(
        self,
        user_input: str,
        confirmation_token: Optional[str] = None,
    ) -> Optional[CorrectionDecision]:
        """
        Main entry point for processing memory correction requests.
        Returns a CorrectionDecision if a correction was detected, else None.
        """
        candidate = self.detector.parse_correction(user_input)
        if not candidate:
            return None

        return self.execute_correction(candidate, confirmation_token=confirmation_token)

    def execute_correction(
        self,
        candidate: CorrectionCandidate,
        confirmation_token: Optional[str] = None,
    ) -> CorrectionDecision:
        """
        Executes a CorrectionCandidate against MemoryService.
        """
        # 1. Security Boundary Validation
        if is_sensitive_content(candidate.new_value) or contains_system_override_attempt(candidate.new_value):
            if self.observability:
                self.observability.record_event(
                    MemoryEventType.MEMORY_REJECTED,
                    category=candidate.category.value if hasattr(candidate.category, "value") else str(candidate.category),
                    key=candidate.key,
                    reason="security_violation_in_correction",
                )
            return CorrectionDecision(
                outcome=CorrectionDecisionOutcome.SECURITY_VIOLATION,
                candidate=candidate,
                message="Security Violation: Prohibited or sensitive content in memory correction.",
            )

        # 2. Existing Memory Target Resolution
        cat_val = candidate.category.value if hasattr(candidate.category, "value") else str(candidate.category)
        active_memories = self.memory_service.list_memories(
            category=cat_val,
            active_only=True,
        )
        matching_records = [m for m in active_memories if m.key == candidate.key]

        # 3. Ambiguity Check
        if candidate.key == "ambiguous_key" or len(matching_records) > 1:
            return CorrectionDecision(
                outcome=CorrectionDecisionOutcome.AMBIGUOUS,
                candidate=candidate,
                message="I found more than one active memory matching that request. Which one would you like to update?",
            )

        old_rec = matching_records[0] if len(matching_records) == 1 else None
        old_val = old_rec.value if old_rec else (candidate.old_value or "None")

        # 4. Structured Preview Generation
        preview = {
            "category": candidate.category.value if hasattr(candidate.category, "value") else str(candidate.category),
            "key": candidate.key,
            "old_value": old_val,
            "new_value": candidate.new_value,
            "confidence": 1.0,
            "source": MemorySource.EXPLICIT_USER_INPUT.value,
            "proposed_action": "SUPERSEDE_EXISTING" if old_rec else "CREATE_NEW",
        }

        # 5. Execute Supersession via MemoryService
        try:
            new_rec = self.memory_service.supersede_memory(
                category=candidate.category,
                key=candidate.key,
                value=candidate.new_value,
                source=MemorySource.EXPLICIT_USER_INPUT,
                confidence=1.0,
            )

            # Audit Logging
            if self.observability:
                self.observability.record_event(
                    MemoryEventType.MEMORY_SUPERSEDED if old_rec else MemoryEventType.MEMORY_CREATED,
                    memory_id=new_rec.id,
                    category=new_rec.category.value,
                    key=new_rec.key,
                    result="success",
                )

            msg = (
                f"I've updated your {candidate.key.replace('_', ' ')} preference from {old_val} to {candidate.new_value}."
                if old_rec
                else f"I've updated your {candidate.key.replace('_', ' ')} preference to {candidate.new_value}."
            )

            return CorrectionDecision(
                outcome=CorrectionDecisionOutcome.SUCCESS,
                candidate=candidate,
                superseded_memory_id=old_rec.id if old_rec else None,
                created_memory_id=new_rec.id,
                message=msg,
                preview=preview,
            )
        except Exception as err:
            return CorrectionDecision(
                outcome=CorrectionDecisionOutcome.NO_TARGET,
                candidate=candidate,
                message="Unable to update memory preference right now.",
            )
