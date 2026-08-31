import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from backend.memory.boundaries import (
    MAX_VALUE_LENGTH,
    contains_system_override_attempt,
    is_sensitive_content,
)
from backend.memory.models import MemoryCategory, MemorySource


class MemoryPolicyOutcome(str, Enum):
    """
    Decisions produced by the Memory Capture Policy.
    """
    ALLOW_CANDIDATE = "allow_candidate"        # Memory-worthy: high stability, explicitness, and value
    REQUIRE_CONFIRMATION = "require_confirmation" # Plausible candidate, but hedged, uncertain, or shifting
    TEMPORARY_CONTEXT = "temporary_context"    # Short-term, session-bound, or current activity context
    REJECT = "reject"                         # Rejected: secret, injection, low value, joke, or invalid


class MemoryProvenance(str, Enum):
    """
    Origin and provenance of candidate information.
    """
    USER_EXPLICIT = "user_explicit"
    USER_PREFERENCE = "user_preference"
    USER_RULE = "user_rule"
    SYSTEM_INFERRED = "system_inferred"
    ASSISTANT_GENERATED = "assistant_generated"
    TOOL_OUTPUT = "tool_output"


@dataclass
class MemoryPolicyDecision:
    """
    Structured decision output from evaluating a candidate memory statement.
    """
    outcome: MemoryPolicyOutcome
    category: Optional[MemoryCategory]
    confidence: float
    reason: str
    signals: List[str] = field(default_factory=list)
    suggested_key: Optional[str] = None
    suggested_value: Optional[str] = None
    is_time_sensitive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes decision to dictionary format."""
        return {
            "outcome": self.outcome.value,
            "category": self.category.value if self.category else None,
            "confidence": self.confidence,
            "reason": self.reason,
            "signals": self.signals,
            "suggested_key": self.suggested_key,
            "suggested_value": self.suggested_value,
            "is_time_sensitive": self.is_time_sensitive,
        }


# Pattern matching sets for policy classification
TEMPORARY_PATTERNS = [
    re.compile(r"(?i)\b(tonight|today|this evening|this morning|right now|currently|this session|in this chat)\b"),
    re.compile(r"(?i)\b(this assignment|this task|this bug|this function|this error|this code)\b"),
    re.compile(r"(?i)\b(debugging|fixing|testing|studying|working on)\s+(this|the)\b"),
    re.compile(r"(?i)\b(for now|temporarily|just for today)\b"),
]

HEDGED_PATTERNS = [
    re.compile(r"(?i)\b(might|maybe|thinking about|considering|probably|usually)\b"),
    re.compile(r"(?i)\b(I think I|I may|planning to maybe|inclined to|could be)\b"),
    re.compile(r"(?i)\b(switch my|change my|switch to)\b"),
]

HYPOTHETICAL_PATTERNS = [
    re.compile(r"(?i)\b(if i\b|suppose that|what if|hypothetically|in a parallel universe|pretend that)\b"),
]

JOKE_PATTERNS = [
    re.compile(r"(?i)\b(just kidding|jk|haha|lol|rofl|joke)\b"),
]

QUOTE_PATTERNS = [
    re.compile(r"['\"].*?['\"]"),
    re.compile(r"(?i)\b(said|told me|claims that|stated)\b"),
]

RULE_PATTERNS = [
    re.compile(r"(?i)\b(always|never)\s+(explain|show|format|use|include|respond|code)\b"),
    re.compile(r"(?i)\b(format\s+answers\s+as|when\s+coding|prefer\s+code\s+in)\b"),
]

PREFERENCE_PATTERNS = [
    re.compile(r"(?i)\b(prefer|favorite|like|dislike|primary\s+language|primary\s+tool)\b"),
]

PROFILE_PATTERNS = [
    re.compile(r"(?i)\b(my\s+name\s+is|i\s+am\s+a|i\s+live\s+in|i\s+work\s+as|preparing\s+for|my\s+goal\s+is)\b"),
]


class MemoryCapturePolicy:
    """
    Deterministic, side-effect-free policy engine evaluating whether candidate information
    is worthy of persistent memory storage, requires user confirmation, represents temporary context,
    or must be rejected.
    """

    @staticmethod
    def evaluate(
        text: str,
        provenance: MemoryProvenance = MemoryProvenance.USER_EXPLICIT,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryPolicyDecision:
        """
        Evaluates a statement against P.I.X.I.E. Memory Capture Policy.
        This function is strictly side-effect free and performs ZERO database writes.
        """
        signals: List[str] = []

        # 1. Basic validation and input checks
        if not text or not isinstance(text, str) or not text.strip():
            return MemoryPolicyDecision(
                outcome=MemoryPolicyOutcome.REJECT,
                category=None,
                confidence=0.0,
                reason="Invalid or empty input text.",
                signals=["invalid_input"],
            )

        text_clean = text.strip()

        if len(text_clean) > MAX_VALUE_LENGTH:
            return MemoryPolicyDecision(
                outcome=MemoryPolicyOutcome.REJECT,
                category=None,
                confidence=0.0,
                reason=f"Input text exceeds maximum length of {MAX_VALUE_LENGTH} characters.",
                signals=["oversized_input"],
            )

        # 2. Security Boundaries (Phase 6 enforcement)
        if is_sensitive_content(text_clean):
            return MemoryPolicyDecision(
                outcome=MemoryPolicyOutcome.REJECT,
                category=None,
                confidence=0.0,
                reason="Security Violation: Sensitive credentials or secret information detected.",
                signals=["sensitive_content_rejected"],
            )

        if contains_system_override_attempt(text_clean):
            return MemoryPolicyDecision(
                outcome=MemoryPolicyOutcome.REJECT,
                category=None,
                confidence=0.0,
                reason="Security Violation: System override or prompt injection attempt detected.",
                signals=["system_override_rejected"],
            )

        # 3. Provenance & Role Verification
        if role.lower() in ("assistant", "system", "tool") or provenance in (MemoryProvenance.ASSISTANT_GENERATED, MemoryProvenance.TOOL_OUTPUT):
            return MemoryPolicyDecision(
                outcome=MemoryPolicyOutcome.REJECT,
                category=None,
                confidence=0.0,
                reason="Rejection: Only explicit user statements or user rules are eligible for memory capture.",
                signals=["non_user_provenance"],
            )

        # 4. Jokes, Humor, and Sarcasm
        for p in JOKE_PATTERNS:
            if p.search(text_clean):
                return MemoryPolicyDecision(
                    outcome=MemoryPolicyOutcome.REJECT,
                    category=None,
                    confidence=0.0,
                    reason="Rejection: Joke, humor, or non-serious remark detected.",
                    signals=["joke_or_humor_detected"],
                )

        # 5. Hypothetical Statements
        for p in HYPOTHETICAL_PATTERNS:
            if p.search(text_clean):
                return MemoryPolicyDecision(
                    outcome=MemoryPolicyOutcome.REJECT,
                    category=None,
                    confidence=0.0,
                    reason="Rejection: Hypothetical or speculative statement detected.",
                    signals=["hypothetical_statement_detected"],
                )

        # 6. Quoted Statements (Third-party claims)
        for p in QUOTE_PATTERNS:
            if p.search(text_clean):
                signals.append("quoted_content_detected")
                return MemoryPolicyDecision(
                    outcome=MemoryPolicyOutcome.REJECT,
                    category=None,
                    confidence=0.0,
                    reason="Rejection: Quoted or third-party statement detected.",
                    signals=signals,
                )

        # 7. Temporality & Short-Term Context
        is_temp = False
        for p in TEMPORARY_PATTERNS:
            if p.search(text_clean):
                is_temp = True
                signals.append("temporary_context_signal")
                break

        if is_temp:
            return MemoryPolicyDecision(
                outcome=MemoryPolicyOutcome.TEMPORARY_CONTEXT,
                category=None,
                confidence=0.5,
                reason="Temporary Context: Information is short-term or task-specific.",
                signals=signals,
                is_time_sensitive=True,
            )

        # 8. Hedged / Uncertain / Shifting Statements -> REQUIRE_CONFIRMATION
        is_hedged = False
        for p in HEDGED_PATTERNS:
            if p.search(text_clean):
                is_hedged = True
                signals.append("hedged_uncertain_signal")
                break

        if is_hedged:
            # Determine potential category for confirmation suggestion
            cat = MemoryCategory.USER_PREFERENCE
            if any(p.search(text_clean) for p in PROFILE_PATTERNS):
                cat = MemoryCategory.USER_PROFILE
            elif any(p.search(text_clean) for p in RULE_PATTERNS):
                cat = MemoryCategory.CONTEXT_RULE

            return MemoryPolicyDecision(
                outcome=MemoryPolicyOutcome.REQUIRE_CONFIRMATION,
                category=cat,
                confidence=0.75,
                reason="Confirmation Required: Statement contains uncertainty, hedging, or potential preference shift.",
                signals=signals,
                suggested_value=text_clean,
            )

        # 9. Stable Fact / Preference / Context Rule Matching -> ALLOW_CANDIDATE
        for p in RULE_PATTERNS:
            if p.search(text_clean):
                signals.append("explicit_context_rule")
                return MemoryPolicyDecision(
                    outcome=MemoryPolicyOutcome.ALLOW_CANDIDATE,
                    category=MemoryCategory.CONTEXT_RULE,
                    confidence=0.95,
                    reason="Candidate Approved: Explicit operational context rule detected.",
                    signals=signals,
                    suggested_value=text_clean,
                )

        for p in PREFERENCE_PATTERNS:
            if p.search(text_clean):
                signals.append("explicit_user_preference")
                return MemoryPolicyDecision(
                    outcome=MemoryPolicyOutcome.ALLOW_CANDIDATE,
                    category=MemoryCategory.USER_PREFERENCE,
                    confidence=0.95,
                    reason="Candidate Approved: Explicit user preference detected.",
                    signals=signals,
                    suggested_value=text_clean,
                )

        for p in PROFILE_PATTERNS:
            if p.search(text_clean):
                signals.append("explicit_user_profile")
                return MemoryPolicyDecision(
                    outcome=MemoryPolicyOutcome.ALLOW_CANDIDATE,
                    category=MemoryCategory.USER_PROFILE,
                    confidence=0.95,
                    reason="Candidate Approved: Explicit user profile fact detected.",
                    signals=signals,
                    suggested_value=text_clean,
                )

        # General user facts (e.g. "Use Java as my primary language for coding problems")
        if provenance == MemoryProvenance.USER_EXPLICIT or role.lower() == "user":
            signals.append("explicit_user_fact")
            return MemoryPolicyDecision(
                outcome=MemoryPolicyOutcome.ALLOW_CANDIDATE,
                category=MemoryCategory.USER_FACT,
                confidence=0.85,
                reason="Candidate Approved: Explicit user statement.",
                signals=signals,
                suggested_value=text_clean,
            )

        # Fallback rejection
        return MemoryPolicyDecision(
            outcome=MemoryPolicyOutcome.REJECT,
            category=None,
            confidence=0.0,
            reason="Rejection: Unclear intent or insufficient memory value.",
            signals=["unclear_intent"],
        )
