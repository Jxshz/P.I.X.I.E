import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.memory.boundaries import (
    MAX_KEY_LENGTH,
    MAX_VALUE_LENGTH,
    contains_system_override_attempt,
    is_sensitive_content,
)
from backend.memory.models import MemoryCategory, MemorySource
from backend.memory.policy import (
    MemoryCapturePolicy,
    MemoryPolicyDecision,
    MemoryPolicyOutcome,
    MemoryProvenance,
)


@dataclass
class MemoryCandidate:
    """
    Structured representation of a potential persistent memory extracted from user interaction.
    Phase 7.2 produces candidates only. Persistence belongs exclusively to Phase 7.3 (Consent).
    """
    category: MemoryCategory
    key: str
    value: str
    source: MemorySource
    confidence: float
    evidence: str
    policy_decision: MemoryPolicyDecision
    extraction_reason: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes candidate to a JSON-compatible dictionary."""
        return {
            "category": self.category.value,
            "key": self.key,
            "value": self.value,
            "source": self.source.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "policy_decision": self.policy_decision.to_dict(),
            "extraction_reason": self.extraction_reason,
            "metadata": self.metadata,
        }


# Extraction pattern matchers for canonical key/value identification
EXTRACT_NAME_PATTERN = re.compile(r"(?i)\bmy\s+name\s+is\s+([A-Za-z0-9_\-\s]{2,30})\b")
EXTRACT_PREF_STYLE_PATTERN = re.compile(r"(?i)\bprefer\s+([A-Za-z0-9_\-\s]{2,30})\s+(answers|responses|explanations)\b")
EXTRACT_PRIMARY_LANG_PATTERN = re.compile(r"(?i)\buse\s+([A-Za-z0-9_+#\-]+)\s+as\s+my\s+primary\s+(language|tool|framework)\b")
EXTRACT_PREF_LANG_PATTERN = re.compile(r"(?i)\b(prefer|like)\s+([A-Za-z0-9_+#\-]+)\s+(with|for)\b")
EXTRACT_GOAL_PATTERN = re.compile(r"(?i)\b(preparing\s+for|my\s+goal\s+is)\s+([A-Za-z0-9_\-\s]{2,50})\b")
EXTRACT_LOCATION_PATTERN = re.compile(r"(?i)\b(i\s+live\s+in|i\s+reside\s+in)\s+([A-Za-z0-9_\-\s]{2,30})\b")
EXTRACT_ALWAYS_RULE_PATTERN = re.compile(r"(?i)\balways\s+([A-Za-z0-9_\-\s]{2,60})\b")
EXTRACT_NEVER_RULE_PATTERN = re.compile(r"(?i)\bnever\s+([A-Za-z0-9_\-\s]{2,60})\b")


class MemoryCandidateExtractor:
    """
    Deterministic, side-effect-free extraction engine for identifying potential
    long-term memory candidates from user input.
    """

    @staticmethod
    def extract(
        user_input: str,
        provenance: MemoryProvenance = MemoryProvenance.USER_EXPLICIT,
        role: str = "user",
    ) -> List[MemoryCandidate]:
        """
        Extracts structured MemoryCandidate objects from user input.
        This function is strictly side-effect free and performs ZERO database writes.
        """
        # 1. Non-user role / non-user provenance guard
        if role.lower() in ("assistant", "system", "tool") or provenance in (MemoryProvenance.ASSISTANT_GENERATED, MemoryProvenance.TOOL_OUTPUT):
            return []

        # 2. Input validation and security scanning
        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            return []

        input_text = user_input.strip()

        if is_sensitive_content(input_text) or contains_system_override_attempt(input_text):
            return []

        # 3. Decompose composite message into candidate clauses
        # Splits on sentence punctuation and key conjunction clauses (" and ", " plus ", ". ")
        raw_clauses = re.split(r"(?<=[.!?])\s+|\s+\band\b\s+|\s+\bplus\b\s+|;\s*", input_text)
        clauses = [c.strip(" .,;!?") for c in raw_clauses if c and len(c.strip(" .,;!?")) > 3]

        if not clauses:
            clauses = [input_text]

        candidates: List[MemoryCandidate] = []
        seen_keys: set = set()

        for clause in clauses:
            # Enforce boundary security on individual clause
            if is_sensitive_content(clause) or contains_system_override_attempt(clause):
                continue

            decision = MemoryCapturePolicy.evaluate(clause, provenance=provenance, role=role)

            if decision.outcome == MemoryPolicyOutcome.REJECT:
                continue

            # Identify category, key, value, and extraction details
            cat, key, val, reason = MemoryCandidateExtractor._infer_key_value(clause, decision)

            if not key or not val:
                continue

            # Key duplication check within single message
            combo_key = f"{cat.value}:{key.lower()}"
            if combo_key in seen_keys:
                continue
            seen_keys.add(combo_key)

            # Cap lengths according to Phase 6 boundary limits
            safe_key = key[:MAX_KEY_LENGTH]
            safe_val = val[:MAX_VALUE_LENGTH]
            safe_evidence = clause[:MAX_VALUE_LENGTH]

            # Adjust source enum from provenance
            src = MemorySource.EXPLICIT_USER_INPUT if provenance in (
                MemoryProvenance.USER_EXPLICIT, MemoryProvenance.USER_PREFERENCE, MemoryProvenance.USER_RULE
            ) else MemorySource.SYSTEM_INFERRED

            candidate = MemoryCandidate(
                category=cat,
                key=safe_key,
                value=safe_val,
                source=src,
                confidence=decision.confidence,
                evidence=safe_evidence,
                policy_decision=decision,
                extraction_reason=reason,
                metadata={"clause": clause, "outcome": decision.outcome.value},
            )
            candidates.append(candidate)

        return candidates

    @staticmethod
    def _infer_key_value(
        clause: str, decision: MemoryPolicyDecision
    ) -> tuple[MemoryCategory, str, str, str]:
        """
        Infers category, key, value, and reason from a clause and its policy decision.
        """
        cat = decision.category or MemoryCategory.USER_FACT

        # 1. Name Profile Match
        m_name = EXTRACT_NAME_PATTERN.search(clause)
        if m_name:
            return (MemoryCategory.USER_PROFILE, "name", m_name.group(1).strip(), "Extracted user name")

        # 2. Goal Profile Match
        m_goal = EXTRACT_GOAL_PATTERN.search(clause)
        if m_goal:
            return (MemoryCategory.USER_PROFILE, "goal", m_goal.group(2).strip(), "Extracted user preparation/goal")

        # 3. Location Profile Match
        m_loc = EXTRACT_LOCATION_PATTERN.search(clause)
        if m_loc:
            return (MemoryCategory.USER_PROFILE, "location", m_loc.group(2).strip(), "Extracted user location")

        # 4. Response Style Preference Match
        m_style = EXTRACT_PREF_STYLE_PATTERN.search(clause)
        if m_style:
            return (MemoryCategory.USER_PREFERENCE, "response_style", m_style.group(1).strip(), "Extracted response style preference")

        # 5. Primary Language Preference Match
        m_prim_lang = EXTRACT_PRIMARY_LANG_PATTERN.search(clause)
        if m_prim_lang:
            return (MemoryCategory.USER_PREFERENCE, "primary_language", m_prim_lang.group(1).strip(), "Extracted primary language preference")

        # 6. Preferred Coding Language Match
        m_pref_lang = EXTRACT_PREF_LANG_PATTERN.search(clause)
        if m_pref_lang:
            return (MemoryCategory.USER_PREFERENCE, "preferred_language", m_pref_lang.group(2).strip(), "Extracted preferred coding language")

        # 7. Always Context Rule Match
        m_always = EXTRACT_ALWAYS_RULE_PATTERN.search(clause)
        if m_always:
            return (MemoryCategory.CONTEXT_RULE, "always_rule", m_always.group(1).strip(), "Extracted operational rule (always)")

        # 8. Never Context Rule Match
        m_never = EXTRACT_NEVER_RULE_PATTERN.search(clause)
        if m_never:
            return (MemoryCategory.CONTEXT_RULE, "never_rule", m_never.group(1).strip(), "Extracted operational rule (never)")

        # Fallback Key Inferencing based on Category
        if cat == MemoryCategory.USER_PROFILE:
            return (cat, "profile_detail", clause, "Extracted profile detail")
        elif cat == MemoryCategory.USER_PREFERENCE:
            return (cat, "user_preference", clause, "Extracted user preference")
        elif cat == MemoryCategory.CONTEXT_RULE:
            return (cat, "context_rule", clause, "Extracted context rule")
        else:
            return (MemoryCategory.USER_FACT, "user_fact", clause, "Extracted user fact")
