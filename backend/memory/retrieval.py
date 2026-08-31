import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

from backend.memory.boundaries import is_sensitive_content
from backend.memory.models import MemoryCategory, MemoryRecord, MemorySource
from backend.memory.service import MemoryService
from backend.storage.memory_store import MemoryStore

logger = logging.getLogger(__name__)

MAX_RETRIEVAL_LIMIT = 20
DEFAULT_RETRIEVAL_LIMIT = 5
MAX_QUERY_LENGTH = 1000


@dataclass
class MemoryMatch:
    """
    Result of a memory retrieval query, exposing the matched record,
    calculated relevance score, and human-readable match signals.
    """
    record: MemoryRecord
    relevance_score: float
    matched_signals: List[str] = field(default_factory=list)


def _tokenize(text: str) -> Set[str]:
    """Tokenizes and normalizes text into a set of lowercased alphanumeric words."""
    if not text:
        return set()
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return {word for word in cleaned.split() if len(word) > 1}


# Intent Patterns for Query Intent Classification
PROFILE_INTENT_PATTERNS = [
    re.compile(r"(?i)\b(my\s+name|who\s+am\s+i|my\s+goal|my\s+occupation|my\s+job|my\s+location|where\s+do\s+i\s+live|my\s+identity)\b"),
    re.compile(r"(?i)\b(name\?|name\s+is\?|who\s+i\s+am)\b"),
]

PREFERENCE_INTENT_PATTERNS = [
    re.compile(r"(?i)\b(how\s+should\s+you\s+explain|what\s+language|preferred\s+language|primary\s+language|my\s+preference|coding\s+language|favorite|prefer)\b"),
    re.compile(r"(?i)\b(what\s+language\s+should\s+we\s+use|how\s+do\s+i\s+like|style)\b"),
]

RULE_INTENT_PATTERNS = [
    re.compile(r"(?i)\b(format\s+answers|how\s+to\s+format|always\s+explain|never\s+use|coding\s+rule|instructions|remember\s+how\s+i\s+want)\b"),
    re.compile(r"(?i)\b(always|never)\b"),
]

FACT_INTENT_PATTERNS = [
    re.compile(r"(?i)\b(what\s+facts|what\s+do\s+you\s+know|my\s+background|facts\s+about\s+me)\b"),
]


class QueryIntentAnalyzer:
    """
    Deterministic, local query intent classifier for memory retrieval.
    Analyzes query keywords to infer target memory categories.
    """

    @staticmethod
    def analyze_intent(query: str) -> Dict[MemoryCategory, float]:
        """
        Analyzes query intent and returns category boost weights.
        """
        if not query:
            return {}

        query_clean = query.lower().strip()
        boosts: Dict[MemoryCategory, float] = {}

        for p in PROFILE_INTENT_PATTERNS:
            if p.search(query_clean):
                boosts[MemoryCategory.USER_PROFILE] = boosts.get(MemoryCategory.USER_PROFILE, 0.0) + 1.5
                break

        for p in PREFERENCE_INTENT_PATTERNS:
            if p.search(query_clean):
                boosts[MemoryCategory.USER_PREFERENCE] = boosts.get(MemoryCategory.USER_PREFERENCE, 0.0) + 1.5
                break

        for p in RULE_INTENT_PATTERNS:
            if p.search(query_clean):
                boosts[MemoryCategory.CONTEXT_RULE] = boosts.get(MemoryCategory.CONTEXT_RULE, 0.0) + 1.5
                boosts[MemoryCategory.USER_PREFERENCE] = boosts.get(MemoryCategory.USER_PREFERENCE, 0.0) + 0.5
                break

        for p in FACT_INTENT_PATTERNS:
            if p.search(query_clean):
                boosts[MemoryCategory.USER_FACT] = boosts.get(MemoryCategory.USER_FACT, 0.0) + 1.5
                break

        return boosts


class MemoryRetriever:
    """
    Deterministic, intent-aware, read-only memory retrieval engine.
    Applies intent classification, category weighting, source/confidence weights, recency,
    redundancy control, and diversity selection without LLM/vector network overhead.
    """

    def __init__(
        self,
        memory_service: Optional[MemoryService] = None,
        memory_store: Optional[MemoryStore] = None,
        db_path: Optional[str] = None,
        observability: Optional[Any] = None,
    ):
        if memory_service:
            self.service = memory_service
        elif memory_store:
            self.service = MemoryService(memory_store=memory_store)
        else:
            self.service = MemoryService(db_path=db_path)
        self.observability = observability or getattr(self.service, "observability", None)
        self.last_retrieval_failed = False

    def calculate_relevance(
        self,
        query: str,
        record: MemoryRecord,
        category_filter: Optional[Union[MemoryCategory, str]] = None,
        intent_boosts: Optional[Dict[MemoryCategory, float]] = None,
        now: Optional[float] = None,
    ) -> MemoryMatch:
        """
        Calculates a deterministic relevance score for a MemoryRecord against a query string.
        """
        now = now or time.time()
        matched_signals: List[str] = []
        score = 0.0

        query_clean = query.lower().strip()
        query_tokens = _tokenize(query)

        key_clean = record.key.lower().strip()
        key_tokens = _tokenize(record.key)

        val_clean = record.value.lower().strip()
        val_tokens = _tokenize(record.value)

        has_query_signal = False

        # 1. Exact key match or key substring match
        if query_clean == key_clean:
            score += 3.0
            matched_signals.append("exact_key_match")
            has_query_signal = True
        elif key_clean in query_clean or query_clean in key_clean:
            score += 1.5
            matched_signals.append("partial_key_match")
            has_query_signal = True

        # 2. Token overlap matching
        key_overlap = query_tokens.intersection(key_tokens)
        val_overlap = query_tokens.intersection(val_tokens)

        if key_overlap:
            key_token_score = len(key_overlap) * 1.0
            score += key_token_score
            matched_signals.append(f"key_tokens_matched:{','.join(sorted(key_overlap))}")
            has_query_signal = True

        if val_overlap:
            val_token_score = len(val_overlap) * 0.5
            score += val_token_score
            matched_signals.append(f"value_tokens_matched:{','.join(sorted(val_overlap))}")
            has_query_signal = True

        # 3. Substring / phrase match in value
        if len(query_clean) >= 3 and query_clean in val_clean:
            score += 1.0
            matched_signals.append("phrase_match_in_value")
            has_query_signal = True

        # 4. Intent & Category Weighting
        if intent_boosts and record.category in intent_boosts:
            intent_score = intent_boosts[record.category]
            score += intent_score
            matched_signals.append(f"intent_match:{record.category.value}")
            has_query_signal = True

        if category_filter:
            cat_str = category_filter.value if isinstance(category_filter, MemoryCategory) else str(category_filter)
            if record.category.value == cat_str:
                score += 0.5
                matched_signals.append(f"category_match:{cat_str}")
                has_query_signal = True

        # If zero query or category signals matched, return zero score (completely irrelevant)
        if not has_query_signal:
            return MemoryMatch(
                record=record,
                relevance_score=0.0,
                matched_signals=[],
            )

        # 5. Provenance Source Weighting (Explicit User Input > System Inferred)
        if record.source == MemorySource.EXPLICIT_USER_INPUT:
            score += 0.5
            matched_signals.append("source_weight:explicit_user_input")
        elif record.source == MemorySource.SYSTEM_INFERRED:
            score += 0.1
            matched_signals.append("source_weight:system_inferred")

        # 6. Confidence weight (0.0 to 0.3)
        conf_weight = record.confidence * 0.3
        score += conf_weight
        matched_signals.append(f"confidence_weight:{round(conf_weight, 2)}")

        # 7. Recency bonus (max 0.2 score decay over 30 days)
        age_seconds = max(0.0, now - record.updated_at)
        recency_factor = max(0.0, 1.0 - (age_seconds / 2592000.0))
        recency_bonus = recency_factor * 0.2
        score += recency_bonus
        if recency_bonus > 0.05:
            matched_signals.append(f"recency_bonus:{round(recency_bonus, 2)}")

        return MemoryMatch(
            record=record,
            relevance_score=round(score, 4),
            matched_signals=matched_signals,
        )

    def retrieve(
        self,
        query: str,
        category: Optional[Union[MemoryCategory, str]] = None,
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
        min_confidence: float = 0.0,
        min_score: float = 0.1,
    ) -> List[MemoryMatch]:
        """
        Retrieves relevant, active, non-expired memory records ranked deterministically.
        Strictly read-only and fail-safe.
        """
        self.last_retrieval_failed = False
        if not query or not isinstance(query, str) or not query.strip():
            return []

        try:
            safe_query = query[:MAX_QUERY_LENGTH].strip()
            safe_limit = max(1, min(limit, MAX_RETRIEVAL_LIMIT))

            # Analyze query intent
            intent_boosts = QueryIntentAnalyzer.analyze_intent(safe_query)

            # Fetch candidate active memories from MemoryService
            candidates = self.service.list_memories(category=category, active_only=True, limit=200)

            now = time.time()
            valid_matches: List[MemoryMatch] = []
            seen_logical_keys: Dict[str, MemoryMatch] = {}

            for record in candidates:
                # Exclude inactive memories
                if not record.is_active:
                    continue

                # Exclude expired memories
                if record.expires_at is not None and record.expires_at <= now:
                    continue

                # Exclude low-confidence memories below threshold
                if record.confidence < min_confidence:
                    continue

                # Defense-in-depth: Exclude secrets/sensitive content
                if is_sensitive_content(record.key) or is_sensitive_content(record.value):
                    continue

                match = self.calculate_relevance(
                    safe_query, record, category_filter=category, intent_boosts=intent_boosts, now=now
                )

                if match.relevance_score < min_score or not match.matched_signals:
                    continue

                # Redundancy & Conflict Suppression:
                # Keep strongest match per logical key (category, key)
                logical_key = f"{record.category.value}:{record.key.lower()}"
                if logical_key in seen_logical_keys:
                    prev = seen_logical_keys[logical_key]
                    if match.relevance_score > prev.relevance_score:
                        valid_matches.remove(prev)
                        seen_logical_keys[logical_key] = match
                        valid_matches.append(match)
                else:
                    seen_logical_keys[logical_key] = match
                    valid_matches.append(match)

            # Primary: relevance_score DESC, Secondary: updated_at DESC, Tertiary: id ASC
            valid_matches.sort(
                key=lambda m: (-m.relevance_score, -m.record.updated_at, m.record.id)
            )

            # Diversity selection if result set is large and limit >= 3
            final_selection = self._apply_diversity_selection(valid_matches, safe_limit)[:safe_limit]
            if self.observability:
                from backend.storage.memory_audit_store import MemoryEventType
                if final_selection:
                    self.observability.record_event(
                        MemoryEventType.MEMORY_RETRIEVED,
                        metadata={
                            "selected_count": len(final_selection),
                            "max_score": final_selection[0].relevance_score,
                            "selected_ids": [m.record.id for m in final_selection],
                            "categories": [m.record.category.value for m in final_selection],
                        },
                    )
                else:
                    self.observability.record_event(MemoryEventType.MEMORY_RETRIEVAL_EMPTY)

            return final_selection

        except Exception as e:
            self.last_retrieval_failed = True
            if self.observability:
                from backend.storage.memory_audit_store import MemoryEventType
                self.observability.record_event(MemoryEventType.MEMORY_RETRIEVAL_FAILED, reason=str(e))
            logger.warning(f"Fail-safe memory retrieval fallback triggered: {e}")
            return []

    def _apply_diversity_selection(self, sorted_matches: List[MemoryMatch], limit: int) -> List[MemoryMatch]:
        """
        Ensures final results contain a balanced mix across categories if multiple distinct categories exist.
        Never displaces top 2 highest scoring matches.
        """
        if len(sorted_matches) <= limit or limit < 3:
            return sorted_matches[:limit]

        selected: List[MemoryMatch] = []
        selected_ids: Set[str] = set()

        # Always retain top 2 highest scoring matches regardless of category
        for m in sorted_matches[:2]:
            selected.append(m)
            selected_ids.add(m.record.id)

        # Retain top items from underrepresented categories if present
        existing_cats = {m.record.category for m in selected}
        for m in sorted_matches[2:]:
            if len(selected) >= limit:
                break
            if m.record.category not in existing_cats and m.record.id not in selected_ids:
                selected.append(m)
                selected_ids.add(m.record.id)
                existing_cats.add(m.record.category)

        # Fill remaining slots with highest remaining scored matches
        for m in sorted_matches:
            if len(selected) >= limit:
                break
            if m.record.id not in selected_ids:
                selected.append(m)
                selected_ids.add(m.record.id)

        selected.sort(key=lambda m: (-m.relevance_score, -m.record.updated_at, m.record.id))
        return selected

    def close(self) -> None:
        """Closes underlying memory service resources."""
        self.service.close()
