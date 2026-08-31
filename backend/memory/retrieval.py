import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union

from backend.memory.boundaries import is_sensitive_content
from backend.memory.models import MemoryCategory, MemoryRecord
from backend.memory.service import MemoryService
from backend.storage.memory_store import MemoryStore

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
    matched_signals: List[str]


def _tokenize(text: str) -> Set[str]:
    """Tokenizes and normalizes text into a set of lowercased alphanumeric words."""
    if not text:
        return set()
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return {word for word in cleaned.split() if len(word) > 1}


class MemoryRetriever:
    """
    Deterministic, relevance-aware read-only memory retrieval engine.
    Calculates lexical overlap, key matches, confidence, and recency signals
    to rank candidate memories deterministically without LLM network overhead.
    """

    def __init__(
        self,
        memory_service: Optional[MemoryService] = None,
        memory_store: Optional[MemoryStore] = None,
        db_path: Optional[str] = None,
    ):
        if memory_service:
            self.service = memory_service
        elif memory_store:
            self.service = MemoryService(memory_store=memory_store)
        else:
            self.service = MemoryService(db_path=db_path)

    def calculate_relevance(
        self,
        query: str,
        record: MemoryRecord,
        category_filter: Optional[Union[MemoryCategory, str]] = None,
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

        # 1. Exact key match or key substring match
        if query_clean == key_clean:
            score += 3.0
            matched_signals.append("exact_key_match")
        elif key_clean in query_clean or query_clean in key_clean:
            score += 1.5
            matched_signals.append("partial_key_match")

        # 2. Token overlap matching
        key_overlap = query_tokens.intersection(key_tokens)
        val_overlap = query_tokens.intersection(val_tokens)

        if key_overlap:
            key_token_score = len(key_overlap) * 1.0
            score += key_token_score
            matched_signals.append(f"key_tokens_matched:{','.join(sorted(key_overlap))}")

        if val_overlap:
            val_token_score = len(val_overlap) * 0.5
            score += val_token_score
            matched_signals.append(f"value_tokens_matched:{','.join(sorted(val_overlap))}")

        # 3. Substring / phrase match in value
        if len(query_clean) >= 3 and query_clean in val_clean:
            score += 1.0
            matched_signals.append("phrase_match_in_value")

        # 4. Category match bonus (if category filter was requested)
        if category_filter:
            cat_str = category_filter.value if isinstance(category_filter, MemoryCategory) else str(category_filter)
            if record.category.value == cat_str:
                score += 0.5
                matched_signals.append(f"category_match:{cat_str}")

        # 5. Confidence weight (0.0 to 0.5)
        conf_weight = record.confidence * 0.5
        score += conf_weight

        # 6. Recency bonus (small tie-breaker, max 0.2 score)
        age_seconds = max(0.0, now - record.updated_at)
        # Decays over 30 days (2,592,000 seconds)
        recency_factor = max(0.0, 1.0 - (age_seconds / 2592000.0))
        recency_bonus = recency_factor * 0.2
        score += recency_bonus

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

        Enforces hard bounds:
        - Query truncated to MAX_QUERY_LENGTH
        - Result count capped to MAX_RETRIEVAL_LIMIT
        - Read-only: does not alter stored records
        - Defense-in-depth: excludes records containing sensitive secrets
        """
        if not query or not isinstance(query, str) or not query.strip():
            return []

        # Enforce bounds on input parameters
        safe_query = query[:MAX_QUERY_LENGTH].strip()
        safe_limit = max(1, min(limit, MAX_RETRIEVAL_LIMIT))

        # Fetch candidate active memories from MemoryService
        candidates = self.service.list_memories(category=category, active_only=True, limit=200)

        now = time.time()
        valid_matches: List[MemoryMatch] = []

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

            match = self.calculate_relevance(safe_query, record, category_filter=category, now=now)

            # Exclude matches below minimum relevance score
            if match.relevance_score >= min_score and match.matched_signals:
                valid_matches.append(match)

        # Deterministic sorting:
        # Primary: relevance_score DESC
        # Secondary: updated_at DESC
        # Tertiary: id ASC (alphabetical string tie-breaker)
        valid_matches.sort(
            key=lambda m: (-m.relevance_score, -m.record.updated_at, m.record.id)
        )

        return valid_matches[:safe_limit]

    def close(self) -> None:
        """Closes underlying memory service resources."""
        self.service.close()
