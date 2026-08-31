import logging
from typing import Any, Dict, List, Optional, Union

from backend.memory.boundaries import format_memory_context_untrusted
from backend.memory.models import MemoryCategory, MemoryRecord
from backend.memory.retrieval import MemoryMatch, MemoryRetriever

logger = logging.getLogger(__name__)


class MemoryContextBuilder:
    """
    Adapter between AgentCore and MemoryRetriever.
    Safely retrieves relevant memories, formats them as untrusted context blocks,
    exposes observability metrics, and handles retrieval failures gracefully without interrupting agent execution.
    """

    def __init__(self, retriever: Optional[MemoryRetriever] = None):
        self.retriever = retriever
        self.last_retrieval_stats: Dict[str, Any] = {
            "retrieved": False,
            "count": 0,
            "relevance_scores": [],
            "categories": [],
            "memory_ids": [],
            "retrieval_failed": False,
        }

    def build_memory_context(
        self,
        query: str,
        category: Optional[Union[MemoryCategory, str]] = None,
        limit: int = 5,
        min_confidence: float = 0.0,
    ) -> str:
        """
        Retrieves relevant memories for the query string and formats them as untrusted context.

        Fail-safe invariant: Returns empty string on any retrieval error or missing retriever.
        """
        self.last_retrieval_stats = {
            "retrieved": False,
            "count": 0,
            "relevance_scores": [],
            "categories": [],
            "memory_ids": [],
            "retrieval_failed": False,
        }

        if not self.retriever or not query or not isinstance(query, str) or not query.strip():
            return ""

        try:
            matches: List[MemoryMatch] = self.retriever.retrieve(
                query=query,
                category=category,
                limit=limit,
                min_confidence=min_confidence,
            )
            if getattr(self.retriever, "last_retrieval_failed", False):
                self.last_retrieval_stats["retrieval_failed"] = True
                return ""

            if not matches:
                return ""

            # Record observability statistics (without logging sensitive memory content)
            self.last_retrieval_stats = {
                "retrieved": True,
                "count": len(matches),
                "relevance_scores": [m.relevance_score for m in matches],
                "categories": [m.record.category.value for m in matches],
                "memory_ids": [m.record.id for m in matches],
                "retrieval_failed": False,
            }

            records = [match.record for match in matches]
            return format_memory_context_untrusted(records)
        except Exception as e:
            logger.warning(
                f"Memory retrieval failed in MemoryContextBuilder: {e}. Continuing without memory context."
            )
            self.last_retrieval_stats["retrieval_failed"] = True
            return ""
