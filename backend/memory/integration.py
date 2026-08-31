import logging
from typing import List, Optional, Union

from backend.memory.boundaries import format_memory_context_untrusted
from backend.memory.models import MemoryCategory, MemoryRecord
from backend.memory.retrieval import MemoryMatch, MemoryRetriever

logger = logging.getLogger(__name__)


class MemoryContextBuilder:
    """
    Adapter between AgentCore and MemoryRetriever.
    Safely retrieves relevant memories, formats them as untrusted context blocks,
    and handles retrieval failures gracefully without interrupting agent execution.
    """

    def __init__(self, retriever: Optional[MemoryRetriever] = None):
        self.retriever = retriever

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
        if not self.retriever or not query or not isinstance(query, str) or not query.strip():
            return ""

        try:
            matches: List[MemoryMatch] = self.retriever.retrieve(
                query=query,
                category=category,
                limit=limit,
                min_confidence=min_confidence,
            )
            if not matches:
                return ""

            records = [match.record for match in matches]
            return format_memory_context_untrusted(records)
        except Exception as e:
            logger.warning(
                f"Memory retrieval failed in MemoryContextBuilder: {e}. Continuing without memory context."
            )
            return ""
