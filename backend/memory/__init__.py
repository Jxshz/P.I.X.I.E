"""
P.I.X.I.E. Memory Package
Defines persistent memory architecture, data models, categories, boundaries, and security rules.
"""

from backend.memory.models import (
    MemoryCategory,
    MemorySource,
    MemoryRecord,
    MemoryValidationError,
)
from backend.memory.boundaries import (
    MemoryBoundaryValidator,
    format_memory_context_untrusted,
    is_sensitive_content,
    contains_system_override_attempt,
    validate_metadata_json,
)
from backend.storage.memory_store import MemoryStore
from backend.memory.service import MemoryService
from backend.memory.retrieval import MemoryMatch, MemoryRetriever
from backend.memory.integration import MemoryContextBuilder

__all__ = [
    "MemoryCategory",
    "MemorySource",
    "MemoryRecord",
    "MemoryValidationError",
    "MemoryBoundaryValidator",
    "format_memory_context_untrusted",
    "is_sensitive_content",
    "contains_system_override_attempt",
    "validate_metadata_json",
    "MemoryStore",
    "MemoryService",
    "MemoryMatch",
    "MemoryRetriever",
    "MemoryContextBuilder",
]
