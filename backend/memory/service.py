import time
import uuid
from typing import Any, Dict, List, Optional, Union

from backend.memory.boundaries import MemoryBoundaryValidator
from backend.memory.models import (
    MemoryCategory,
    MemoryRecord,
    MemorySource,
    MemoryValidationError,
)
from backend.storage.memory_store import MemoryStore


class MemoryService:
    """
    Application-level CRUD service for P.I.X.I.E. persistent memory.
    Wraps MemoryStore, enforcing model validation, security boundaries,
    and business logic while hiding storage internals.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        memory_store: Optional[MemoryStore] = None,
    ):
        self.store = memory_store or MemoryStore(db_path=db_path)

    def create_memory(
        self,
        category: Union[MemoryCategory, str],
        key: str,
        value: str,
        source: Union[MemorySource, str] = MemorySource.EXPLICIT_USER_INPUT,
        confidence: float = 1.0,
        metadata_json: Optional[str] = None,
        expires_at: Optional[float] = None,
        is_active: bool = True,
        memory_id: Optional[str] = None,
    ) -> MemoryRecord:
        """
        Creates and persists a new memory record after boundary and schema validation.
        """
        try:
            cat_enum = category if isinstance(category, MemoryCategory) else MemoryCategory(category)
        except ValueError as e:
            raise MemoryValidationError(f"Invalid memory category: {category}") from e

        try:
            src_enum = source if isinstance(source, MemorySource) else MemorySource(source)
        except ValueError as e:
            raise MemoryValidationError(f"Invalid memory source: {source}") from e

        now = time.time()
        record = MemoryRecord(
            id=memory_id or str(uuid.uuid4()),
            category=cat_enum,
            key=key,
            value=value,
            source=src_enum,
            confidence=confidence,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            is_active=is_active,
            metadata_json=metadata_json,
        )

        # Store performs record and boundary validation before writing
        return self.store.save_memory(record)

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieves a memory record by ID."""
        if not memory_id or not isinstance(memory_id, str):
            return None
        return self.store.get_memory(memory_id)

    def get_memory_by_key(
        self, category: Union[MemoryCategory, str], key: str, active_only: bool = True
    ) -> Optional[MemoryRecord]:
        """Retrieves a memory record by its logical category and key."""
        if not key or not isinstance(key, str):
            return None
        return self.store.get_memory_by_key(category, key, active_only=active_only)

    def list_memories(
        self,
        category: Optional[Union[MemoryCategory, str]] = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> List[MemoryRecord]:
        """Lists stored memories with optional category/active filtering."""
        return self.store.list_memories(category=category, active_only=active_only, limit=limit)

    def update_memory(
        self,
        memory_id: str,
        value: Optional[str] = None,
        metadata_json: Optional[str] = None,
        confidence: Optional[float] = None,
        expires_at: Optional[float] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[MemoryRecord]:
        """
        Updates fields of an existing memory record.
        Re-validates updated values against schema and security boundaries before saving.
        Returns the updated MemoryRecord, or None if memory_id does not exist.
        """
        existing = self.get_memory(memory_id)
        if not existing:
            return None

        new_value = value if value is not None else existing.value
        new_metadata_json = metadata_json if metadata_json is not None else existing.metadata_json
        new_confidence = confidence if confidence is not None else existing.confidence
        new_expires_at = expires_at if expires_at is not None else existing.expires_at
        new_is_active = is_active if is_active is not None else existing.is_active

        updated_record = MemoryRecord(
            id=existing.id,
            category=existing.category,
            key=existing.key,
            value=new_value,
            source=existing.source,
            confidence=new_confidence,
            created_at=existing.created_at,
            updated_at=time.time(),
            expires_at=new_expires_at,
            is_active=new_is_active,
            metadata_json=new_metadata_json,
        )

        return self.store.save_memory(updated_record)

    def delete_memory(self, memory_id: str, hard_delete: bool = True) -> bool:
        """
        Deletes or deactivates a memory record by ID.
        Returns True if successful, False if record did not exist.
        """
        if not memory_id or not isinstance(memory_id, str):
            return False
        return self.store.delete_memory(memory_id, hard_delete=hard_delete)

    def count_memories(self, active_only: bool = True) -> int:
        """Returns count of stored memory records."""
        return self.store.count_memories(active_only=active_only)

    def close(self) -> None:
        """Closes underlying storage resources cleanly."""
        self.store.close()
