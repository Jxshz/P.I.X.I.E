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
    Application-level CRUD and Lifecycle management service for P.I.X.I.E. persistent memory.
    Wraps MemoryStore, enforcing model validation, security boundaries,
    and lifecycle state transitions while hiding storage internals.
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
        Preserves original created_at and does not implicitly reactivate expired memories.
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

    def supersede_memory(
        self,
        category: Union[MemoryCategory, str],
        key: str,
        value: str,
        source: Union[MemorySource, str] = MemorySource.EXPLICIT_USER_INPUT,
        confidence: float = 1.0,
        metadata_json: Optional[str] = None,
        expires_at: Optional[float] = None,
    ) -> MemoryRecord:
        """
        Supersedes an existing logical memory for (category, key) with a new value.
        Updates the existing logical key record atomically if present, or creates a new active memory.
        """
        existing = self.get_memory_by_key(category, key, active_only=False)
        if existing:
            updated = self.update_memory(
                memory_id=existing.id,
                value=value,
                metadata_json=metadata_json,
                confidence=confidence,
                expires_at=expires_at,
                is_active=True, # Superseding explicitly activates the new memory state
            )
            if updated is None:
                raise MemoryValidationError(f"Failed to supersede memory with key: {key}")
            return updated
        else:
            return self.create_memory(
                category=category,
                key=key,
                value=value,
                source=source,
                confidence=confidence,
                metadata_json=metadata_json,
                expires_at=expires_at,
                is_active=True,
            )

    def forget_memory(self, memory_id: str) -> bool:
        """
        Soft-deactivates (forgets) a memory record by ID (is_active = False).
        Returns True if memory existed and was deactivated, False otherwise.
        """
        if not memory_id or not isinstance(memory_id, str):
            return False
        return self.store.delete_memory(memory_id, hard_delete=False)

    def forget_memory_by_key(self, category: Union[MemoryCategory, str], key: str) -> bool:
        """
        Soft-deactivates an active memory record by category and logical key.
        Returns True if record existed and was deactivated, False otherwise.
        """
        existing = self.get_memory_by_key(category, key, active_only=True)
        if not existing:
            return False
        return self.forget_memory(existing.id)

    def reactivate_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        """
        Explicitly reactivates a forgotten/inactive memory record (is_active = True).
        Re-validates record boundaries before saving.
        Raises MemoryValidationError if memory has expired.
        """
        existing = self.get_memory(memory_id)
        if not existing:
            return None

        now = time.time()
        if existing.expires_at is not None and existing.expires_at <= now:
            raise MemoryValidationError("Cannot reactivate an expired memory.")

        return self.update_memory(memory_id=memory_id, is_active=True)

    def prune_expired_memories(self, hard_delete: bool = False) -> int:
        """
        Prunes (deactivates or permanently deletes) memories whose expires_at timestamp has passed.
        Returns the total count of pruned records.
        """
        return self.store.prune_expired_memories(hard_delete=hard_delete)

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
