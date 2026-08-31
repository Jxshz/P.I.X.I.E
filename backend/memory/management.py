import json
import time
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
from backend.memory.service import MemoryService


class MemoryManagementAPI:
    """
    High-level application management API for P.I.X.I.E. persistent memory.
    Wraps MemoryService to provide structured listing, searching, CRUD, lifecycle management,
    inspection of metadata/provenance/expiration/supersession, and security sanitization.
    """

    def __init__(
        self,
        memory_service: Optional[MemoryService] = None,
        db_path: Optional[str] = None,
    ):
        self.memory_service = memory_service or MemoryService(db_path=db_path)
        self._observability_api = None

    @property
    def observability_api(self):
        """Lazy-loaded instance of MemoryObservabilityAPI."""
        if self._observability_api is None:
            from backend.memory.observability_api import MemoryObservabilityAPI
            obs_service = getattr(self.memory_service, "observability", None)
            self._observability_api = MemoryObservabilityAPI(observability_service=obs_service)
        return self._observability_api

    # ------------------------------------------------------------------
    # Query & Search Interface
    # ------------------------------------------------------------------

    def list_memories(
        self,
        category: Optional[Union[MemoryCategory, str]] = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> List[MemoryRecord]:
        """
        Lists persistent memory records with optional category and active filtering.
        """
        try:
            return self.memory_service.list_memories(
                category=category, active_only=active_only, limit=limit
            )
        except Exception:
            return []

    def search_memories(
        self,
        query: str,
        category: Optional[Union[MemoryCategory, str]] = None,
        limit: int = 20,
    ) -> List[MemoryRecord]:
        """
        Searches memories by matching query keywords in memory key or value.
        Only returns non-sensitive active memories.
        """
        if not query or not isinstance(query, str) or not query.strip():
            return []

        try:
            query_lower = query.strip().lower()
            all_memories = self.list_memories(category=category, active_only=True, limit=1000)

            matched: List[MemoryRecord] = []
            for rec in all_memories:
                if is_sensitive_content(rec.value) or contains_system_override_attempt(rec.value):
                    continue

                if query_lower in rec.key.lower() or query_lower in rec.value.lower():
                    matched.append(rec)
                    if len(matched) >= limit:
                        break

            return matched
        except Exception:
            return []

    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieves a memory record by ID."""
        if not memory_id or not isinstance(memory_id, str):
            return None
        try:
            return self.memory_service.get_memory(memory_id)
        except Exception:
            return None

    def get_memory_by_key(
        self,
        category: Union[MemoryCategory, str],
        key: str,
        active_only: bool = True,
    ) -> Optional[MemoryRecord]:
        """Retrieves a memory record by category and logical key."""
        if not key or not isinstance(key, str):
            return None
        try:
            return self.memory_service.get_memory_by_key(
                category=category, key=key, active_only=active_only
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # CRUD & Lifecycle Operations
    # ------------------------------------------------------------------

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
    ) -> MemoryRecord:
        """
        Creates and persists a new memory record after boundary and schema validation.
        """
        return self.memory_service.create_memory(
            category=category,
            key=key,
            value=value,
            source=source,
            confidence=confidence,
            metadata_json=metadata_json,
            expires_at=expires_at,
            is_active=is_active,
        )

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
        """
        return self.memory_service.update_memory(
            memory_id=memory_id,
            value=value,
            metadata_json=metadata_json,
            confidence=confidence,
            expires_at=expires_at,
            is_active=is_active,
        )

    def forget_memory(self, memory_id: str) -> bool:
        """
        Soft-deactivates (forgets) a memory record by ID (is_active = False).
        """
        return self.memory_service.forget_memory(memory_id)

    def forget_memory_by_key(
        self, category: Union[MemoryCategory, str], key: str
    ) -> bool:
        """
        Soft-deactivates a memory record by category and logical key.
        """
        return self.memory_service.forget_memory_by_key(category, key)

    def reactivate_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        """
        Reactivates a soft-deactivated memory record.
        """
        return self.memory_service.reactivate_memory(memory_id)

    def permanently_delete_memory(self, memory_id: str) -> bool:
        """
        Hard-deletes a memory record from storage permanently.
        """
        return self.memory_service.delete_memory(memory_id, hard_delete=True)

    # ------------------------------------------------------------------
    # Inspection & Audit Methods
    # ------------------------------------------------------------------

    def inspect_memory_metadata(self, memory_id: str) -> Dict[str, Any]:
        """
        Parses and returns the metadata dictionary for a memory record.
        """
        rec = self.get_memory_by_id(memory_id)
        if not rec or not rec.metadata_json:
            return {}

        try:
            parsed = json.loads(rec.metadata_json)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def inspect_memory_confidence_source(self, memory_id: str) -> Dict[str, Any]:
        """
        Inspects memory confidence, source provenance, and recency timestamps.
        """
        rec = self.get_memory_by_id(memory_id)
        if not rec:
            return {
                "exists": False,
                "confidence": 0.0,
                "source": None,
                "created_at": None,
                "updated_at": None,
            }

        return {
            "exists": True,
            "id": rec.id,
            "category": rec.category.value if hasattr(rec.category, "value") else str(rec.category),
            "key": rec.key,
            "confidence": rec.confidence,
            "source": rec.source.value if hasattr(rec.source, "value") else str(rec.source),
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
        }

    def inspect_expiration(self, memory_id: str) -> Dict[str, Any]:
        """
        Inspects memory expiration state, calculating remaining TTL seconds.
        """
        rec = self.get_memory_by_id(memory_id)
        if not rec:
            return {
                "exists": False,
                "expires_at": None,
                "is_expired": False,
                "remaining_seconds": None,
            }

        now = time.time()
        if rec.expires_at is None:
            return {
                "exists": True,
                "expires_at": None,
                "is_expired": False,
                "remaining_seconds": None,
            }

        is_expired = now >= rec.expires_at
        remaining = max(0.0, rec.expires_at - now) if not is_expired else 0.0

        return {
            "exists": True,
            "expires_at": rec.expires_at,
            "is_expired": is_expired,
            "remaining_seconds": remaining,
        }

    def inspect_supersession_state(self, memory_id: str) -> Dict[str, Any]:
        """
        Inspects whether a memory record is the active record for its logical key or superseded.
        """
        rec = self.get_memory_by_id(memory_id)
        if not rec:
            return {
                "exists": False,
                "is_active": False,
                "is_current_for_key": False,
                "latest_active_memory_id": None,
            }

        active_rec = self.get_memory_by_key(rec.category, rec.key, active_only=True)
        is_current = active_rec is not None and active_rec.id == rec.id

        return {
            "exists": True,
            "is_active": rec.is_active,
            "is_current_for_key": is_current,
            "latest_active_memory_id": active_rec.id if active_rec else None,
        }

    def count_memories(self, active_only: bool = True) -> int:
        """Returns the total count of stored memory records."""
        try:
            return self.memory_service.count_memories(active_only=active_only)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Privacy Controls Interface
    # ------------------------------------------------------------------

    def get_privacy_settings(self) -> Dict[str, Any]:
        """Returns current privacy settings and active memory metrics."""
        try:
            return self.memory_service.get_privacy_summary()
        except Exception:
            return {
                "memory_enabled": True,
                "capture_enabled": True,
                "retrieval_enabled": True,
                "active_memory_count": 0,
            }

    def set_memory_enabled(self, enabled: bool) -> bool:
        """Enables or disables global memory functionality."""
        try:
            return self.memory_service.set_memory_enabled(enabled)
        except Exception:
            return False

    def set_capture_enabled(self, enabled: bool) -> bool:
        """Enables or disables memory candidate capture."""
        try:
            return self.memory_service.set_capture_enabled(enabled)
        except Exception:
            return False

    def set_retrieval_enabled(self, enabled: bool) -> bool:
        """Enables or disables memory context retrieval."""
        try:
            return self.memory_service.set_retrieval_enabled(enabled)
        except Exception:
            return False

    def close(self) -> None:
        """Closes memory service and underlying storage connections."""
        if self.memory_service:
            self.memory_service.close()
