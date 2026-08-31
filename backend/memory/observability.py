import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from backend.memory.boundaries import contains_system_override_attempt, is_sensitive_content
from backend.storage.memory_audit_store import AuditEvent, MemoryAuditStore, MemoryEventType

logger = logging.getLogger(__name__)


def sanitize_audit_text(text: Optional[str]) -> Optional[str]:
    """
    Sanitizes audit text fields to prevent sensitive secrets or raw values from reaching logs/DB.
    Returns None if text is empty, '[REDACTED_SENSITIVE_CONTENT]' if sensitive/injection content exists,
    or trimmed safe text otherwise.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return None

    cleaned = text.strip()
    if is_sensitive_content(cleaned) or contains_system_override_attempt(cleaned):
        return "[REDACTED_SENSITIVE_CONTENT]"

    # Truncate safe reason/result text to 256 chars maximum to prevent accidental payload dumps
    return cleaned[:256]


class MemoryObservabilityService:
    """
    High-level, privacy-preserving, read-only audit service for the P.I.X.I.E. memory subsystem.
    Applies strict text sanitization and fail-safe exception boundaries.
    """

    def __init__(
        self,
        audit_store: Optional[MemoryAuditStore] = None,
        db_path: str = ":memory:",
    ):
        if audit_store:
            self.store = audit_store
        else:
            self.store = MemoryAuditStore(db_path=db_path)

    def record_event(
        self,
        event_type: MemoryEventType,
        memory_id: Optional[str] = None,
        category: Optional[str] = None,
        key: Optional[str] = None,
        source: Optional[str] = None,
        confidence: Optional[float] = None,
        result: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AuditEvent]:
        """
        Constructs, sanitizes, and records an AuditEvent transactionally.
        Fail-safe invariant: Swallows storage failures to ensure memory operations never break.
        """
        try:
            safe_key = sanitize_audit_text(key)
            safe_result = sanitize_audit_text(result)
            safe_reason = sanitize_audit_text(reason)

            # Ensure metadata does not contain raw values or sensitive keys
            safe_metadata = None
            if metadata and isinstance(metadata, dict):
                safe_metadata = {}
                for k, v in metadata.items():
                    if k.lower() in ("value", "raw_value", "prompt", "response", "secret", "password"):
                        safe_metadata[k] = "[REDACTED]"
                    elif isinstance(v, str) and (is_sensitive_content(v) or contains_system_override_attempt(v)):
                        safe_metadata[k] = "[REDACTED_SENSITIVE_CONTENT]"
                    else:
                        safe_metadata[k] = v

            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=time.time(),
                memory_id=memory_id,
                category=category,
                key=safe_key,
                source=source,
                confidence=confidence,
                result=safe_result,
                reason=safe_reason,
                metadata=safe_metadata,
            )
            self.store.append_event(event)
            return event
        except Exception as e:
            logger.warning(f"Fail-safe observability exception: {e}")
            return None

    def get_recent_events(self, limit: int = 50) -> List[AuditEvent]:
        """Retrieves recent audit events sorted by timestamp DESC. Read-only."""
        return self.store.get_recent_events(limit=limit)

    def get_events_for_memory(self, memory_id: str, limit: int = 50) -> List[AuditEvent]:
        """Retrieves audit events for a specific memory_id. Read-only."""
        return self.store.get_events_for_memory(memory_id=memory_id, limit=limit)

    def get_events_by_type(
        self, event_type: Union[MemoryEventType, str], limit: int = 50
    ) -> List[AuditEvent]:
        """Retrieves audit events matching event_type. Read-only."""
        return self.store.get_events_by_type(event_type=event_type, limit=limit)

    def count_events(self, event_type: Optional[Union[MemoryEventType, str]] = None) -> int:
        """Counts stored audit events. Read-only."""
        return self.store.count_events(event_type=event_type)

    def close(self) -> None:
        """Closes the underlying audit store."""
        self.store.close()
