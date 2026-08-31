import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from backend.memory.boundaries import (
    contains_system_override_attempt,
    is_sensitive_content,
)
from backend.memory.observability import (
    MemoryObservabilityService,
    sanitize_audit_text,
)
from backend.storage.memory_audit_store import AuditEvent, MemoryEventType

logger = logging.getLogger(__name__)


@dataclass
class ObservabilityEventDTO:
    """
    Sanitized, read-only DTO for external inspection of memory subsystem audit events.
    Guarantees no raw memory values, prompts, assistant responses, secrets, or file paths are exposed.
    """

    event_id: str
    event_type: str
    timestamp: float
    memory_id: Optional[str] = None
    category: Optional[str] = None
    key: Optional[str] = None
    result: Optional[str] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes DTO into a plain dictionary."""
        return asdict(self)


@dataclass
class MemoryLifecycleStep:
    """
    Sanitized representation of a single memory lifecycle state transition.
    """

    event_type: str
    timestamp: float
    result: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes lifecycle step to dictionary."""
        return asdict(self)


class MemoryObservabilityAPI:
    """
    High-level, application-level read-only observability API for P.I.X.I.E. persistent memory.
    Wraps MemoryObservabilityService with strict sanitization, fail-safe boundaries,
    and structured aggregation statistics.
    """

    def __init__(
        self,
        observability_service: Optional[MemoryObservabilityService] = None,
        db_path: str = ":memory:",
    ):
        if observability_service:
            self.service = observability_service
        else:
            self.service = MemoryObservabilityService(db_path=db_path)

    def _sanitize_string(self, text: Optional[str]) -> Optional[str]:
        """Sanitizes text fields to remove sensitive credentials, prompts, and file paths."""
        if not text or not isinstance(text, str):
            return None
        cleaned = text.strip()
        if any(term in cleaned.lower() for term in ["sqlite", ".db", "traceback", "exception", "/users/", "c:\\"]):
            return "Sanitized system event detail."
        if is_sensitive_content(cleaned) or contains_system_override_attempt(cleaned):
            return "[REDACTED_SENSITIVE_CONTENT]"
        return cleaned[:256]

    def _sanitize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Sanitizes metadata dictionary fields cleanly."""
        if not metadata or not isinstance(metadata, dict):
            return {}
        safe_meta: Dict[str, Any] = {}
        for k, v in metadata.items():
            if k.lower() in ("value", "raw_value", "prompt", "response", "secret", "password", "file_path", "db_path"):
                safe_meta[k] = "[REDACTED]"
            elif isinstance(v, str):
                safe_meta[k] = self._sanitize_string(v)
            else:
                safe_meta[k] = v
        return safe_meta

    def _event_to_dto(self, event: AuditEvent) -> ObservabilityEventDTO:
        """Converts an AuditEvent to a safe ObservabilityEventDTO."""
        return ObservabilityEventDTO(
            event_id=event.event_id,
            event_type=event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
            timestamp=event.timestamp,
            memory_id=event.memory_id,
            category=event.category,
            key=self._sanitize_string(event.key),
            result=self._sanitize_string(event.result),
            reason=self._sanitize_string(event.reason),
            metadata=self._sanitize_metadata(event.metadata),
        )

    def get_recent_events(
        self,
        limit: int = 50,
        event_type: Optional[Union[MemoryEventType, str]] = None,
        category: Optional[str] = None,
        memory_id: Optional[str] = None,
        session_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves recent audit events matching optional filters, sorted by timestamp DESC.
        Read-only and fail-safe.
        """
        try:
            safe_limit = max(1, min(limit, 500))
            events = self.service.store.get_events_filtered(
                event_type=event_type,
                memory_id=memory_id,
                session_id=session_id,
                category=category,
                start_time=start_time,
                end_time=end_time,
                limit=safe_limit,
                ascending=False,
            )
            return [self._event_to_dto(e).to_dict() for e in events]
        except Exception as e:
            logger.warning(f"Fail-safe ObservabilityAPI exception: {e}")
            return []

    def get_events_for_memory(self, memory_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves audit events associated with a specific memory_id."""
        if not memory_id:
            return []
        return self.get_recent_events(limit=limit, memory_id=memory_id)

    def get_events_for_session(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves audit events associated with a specific session_id."""
        if not session_id:
            return []
        return self.get_recent_events(limit=limit, session_id=session_id)

    def get_events_by_type(
        self, event_type: Union[MemoryEventType, str], limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieves audit events matching a specific event type."""
        return self.get_recent_events(limit=limit, event_type=event_type)

    def get_event_count(self, event_type: Optional[Union[MemoryEventType, str]] = None) -> int:
        """Returns count of audit events matching optional event type filter."""
        try:
            return self.service.count_events(event_type=event_type)
        except Exception:
            return 0

    def get_memory_activity(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves general memory management activity events."""
        return self.get_recent_events(limit=limit)

    def get_lifecycle_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the chronological lifecycle transition sequence for a specific memory record.
        Returned in timestamp ASC order.
        """
        if not memory_id:
            return []
        try:
            events = self.service.store.get_events_filtered(
                memory_id=memory_id,
                limit=500,
                ascending=True,
            )
            lifecycle_types = {
                MemoryEventType.MEMORY_CREATED.value,
                MemoryEventType.MEMORY_UPDATED.value,
                MemoryEventType.MEMORY_SUPERSEDED.value,
                MemoryEventType.MEMORY_FORGOTTEN.value,
                MemoryEventType.MEMORY_REACTIVATED.value,
                MemoryEventType.MEMORY_DELETED.value,
                MemoryEventType.MEMORY_EXPIRED.value,
            }
            steps: List[Dict[str, Any]] = []
            for e in events:
                evt_str = e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type)
                if evt_str in lifecycle_types:
                    step = MemoryLifecycleStep(
                        event_type=evt_str,
                        timestamp=e.timestamp,
                        result=self._sanitize_string(e.result),
                        reason=self._sanitize_string(e.reason),
                    )
                    steps.append(step.to_dict())
            return steps
        except Exception as e:
            logger.warning(f"Fail-safe ObservabilityAPI exception: {e}")
            return []

    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """
        Calculates aggregate retrieval performance metrics.
        """
        try:
            retrieved = self.service.count_events(MemoryEventType.MEMORY_RETRIEVED)
            empty = self.service.count_events(MemoryEventType.MEMORY_RETRIEVAL_EMPTY)
            failed = self.service.count_events(MemoryEventType.MEMORY_RETRIEVAL_FAILED)
            total = retrieved + empty + failed
            success_rate = (retrieved / total) if total > 0 else 0.0

            return {
                "total_retrievals": total,
                "successful_retrievals": retrieved,
                "empty_retrievals": empty,
                "failed_retrievals": failed,
                "success_rate": round(success_rate, 4),
            }
        except Exception:
            return {
                "total_retrievals": 0,
                "successful_retrievals": 0,
                "empty_retrievals": 0,
                "failed_retrievals": 0,
                "success_rate": 0.0,
            }

    def get_security_event_statistics(self) -> Dict[str, Any]:
        """
        Calculates aggregate security rejection metrics without exposing malicious payloads.
        """
        try:
            sec_events = self.service.store.get_events_by_type(MemoryEventType.MEMORY_SECURITY_REJECTED, limit=100)
            dtos = [self._event_to_dto(e).to_dict() for e in sec_events]
            return {
                "total_security_rejections": len(sec_events),
                "recent_rejection_timestamps": [e.timestamp for e in sec_events[:10]],
                "events": dtos,
            }
        except Exception:
            return {
                "total_security_rejections": 0,
                "recent_rejection_timestamps": [],
                "events": [],
            }

    def get_privacy_event_statistics(self) -> Dict[str, Any]:
        """
        Calculates aggregate privacy state transition metrics.
        """
        try:
            enabled_cnt = self.service.count_events(MemoryEventType.PRIVACY_ENABLED)
            disabled_cnt = self.service.count_events(MemoryEventType.PRIVACY_DISABLED)
            changed_cnt = self.service.count_events(MemoryEventType.PRIVACY_SETTING_CHANGED)
            retention_cnt = self.service.count_events(MemoryEventType.RETENTION_CHECKED)

            return {
                "total_privacy_events": enabled_cnt + disabled_cnt + changed_cnt + retention_cnt,
                "privacy_enabled_count": enabled_cnt,
                "privacy_disabled_count": disabled_cnt,
                "privacy_setting_changed_count": changed_cnt,
                "retention_checked_count": retention_cnt,
            }
        except Exception:
            return {
                "total_privacy_events": 0,
                "privacy_enabled_count": 0,
                "privacy_disabled_count": 0,
                "privacy_setting_changed_count": 0,
                "retention_checked_count": 0,
            }

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns a comprehensive aggregate summary of the memory subsystem observability state.
        """
        try:
            total_events = self.service.count_events()
            retrieval_stats = self.get_retrieval_statistics()
            security_stats = self.get_security_event_statistics()
            privacy_stats = self.get_privacy_event_statistics()

            return {
                "total_events": total_events,
                "retrieval_statistics": retrieval_stats,
                "security_statistics": security_stats,
                "privacy_statistics": privacy_stats,
            }
        except Exception:
            return {
                "total_events": 0,
                "retrieval_statistics": self.get_retrieval_statistics(),
                "security_statistics": self.get_security_event_statistics(),
                "privacy_statistics": self.get_privacy_event_statistics(),
            }

    def close(self) -> None:
        """Closes underlying audit service resources."""
        if self.service:
            self.service.close()
