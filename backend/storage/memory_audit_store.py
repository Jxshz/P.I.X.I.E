import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class MemoryEventType(str, Enum):
    """Event types for memory subsystem audit logging."""
    MEMORY_CREATED = "MEMORY_CREATED"
    MEMORY_UPDATED = "MEMORY_UPDATED"
    MEMORY_SUPERSEDED = "MEMORY_SUPERSEDED"
    MEMORY_FORGOTTEN = "MEMORY_FORGOTTEN"
    MEMORY_REACTIVATED = "MEMORY_REACTIVATED"
    MEMORY_DELETED = "MEMORY_DELETED"
    MEMORY_EXPIRED = "MEMORY_EXPIRED"
    MEMORY_REJECTED = "MEMORY_REJECTED"
    MEMORY_RETRIEVED = "MEMORY_RETRIEVED"
    MEMORY_RETRIEVAL_EMPTY = "MEMORY_RETRIEVAL_EMPTY"
    MEMORY_RETRIEVAL_FAILED = "MEMORY_RETRIEVAL_FAILED"
    MEMORY_CANDIDATE_EXTRACTED = "MEMORY_CANDIDATE_EXTRACTED"
    MEMORY_CONSENT_PENDING = "MEMORY_CONSENT_PENDING"
    MEMORY_CONSENT_APPROVED = "MEMORY_CONSENT_APPROVED"
    MEMORY_CONSENT_REJECTED = "MEMORY_CONSENT_REJECTED"
    MEMORY_CONFLICT_DETECTED = "MEMORY_CONFLICT_DETECTED"
    MEMORY_CONFLICT_RESOLVED = "MEMORY_CONFLICT_RESOLVED"
    MEMORY_SECURITY_REJECTED = "MEMORY_SECURITY_REJECTED"
    PRIVACY_ENABLED = "PRIVACY_ENABLED"
    PRIVACY_DISABLED = "PRIVACY_DISABLED"
    PRIVACY_SETTING_CHANGED = "PRIVACY_SETTING_CHANGED"
    RETENTION_CHECKED = "RETENTION_CHECKED"


@dataclass
class AuditEvent:
    """
    Data model for structured, privacy-preserving memory audit events.
    Never contains raw memory values, credentials, secrets, or full prompt payloads.
    """
    event_id: str
    event_type: MemoryEventType
    timestamp: float
    memory_id: Optional[str] = None
    category: Optional[str] = None
    key: Optional[str] = None
    source: Optional[str] = None
    confidence: Optional[float] = None
    result: Optional[str] = None
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if isinstance(self.event_type, str) and not isinstance(self.event_type, MemoryEventType):
            self.event_type = MemoryEventType(self.event_type)


class MemoryAuditStore:
    """
    Dedicated, transactional SQLite audit store for memory events.
    Isolated from sessions.db and memory.db.
    Fail-safe: Storage failures are caught silently to prevent interrupting core operations.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        try:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_audit_events (
                        id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        memory_id TEXT,
                        category TEXT,
                        key TEXT,
                        source TEXT,
                        confidence REAL,
                        result TEXT,
                        reason TEXT,
                        metadata_json TEXT
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_event_type ON memory_audit_events(event_type);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_memory_id ON memory_audit_events(memory_id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON memory_audit_events(timestamp DESC);"
                )
        except Exception as e:
            logger.warning(f"MemoryAuditStore DB initialization error: {e}")

    def append_event(self, event: AuditEvent) -> None:
        """
        Appends a sanitized AuditEvent to storage transactionally.
        Fail-safe: Errors are logged as warnings and swallowed.
        """
        try:
            conn = self._get_connection()
            metadata_json = json.dumps(event.metadata) if event.metadata else None
            with conn:
                conn.execute(
                    """
                    INSERT INTO memory_audit_events (
                        id, event_type, timestamp, memory_id, category, key, source, confidence, result, reason, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.event_type.value,
                        event.timestamp,
                        event.memory_id,
                        event.category,
                        event.key,
                        event.source,
                        event.confidence,
                        event.result,
                        event.reason,
                        metadata_json,
                    ),
                )
        except Exception as e:
            logger.warning(f"Failed to record audit event {event.event_type}: {e}")

    def get_recent_events(self, limit: int = 50) -> List[AuditEvent]:
        """Retrieves recent audit events sorted by timestamp DESC."""
        safe_limit = max(1, min(limit, 500))
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM memory_audit_events ORDER BY timestamp DESC, id ASC LIMIT ?",
                (safe_limit,),
            )
            rows = cursor.fetchall()
            return [self._row_to_event(r) for r in rows]
        except Exception as e:
            logger.warning(f"Failed to retrieve recent audit events: {e}")
            return []

    def get_events_for_memory(self, memory_id: str, limit: int = 50) -> List[AuditEvent]:
        """Retrieves audit events for a specific memory_id sorted by timestamp DESC."""
        if not memory_id:
            return []
        safe_limit = max(1, min(limit, 500))
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM memory_audit_events WHERE memory_id = ? ORDER BY timestamp DESC, id ASC LIMIT ?",
                (memory_id, safe_limit),
            )
            rows = cursor.fetchall()
            return [self._row_to_event(r) for r in rows]
        except Exception as e:
            logger.warning(f"Failed to retrieve audit events for memory {memory_id}: {e}")
            return []

    def get_events_by_type(
        self, event_type: Union[MemoryEventType, str], limit: int = 50
    ) -> List[AuditEvent]:
        """Retrieves audit events matching event_type sorted by timestamp DESC."""
        safe_type = event_type.value if isinstance(event_type, MemoryEventType) else str(event_type)
        safe_limit = max(1, min(limit, 500))
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM memory_audit_events WHERE event_type = ? ORDER BY timestamp DESC, id ASC LIMIT ?",
                (safe_type, safe_limit),
            )
            rows = cursor.fetchall()
            return [self._row_to_event(r) for r in rows]
        except Exception as e:
            logger.warning(f"Failed to retrieve audit events by type {safe_type}: {e}")
            return []

    def count_events(self, event_type: Optional[Union[MemoryEventType, str]] = None) -> int:
        """Counts stored audit events, optionally filtered by event_type."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            if event_type:
                safe_type = event_type.value if isinstance(event_type, MemoryEventType) else str(event_type)
                cursor.execute(
                    "SELECT COUNT(*) FROM memory_audit_events WHERE event_type = ?", (safe_type,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM memory_audit_events")
            row = cursor.fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.warning(f"Failed to count audit events: {e}")
            return 0

    def get_events_filtered(
        self,
        event_type: Optional[Union[MemoryEventType, str]] = None,
        memory_id: Optional[str] = None,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 50,
        ascending: bool = False,
    ) -> List[AuditEvent]:
        """Retrieves audit events using parameterized SQL filtering."""
        safe_limit = max(1, min(limit, 500))
        clauses: List[str] = []
        params: List[Any] = []

        if event_type:
            safe_type = event_type.value if isinstance(event_type, MemoryEventType) else str(event_type)
            clauses.append("event_type = ?")
            params.append(safe_type)

        if memory_id:
            clauses.append("memory_id = ?")
            params.append(str(memory_id))

        if category:
            clauses.append("category = ?")
            params.append(str(category))

        if start_time is not None:
            clauses.append("timestamp >= ?")
            params.append(float(start_time))

        if end_time is not None:
            clauses.append("timestamp <= ?")
            params.append(float(end_time))

        where_stmt = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        order_dir = "ASC" if ascending else "DESC"
        sql = f"SELECT * FROM memory_audit_events{where_stmt} ORDER BY timestamp {order_dir}, id ASC LIMIT ?"
        params.append(safe_limit)

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            events = [self._row_to_event(r) for r in rows]

            if session_id:
                events = [
                    e for e in events
                    if e.metadata and isinstance(e.metadata, dict) and str(e.metadata.get("session_id")) == str(session_id)
                ]

            return events
        except Exception as e:
            logger.warning(f"Failed to query filtered audit events: {e}")
            return []

    def _row_to_event(self, row: sqlite3.Row) -> AuditEvent:
        metadata = None
        if row["metadata_json"]:
            try:
                metadata = json.loads(row["metadata_json"])
            except Exception:
                pass

        return AuditEvent(
            event_id=row["id"],
            event_type=MemoryEventType(row["event_type"]),
            timestamp=row["timestamp"],
            memory_id=row["memory_id"],
            category=row["category"],
            key=row["key"],
            source=row["source"],
            confidence=row["confidence"],
            result=row["result"],
            reason=row["reason"],
            metadata=metadata,
        )

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
