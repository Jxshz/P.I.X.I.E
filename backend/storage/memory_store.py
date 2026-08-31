import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.memory.boundaries import MemoryBoundaryValidator
from backend.memory.models import (
    MemoryCategory,
    MemoryRecord,
    MemorySource,
    MemoryValidationError,
)


class MemoryStore:
    """
    SQLite persistence layer for P.I.X.I.E. persistent long-term memory.
    Decoupled from SessionStore and session history.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path(__file__).parent / "memory.db")
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._is_memory = db_path == ":memory:" or "mode=memory" in db_path

        if self._is_memory:
            # Shared connection for in-memory database across threads in testing
            self._shared_conn = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
            self._shared_conn.row_factory = sqlite3.Row
            self._init_db(self._shared_conn)
        else:
            self._init_db(self._get_connection())

    def _get_connection(self) -> sqlite3.Connection:
        if self._is_memory:
            return self._shared_conn

        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self, conn: sqlite3.Connection):
        with self._lock:
            if not self._is_memory:
                conn.execute("PRAGMA journal_mode=WAL")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT
                )
                """
            )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_category_key ON memories(category, key)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_category_active ON memories(category, is_active)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories(expires_at)"
            )
            conn.commit()

    def _record_from_row(self, row: sqlite3.Row) -> MemoryRecord:
        data = dict(row)
        return MemoryRecord(
            id=data["id"],
            category=MemoryCategory(data["category"]),
            key=data["key"],
            value=data["value"],
            source=MemorySource(data["source"]),
            confidence=float(data["confidence"]),
            created_at=float(data["created_at"]),
            updated_at=float(data["updated_at"]),
            expires_at=float(data["expires_at"]) if data.get("expires_at") is not None else None,
            is_active=bool(data["is_active"]),
            metadata_json=data.get("metadata_json"),
        )

    def save_memory(self, record: MemoryRecord) -> MemoryRecord:
        """
        Validates and persists a MemoryRecord atomically.
        Enforces Phase 6.1 boundary validation prior to storage.
        If an active memory with the same (category, key) exists, it is logically updated.
        """
        # Validate record and boundary constraints before starting transaction
        record.validate()
        cat_enum = record.category if isinstance(record.category, MemoryCategory) else MemoryCategory(record.category)
        MemoryBoundaryValidator.validate_memory_candidate(
            key=record.key,
            value=record.value,
            category=cat_enum,
            metadata_json=record.metadata_json,
        )

        conn = self._get_connection()
        try:
            # Check if record with same ID exists
            cursor = conn.execute("SELECT id FROM memories WHERE id = ?", (record.id,))
            existing_by_id = cursor.fetchone()

            if existing_by_id:
                conn.execute(
                    """
                    UPDATE memories
                    SET category = ?, key = ?, value = ?, source = ?, confidence = ?,
                        updated_at = ?, expires_at = ?, is_active = ?, metadata_json = ?
                    WHERE id = ?
                    """,
                    (
                        cat_enum.value,
                        record.key,
                        record.value,
                        record.source.value,
                        record.confidence,
                        record.updated_at,
                        record.expires_at,
                        1 if record.is_active else 0,
                        record.metadata_json,
                        record.id,
                    ),
                )
            else:
                # Check if an active record with same (category, key) exists to prevent duplicate logical records
                cursor = conn.execute(
                    """
                    SELECT id FROM memories
                    WHERE category = ? AND key = ? AND is_active = 1
                    """,
                    (cat_enum.value, record.key),
                )
                existing_key_row = cursor.fetchone()

                if existing_key_row:
                    target_id = existing_key_row["id"]
                    conn.execute(
                        """
                        UPDATE memories
                        SET id = ?, value = ?, source = ?, confidence = ?,
                            updated_at = ?, expires_at = ?, is_active = ?, metadata_json = ?
                        WHERE id = ?
                        """,
                        (
                            record.id,
                            record.value,
                            record.source.value,
                            record.confidence,
                            record.updated_at,
                            record.expires_at,
                            1 if record.is_active else 0,
                            record.metadata_json,
                            target_id,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO memories (
                            id, category, key, value, source, confidence,
                            created_at, updated_at, expires_at, is_active, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.id,
                            cat_enum.value,
                            record.key,
                            record.value,
                            record.source.value,
                            record.confidence,
                            record.created_at,
                            record.updated_at,
                            record.expires_at,
                            1 if record.is_active else 0,
                            record.metadata_json,
                        ),
                    )
            conn.commit()
            return record
        except Exception:
            conn.rollback()
            raise

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieves a MemoryRecord by its unique ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT id, category, key, value, source, confidence,
                   created_at, updated_at, expires_at, is_active, metadata_json
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._record_from_row(row)

    def get_memory_by_key(
        self, category: Union[MemoryCategory, str], key: str, active_only: bool = True
    ) -> Optional[MemoryRecord]:
        """Retrieves a MemoryRecord by category and logical key."""
        cat_str = category.value if isinstance(category, MemoryCategory) else str(category)
        conn = self._get_connection()
        if active_only:
            now = time.time()
            cursor = conn.execute(
                """
                SELECT id, category, key, value, source, confidence,
                       created_at, updated_at, expires_at, is_active, metadata_json
                FROM memories
                WHERE category = ? AND key = ? AND is_active = 1
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY updated_at DESC, id ASC
                LIMIT 1
                """,
                (cat_str, key, now),
            )
        else:
            cursor = conn.execute(
                """
                SELECT id, category, key, value, source, confidence,
                       created_at, updated_at, expires_at, is_active, metadata_json
                FROM memories
                WHERE category = ? AND key = ?
                ORDER BY updated_at DESC, id ASC
                LIMIT 1
                """,
                (cat_str, key),
            )
        row = cursor.fetchone()
        if not row:
            return None
        return self._record_from_row(row)

    def list_memories(
        self,
        category: Optional[Union[MemoryCategory, str]] = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> List[MemoryRecord]:
        """
        Lists MemoryRecords ordered deterministically by updated_at DESC, id ASC.
        Filters by category and active/expiration state if specified.
        """
        conn = self._get_connection()
        now = time.time()
        query_parts = ["SELECT id, category, key, value, source, confidence, created_at, updated_at, expires_at, is_active, metadata_json FROM memories"]
        conditions = []
        params: List[Any] = []

        if category is not None:
            cat_str = category.value if isinstance(category, MemoryCategory) else str(category)
            conditions.append("category = ?")
            params.append(cat_str)

        if active_only:
            conditions.append("is_active = 1")
            conditions.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(now)

        if conditions:
            query_parts.append("WHERE " + " AND ".join(conditions))

        query_parts.append("ORDER BY updated_at DESC, id ASC LIMIT ?")
        params.append(limit)

        query = " ".join(query_parts)
        cursor = conn.execute(query, tuple(params))
        return [self._record_from_row(row) for row in cursor.fetchall()]

    def delete_memory(self, memory_id: str, hard_delete: bool = True) -> bool:
        """
        Deletes a memory record by ID.
        If hard_delete is True, physically removes row. Otherwise sets is_active = 0.
        Returns True if a record was modified/deleted, False if memory_id did not exist.
        """
        conn = self._get_connection()
        try:
            if hard_delete:
                cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            else:
                cursor = conn.execute("UPDATE memories SET is_active = 0 WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception:
            conn.rollback()
            raise

    def count_memories(self, active_only: bool = True) -> int:
        """Returns total count of memory records in database."""
        conn = self._get_connection()
        now = time.time()
        if active_only:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE is_active = 1 AND (expires_at IS NULL OR expires_at > ?)",
                (now,),
            )
        else:
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
        row = cursor.fetchone()
        return row[0] if row else 0

    def prune_expired_memories(self, hard_delete: bool = False) -> int:
        """
        Deactivates or permanently deletes memories whose expires_at timestamp has passed.
        Returns the count of pruned memory records.
        """
        conn = self._get_connection()
        now = time.time()
        try:
            if hard_delete:
                cursor = conn.execute(
                    "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (now,),
                )
            else:
                cursor = conn.execute(
                    "UPDATE memories SET is_active = 0 WHERE expires_at IS NOT NULL AND expires_at <= ? AND is_active = 1",
                    (now,),
                )
            conn.commit()
            return cursor.rowcount
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        """Closes database connections safely."""
        with self._lock:
            if self._is_memory:
                if hasattr(self, "_shared_conn") and self._shared_conn:
                    try:
                        self._shared_conn.close()
                    except Exception:
                        pass
            else:
                if hasattr(self._local, "conn") and self._local.conn:
                    try:
                        self._local.conn.close()
                        del self._local.conn
                    except Exception:
                        pass
