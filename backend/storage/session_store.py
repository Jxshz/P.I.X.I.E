import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class SessionStore:
    """
    SQLite persistence layer for P.I.X.I.E. chat sessions and message history.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path(__file__).parent / "sessions.db")
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._is_memory = db_path == ":memory:" or "mode=memory" in db_path

        if self._is_memory:
            # Shared connection for in-memory database
            self._shared_conn = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
            self._shared_conn.row_factory = sqlite3.Row
            self._shared_conn.execute("PRAGMA foreign_keys = ON")
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
            conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn = conn
        else:
            # Ensure foreign keys are enabled on existing thread connection
            self._local.conn.execute("PRAGMA foreign_keys = ON")
        return self._local.conn

    def _init_db(self, conn: sqlite3.Connection):
        with self._lock:
            conn.execute("PRAGMA foreign_keys = ON")
            if not self._is_memory:
                conn.execute("PRAGMA journal_mode=WAL")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls_json TEXT,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
                """
            )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp ON messages(session_id, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC)"
            )
            conn.commit()

    def create_session(
        self, title: str = "New Chat", session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates a new session record."""
        conn = self._get_connection()
        sid = session_id or str(uuid.uuid4())
        now = time.time()
        conn.execute(
            """
            INSERT INTO sessions (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (sid, title, now, now),
        )
        conn.commit()
        return {
            "id": sid,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves session metadata by session_id."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists sessions ordered by newest updated_at timestamp."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def update_session_title(
        self, session_id: str, title: str
    ) -> Optional[Dict[str, Any]]:
        """Updates the title and updated_at timestamp of a session."""
        conn = self._get_connection()
        now = time.time()
        cursor = conn.execute(
            """
            UPDATE sessions
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (title, now, session_id),
        )
        conn.commit()
        if cursor.rowcount == 0:
            return None
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session and all its associated messages via cascade."""
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM sessions WHERE id = ?", (session_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls_json: Optional[str] = None,
        message_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Adds a message to the session.
        Raises sqlite3.IntegrityError if session_id does not exist.
        """
        conn = self._get_connection()
        mid = message_id or str(uuid.uuid4())
        ts = timestamp if timestamp is not None else time.time()

        # Insert message (foreign key enforcement guarantees session_id exists)
        conn.execute(
            """
            INSERT INTO messages (id, session_id, role, content, tool_calls_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (mid, session_id, role, content, tool_calls_json, ts),
        )

        # Update parent session's updated_at timestamp
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (ts, session_id)
        )
        conn.commit()

        return {
            "id": mid,
            "session_id": session_id,
            "role": role,
            "content": content,
            "tool_calls_json": tool_calls_json,
            "timestamp": ts,
        }

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves all messages for a session ordered by timestamp."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT id, session_id, role, content, tool_calls_json, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC, rowid ASC
            """,
            (session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Closes connections cleanly."""
        if self._is_memory and hasattr(self, "_shared_conn"):
            self._shared_conn.close()
        elif hasattr(self._local, "conn"):
            self._local.conn.close()
            delattr(self._local, "conn")
