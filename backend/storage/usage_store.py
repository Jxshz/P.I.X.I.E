import sqlite3
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class UsageStore:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Store in the same directory as this file by default
            db_path = str(Path(__file__).parent / "usage.db")
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            # Enable WAL mode for better concurrency and performance
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_db(self):
        conn = self._get_connection()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                date_str TEXT NOT NULL,
                model TEXT NOT NULL,
                request_tokens INTEGER,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                was_rate_limited BOOLEAN NOT NULL DEFAULT 0
            )
        ''')
        # Index for faster daily aggregation queries
        conn.execute('CREATE INDEX IF NOT EXISTS idx_date_str ON usage_logs(date_str)')
        conn.commit()

    def record_success(self, model: str, total_tokens: int, request_tokens: Optional[int] = None):
        """Records a successful API request and its token usage."""
        conn = self._get_connection()
        now = time.time()
        date_str = datetime.fromtimestamp(now).strftime('%Y-%m-%d')
        conn.execute('''
            INSERT INTO usage_logs (timestamp, date_str, model, request_tokens, total_tokens, was_rate_limited)
            VALUES (?, ?, ?, ?, ?, 0)
        ''', (now, date_str, model, request_tokens, total_tokens))
        conn.commit()

    def record_rate_limit(self, model: str):
        """Records a request blocked by the TokenGovernor."""
        conn = self._get_connection()
        now = time.time()
        date_str = datetime.fromtimestamp(now).strftime('%Y-%m-%d')
        conn.execute('''
            INSERT INTO usage_logs (timestamp, date_str, model, request_tokens, total_tokens, was_rate_limited)
            VALUES (?, ?, ?, NULL, 0, 1)
        ''', (now, date_str, model))
        conn.commit()

    def get_daily_history(self, days: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns aggregated usage for the last N days, newest first.
        """
        conn = self._get_connection()
        # Get data grouped by date_str
        cursor = conn.execute('''
            SELECT
                date_str as date,
                COUNT(CASE WHEN was_rate_limited = 0 THEN 1 END) as requests,
                SUM(total_tokens) as tokens,
                SUM(was_rate_limited) as rate_limit_blocks
            FROM usage_logs
            GROUP BY date_str
            ORDER BY date_str DESC
            LIMIT ?
        ''', (days,))

        results = []
        for row in cursor:
            results.append({
                "date": row["date"],
                "requests": row["requests"],
                "tokens": row["tokens"] or 0,
                "rate_limit_blocks": row["rate_limit_blocks"] or 0
            })

        return {"days": results}
