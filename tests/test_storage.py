import os
import pytest
import sqlite3
import time
from datetime import datetime
from backend.storage.usage_store import UsageStore
from backend.agent.core import AgentCore, RateLimitException
from backend.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_usage.db"
    store = UsageStore(str(db_path))
    yield store
    if db_path.exists():
        db_path.unlink()

def test_usage_store_initialization(temp_db):
    conn = temp_db._get_connection()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usage_logs'")
    assert cursor.fetchone() is not None

def test_record_success(temp_db):
    temp_db.record_success("test-model", request_tokens=100, total_tokens=150)
    conn = temp_db._get_connection()
    cursor = conn.execute("SELECT * FROM usage_logs")
    row = cursor.fetchone()
    assert row["model"] == "test-model"
    assert row["request_tokens"] == 100
    assert row["total_tokens"] == 150
    assert row["was_rate_limited"] == 0

def test_record_rate_limit(temp_db):
    temp_db.record_rate_limit("test-model")
    conn = temp_db._get_connection()
    cursor = conn.execute("SELECT * FROM usage_logs")
    row = cursor.fetchone()
    assert row["model"] == "test-model"
    assert row["request_tokens"] is None
    assert row["total_tokens"] == 0
    assert row["was_rate_limited"] == 1

def test_daily_aggregation(temp_db):
    # Record a mix of successes and blocks
    temp_db.record_success("test-model", request_tokens=10, total_tokens=20)
    temp_db.record_success("test-model", request_tokens=5, total_tokens=10)
    temp_db.record_rate_limit("test-model")

    history = temp_db.get_daily_history()
    days = history["days"]
    assert len(days) == 1

    today = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
    assert days[0]["date"] == today
    assert days[0]["requests"] == 2
    assert days[0]["tokens"] == 30
    assert days[0]["rate_limit_blocks"] == 1

def test_history_endpoint():
    # Make sure we don't break the actual endpoint
    response = client.get("/usage/history")
    assert response.status_code == 200
    data = response.json()
    assert "days" in data
    assert isinstance(data["days"], list)
