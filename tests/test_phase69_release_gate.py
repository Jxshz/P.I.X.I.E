import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock
import pytest

from backend.agent.core import AgentCore
from backend.memory import (
    MemoryCategory,
    MemoryContextBuilder,
    MemoryRecord,
    MemoryRetriever,
    MemoryService,
    MemorySource,
    MemoryStore,
    MemoryValidationError,
    format_memory_context_untrusted,
)
from backend.storage.session_store import SessionStore


@pytest.fixture
def temp_db_path():
    """Provides a temporary file path for disk-backed database testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def test_gate_1_persistence_across_restarts(temp_db_path):
    """Subphase 6.9.5: Verifies memory persistence, supersession, and lifecycle across DB restarts."""
    # 1. Open DB and create memory
    svc1 = MemoryService(db_path=temp_db_path)
    rec1 = svc1.create_memory(
        category=MemoryCategory.USER_PROFILE,
        key="name",
        value="Joshva",
        metadata_json='{"version": 1}',
    )
    rec_id = rec1.id
    rec_created_at = rec1.created_at
    svc1.close()

    # 2. Reopen DB and verify retrieval
    svc2 = MemoryService(db_path=temp_db_path)
    fetched2 = svc2.get_memory(rec_id)
    assert fetched2 is not None
    assert fetched2.value == "Joshva"
    assert fetched2.created_at == rec_created_at

    # 3. Update memory in DB session 2
    time.sleep(0.01)
    svc2.update_memory(rec_id, value="Joshva N.", metadata_json='{"version": 2}')
    svc2.close()

    # 4. Reopen DB in session 3 and verify updated record
    svc3 = MemoryService(db_path=temp_db_path)
    fetched3 = svc3.get_memory(rec_id)
    assert fetched3 is not None
    assert fetched3.value == "Joshva N."
    assert fetched3.updated_at > rec_created_at
    assert fetched3.metadata_json == '{"version": 2}'

    # 5. Supersede memory across restart
    svc3.supersede_memory(
        category=MemoryCategory.USER_PROFILE,
        key="name",
        value="Joshva Novus",
    )
    svc3.close()

    # 6. Reopen DB in session 4 and retrieve
    svc4 = MemoryService(db_path=temp_db_path)
    retriever = MemoryRetriever(memory_service=svc4)
    matches = retriever.retrieve("name")
    assert len(matches) == 1
    assert matches[0].record.value == "Joshva Novus"

    retriever.close()
    svc4.close()


def test_gate_2_pipeline_separation_and_boundary(temp_db_path):
    """Subphase 6.9.1: Confirms clear layer separation across MemoryStore -> Service -> Retriever -> Builder -> AgentCore."""
    svc = MemoryService(db_path=temp_db_path)
    # Service owns validation and lifecycle
    rec = svc.create_memory(category=MemoryCategory.USER_PREFERENCE, key="theme", value="Dark Mode")

    # Retriever is read-only
    retriever = MemoryRetriever(memory_service=svc)
    matches = retriever.retrieve("theme")
    assert len(matches) == 1
    # Match does not mutate database
    assert svc.count_memories() == 1

    # ContextBuilder wraps records in untrusted block
    builder = MemoryContextBuilder(retriever=retriever)
    ctx = builder.build_memory_context("theme")
    assert "<retrieved_memory_context>" in ctx
    assert "Dark Mode" in ctx

    retriever.close()
    svc.close()


@pytest.mark.asyncio
async def test_gate_3_failure_injection_read_and_write():
    """Subphase 6.9.9: Verifies read path fails safe and write path fails closed."""
    # Write failure fails closed
    svc = MemoryService(db_path=":memory:")
    with pytest.raises(MemoryValidationError, match="Security Violation"):
        svc.create_memory(
            category=MemoryCategory.USER_FACT,
            key="sec",
            value="sk-1234567890abcdef1234567890",
        )
    assert svc.count_memories() == 0

    # Read failure fails safe without breaking AgentCore
    failing_retriever = MagicMock()
    failing_retriever.retrieve.side_effect = Exception("Disk IO Error /private/var/db.sqlite")

    agent = AgentCore(memory_retriever=failing_retriever, enable_memory=True)
    agent.client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock(content="Clean response.", tool_calls=None)
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_completion.usage = MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    agent.client.chat.completions.create = AsyncMock(return_value=mock_completion)

    display, spoken, meta = await agent.process_intent("Test input")
    assert display is not None
    assert "/private/var/db.sqlite" not in display
    assert "Disk IO Error" not in display
    svc.close()
