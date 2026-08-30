import json
import sqlite3
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agent.core import AgentCore
from backend.agent.personality import SYSTEM_PROMPT
from backend.storage.session_store import SessionStore


@pytest.fixture
def mock_groq():
    with patch("backend.agent.core.AsyncGroq") as mock_groq_cls:
        client_instance = MagicMock()
        mock_groq_cls.return_value = client_instance

        # Default mock completion behavior
        def make_completion(content_text="Mock completion response."):
            mock_comp = MagicMock()
            mock_choice = MagicMock()
            mock_msg = MagicMock()
            mock_msg.content = content_text
            mock_msg.tool_calls = None
            mock_choice.message = mock_msg
            mock_comp.choices = [mock_choice]
            mock_comp.usage = MagicMock(prompt_tokens=15, total_tokens=45)
            return mock_comp

        async def async_create(**kwargs):
            # If side_effect text was set on mock, use it
            resp_text = getattr(client_instance, "_custom_response", "Default response.")
            return make_completion(resp_text)

        client_instance.chat.completions.create = AsyncMock(side_effect=async_create)
        yield client_instance


def set_mock_response(client_instance, text: str):
    client_instance._custom_response = text


@pytest.mark.asyncio
async def test_multiturn_persistence(tmp_path, mock_groq):
    db_file = str(tmp_path / "multiturn.db")
    store = SessionStore(db_file)
    session_id = "sess-multiturn-1"

    agent = AgentCore(session_store=store, session_id=session_id)

    # Turn 1
    set_mock_response(mock_groq, "Transformers are neural network architectures based on self-attention.")
    await agent.process_intent("Explain transformers.")

    # Turn 2
    set_mock_response(mock_groq, "Think of self-attention like highlighting key words in a sentence as you read.")
    await agent.process_intent("Explain that more simply.")

    # Turn 3
    set_mock_response(mock_groq, "For example, in 'The bank of the river', self-attention connects 'river' to 'bank'.")
    await agent.process_intent("Give me an example.")

    # Verify SQLite persistence
    messages = store.get_messages(session_id)
    assert len(messages) == 6
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user", "assistant"]
    assert "Explain transformers" in messages[0]["content"]
    assert "Transformers are neural network" in messages[1]["content"]
    assert "more simply" in messages[2]["content"]
    assert "Give me an example" in messages[4]["content"]

    store.close()


@pytest.mark.asyncio
async def test_agentcore_reconstruction(tmp_path, mock_groq):
    db_file = str(tmp_path / "reconstruct.db")
    store = SessionStore(db_file)
    session_id = "sess-reconstruct"

    # AgentCore A processes turns
    agent_a = AgentCore(session_store=store, session_id=session_id)
    set_mock_response(mock_groq, "Planning is setting goals.")
    await agent_a.process_intent("Explain planning")

    # Destroy AgentCore A
    del agent_a

    # Recreate AgentCore B using same store and session_id
    agent_b = AgentCore(session_store=store, session_id=session_id)
    assert len(agent_b.conversation_history) == 3
    assert agent_b.conversation_history[0]["role"] == "system"
    assert agent_b.conversation_history[1]["content"] == "Explain planning"
    assert agent_b.conversation_history[2]["content"] == "Planning is setting goals."

    store.close()


@pytest.mark.asyncio
async def test_conversation_continuation(tmp_path, mock_groq):
    db_file = str(tmp_path / "continuation.db")
    store = SessionStore(db_file)
    session_id = "sess-continuation"

    # Agent A processes turn 1
    agent_a = AgentCore(session_store=store, session_id=session_id)
    set_mock_response(mock_groq, "First response.")
    await agent_a.process_intent("Turn 1")
    del agent_a

    # Agent B restores and processes turn 2
    agent_b = AgentCore(session_store=store, session_id=session_id)
    set_mock_response(mock_groq, "Second response.")
    await agent_b.process_intent("Turn 2")

    # Verify context and SQLite contain complete history
    persisted = store.get_messages(session_id)
    assert len(persisted) == 4
    assert [m["content"] for m in persisted] == ["Turn 1", "First response.", "Turn 2", "Second response."]
    store.close()


def test_system_prompt_integrity(tmp_path):
    db_file = str(tmp_path / "sys_prompt.db")
    store = SessionStore(db_file)
    session_id = "sess-sys-prompt"

    agent1 = AgentCore(session_store=store, session_id=session_id)
    store.add_message(session_id, "user", "Hello")
    store.add_message(session_id, "assistant", "Hi there")

    # Reconstruct multiple times
    agent2 = AgentCore(session_store=store, session_id=session_id)
    agent3 = AgentCore(session_store=store, session_id=session_id)

    for ag in (agent1, agent2, agent3):
        system_msgs = [m for m in ag.conversation_history if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert ag.conversation_history[0]["content"] == SYSTEM_PROMPT

    store.close()


def test_ordering_preservation(tmp_path):
    db_file = str(tmp_path / "order.db")
    store = SessionStore(db_file)
    session_id = "sess-order"
    store.create_session(session_id=session_id)

    t0 = time.time()
    store.add_message(session_id, "user", "First", timestamp=t0)
    store.add_message(session_id, "assistant", "Second", timestamp=t0 + 1.0)
    store.add_message(session_id, "user", "Third", timestamp=t0 + 2.0)

    agent = AgentCore(session_store=store, session_id=session_id)
    contents = [m["content"] for m in agent.conversation_history[1:]]
    assert contents == ["First", "Second", "Third"]
    store.close()


def test_context_trimming_working_memory(tmp_path):
    db_file = str(tmp_path / "trim.db")
    store = SessionStore(db_file)
    session_id = "sess-trim"

    agent = AgentCore(session_store=store, session_id=session_id)
    for i in range(15):
        agent.conversation_history.append({"role": "user", "content": f"User Turn {i} " + "x" * 150})
        agent.conversation_history.append({"role": "assistant", "content": f"Asst Turn {i} " + "y" * 150})

    # Trim working memory
    agent._trim_context()

    # Active context trimmed to budget
    assert agent.conversation_history[0]["role"] == "system"
    assert agent.governor.estimate_tokens(agent.conversation_history) <= 3000
    store.close()


def test_database_history_unaffected_by_trimming(tmp_path):
    db_file = str(tmp_path / "db_unaffected.db")
    store = SessionStore(db_file)
    session_id = "sess-unaffected"

    # Insert 20 long turns into SQLite (totaling >3500 tokens)
    store.create_session(session_id=session_id)
    for i in range(20):
        store.add_message(session_id, "user", f"User question {i} " + "context " * 40)
        store.add_message(session_id, "assistant", f"Assistant answer {i} " + "detail " * 40)

    # Initialize AgentCore and trim working memory
    agent = AgentCore(session_store=store, session_id=session_id)
    agent._trim_context()

    # Active memory is trimmed
    assert len(agent.conversation_history) < 41

    # Long-term SQLite history remains 100% intact (40 messages)
    db_msgs = store.get_messages(session_id)
    assert len(db_msgs) == 40

    store.close()


@pytest.mark.asyncio
async def test_formatting_integrity(tmp_path, mock_groq):
    db_file = str(tmp_path / "fmt.db")
    store = SessionStore(db_file)
    session_id = "sess-fmt"

    agent = AgentCore(session_store=store, session_id=session_id)
    set_mock_response(mock_groq, "A manager plans and coordinates. An entrepreneur innovates and takes risks.")
    display, spoken, action = await agent.process_intent("Compare manager vs entrepreneur")

    assert "##" not in display
    assert "**" not in display
    assert spoken is not None
    store.close()


def test_tool_call_metadata_restoration(tmp_path):
    db_file = str(tmp_path / "tool_meta.db")
    store = SessionStore(db_file)
    session_id = "sess-tool-meta"
    store.create_session(session_id=session_id)

    tc_json = json.dumps([{"id": "call_abc", "type": "function", "function": {"name": "system_diagnostics", "arguments": "{}"}}])
    store.add_message(session_id, "assistant", "Checking system...", tool_calls_json=tc_json)
    store.add_message(session_id, "tool", '{"status": "ok"}', tool_calls_json=json.dumps({"tool_call_id": "call_abc", "name": "system_diagnostics"}))

    agent = AgentCore(session_store=store, session_id=session_id)
    assert len(agent.conversation_history) == 3
    assert agent.conversation_history[1]["tool_calls"][0]["id"] == "call_abc"
    assert agent.conversation_history[2]["tool_call_id"] == "call_abc"
    assert agent.conversation_history[2]["name"] == "system_diagnostics"
    store.close()


@pytest.mark.asyncio
async def test_database_close_and_reopen(tmp_path, mock_groq):
    db_file = str(tmp_path / "close_reopen.db")
    session_id = "sess-reopen"

    # Phase 1: Write conversation to disk
    store1 = SessionStore(db_file)
    agent1 = AgentCore(session_store=store1, session_id=session_id)
    set_mock_response(mock_groq, "Phase 1 completion.")
    await agent1.process_intent("Phase 1 request")
    store1.close()

    # Phase 2: Reopen store from disk file
    store2 = SessionStore(db_file)
    agent2 = AgentCore(session_store=store2, session_id=session_id)

    assert len(agent2.conversation_history) == 3
    assert agent2.conversation_history[1]["content"] == "Phase 1 request"
    assert agent2.conversation_history[2]["content"] == "Phase 1 completion."

    # Phase 3: Continue conversation
    set_mock_response(mock_groq, "Phase 2 completion.")
    await agent2.process_intent("Phase 2 request")

    persisted = store2.get_messages(session_id)
    assert len(persisted) == 4
    assert persisted[3]["content"] == "Phase 2 completion."
    store2.close()


def test_missing_session(tmp_path):
    db_file = str(tmp_path / "missing.db")
    store = SessionStore(db_file)

    agent = AgentCore(session_store=store, session_id="auto-create-me")
    assert agent.session_id == "auto-create-me"
    assert store.get_session("auto-create-me") is not None
    store.close()


def test_malformed_message(tmp_path):
    db_file = str(tmp_path / "malformed.db")
    store = SessionStore(db_file)
    session_id = "sess-malformed"
    store.create_session(session_id=session_id)

    # Insert valid message + message with invalid JSON tool_calls
    store.add_message(session_id, "user", "Valid message")
    store.add_message(session_id, "assistant", "Corrupted tool call", tool_calls_json="{invalid_json_content...")

    agent = AgentCore(session_store=store, session_id=session_id)
    # Should not crash, and should load valid content safely
    assert len(agent.conversation_history) == 3
    assert agent.conversation_history[1]["content"] == "Valid message"
    assert agent.conversation_history[2]["content"] == "Corrupted tool call"
    store.close()


@pytest.mark.asyncio
async def test_persistence_failure(mock_groq):
    failing_store = MagicMock()
    failing_store.get_session.return_value = {"id": "fail-id", "title": "Test"}
    failing_store.get_messages.return_value = []
    failing_store.add_message.side_effect = sqlite3.OperationalError("Disk full")

    agent = AgentCore(session_store=failing_store, session_id="fail-id")
    set_mock_response(mock_groq, "Fallback text")
    display, spoken, action = await agent.process_intent("Test input")

    # In-memory execution must succeed without crash or corruption
    assert display is not None
    assert spoken is not None
    assert len(agent.conversation_history) == 3


@pytest.mark.asyncio
async def test_unicode_conversation(tmp_path, mock_groq):
    db_file = str(tmp_path / "unicode.db")
    store = SessionStore(db_file)
    session_id = "sess-unicode"

    agent1 = AgentCore(session_store=store, session_id=session_id)
    unicode_q = "こんにちは！ P.I.X.I.E. 🚀 support test: Привет, こんにちは, 🤖."
    unicode_a = "Sir, こんにちは！ Unicode is fully supported: 🚀, 💬, 🤖."

    set_mock_response(mock_groq, unicode_a)
    await agent1.process_intent(unicode_q)

    # Reconstruct
    agent2 = AgentCore(session_store=store, session_id=session_id)
    assert agent2.conversation_history[1]["content"] == unicode_q
    assert agent2.conversation_history[2]["content"] == unicode_a
    store.close()


@pytest.mark.asyncio
async def test_long_conversation(tmp_path, mock_groq):
    db_file = str(tmp_path / "long.db")
    store = SessionStore(db_file)
    session_id = "sess-long"

    agent1 = AgentCore(session_store=store, session_id=session_id)
    long_q = "Detail request: " + "data " * 1000  # ~5,000 chars
    long_a = "Detailed explanation: " + "info " * 1000  # ~5,000 chars

    set_mock_response(mock_groq, long_a)
    await agent1.process_intent(long_q)

    # Reconstruct
    agent2 = AgentCore(session_store=store, session_id=session_id)
    assert len(agent2.conversation_history[1]["content"]) > 4000
    assert len(agent2.conversation_history[2]["content"]) > 4000
    store.close()


@pytest.mark.asyncio
async def test_rapid_consecutive_turns(tmp_path, mock_groq):
    db_file = str(tmp_path / "rapid.db")
    store = SessionStore(db_file)
    session_id = "sess-rapid"

    agent = AgentCore(session_store=store, session_id=session_id)
    for i in range(10):
        set_mock_response(mock_groq, f"Response {i}")
        await agent.process_intent(f"Request {i}")

    persisted = store.get_messages(session_id)
    assert len(persisted) == 20
    for i in range(10):
        assert persisted[i * 2]["content"] == f"Request {i}"
        assert persisted[i * 2 + 1]["content"] == f"Response {i}"
    store.close()
