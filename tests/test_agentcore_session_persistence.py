import json
import sqlite3
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agent.core import AgentCore
from backend.agent.personality import SYSTEM_PROMPT
from backend.storage.session_store import SessionStore


@pytest.fixture
def memory_session_store():
    store = SessionStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def mock_groq_client():
    with patch("backend.agent.core.AsyncGroq") as mock_groq_cls:
        client_instance = MagicMock()
        mock_groq_cls.return_value = client_instance

        # Setup mock chat completion response
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "Sir, planning is the fundamental process of setting goals."
        mock_msg.tool_calls = None
        mock_choice.message = mock_msg
        mock_completion.choices = [mock_choice]
        mock_completion.usage = MagicMock(prompt_tokens=10, total_tokens=30)

        client_instance.chat.completions.create = AsyncMock(return_value=mock_completion)
        yield client_instance


def test_agentcore_creates_or_uses_session(memory_session_store):
    # Case 1: No session_id provided -> AgentCore creates a session automatically
    agent1 = AgentCore(session_store=memory_session_store)
    assert agent1.session_id is not None
    assert memory_session_store.get_session(agent1.session_id) is not None

    # Case 2: Specific session_id provided -> AgentCore uses that session_id
    agent2 = AgentCore(session_store=memory_session_store, session_id="custom-sess-1")
    assert agent2.session_id == "custom-sess-1"
    assert memory_session_store.get_session("custom-sess-1") is not None


def test_empty_session_loads_correctly(memory_session_store):
    agent = AgentCore(session_store=memory_session_store, session_id="empty-sess")
    assert len(agent.conversation_history) == 1
    assert agent.conversation_history[0]["role"] == "system"
    assert agent.conversation_history[0]["content"] == SYSTEM_PROMPT


def test_existing_messages_load_into_conversation_history(memory_session_store):
    sess = memory_session_store.create_session(session_id="existing-sess")
    memory_session_store.add_message("existing-sess", "user", "Explain planning")
    memory_session_store.add_message("existing-sess", "assistant", "Planning defines goals.")

    agent = AgentCore(session_store=memory_session_store, session_id="existing-sess")
    assert len(agent.conversation_history) == 3
    assert agent.conversation_history[0]["role"] == "system"
    assert agent.conversation_history[1]["role"] == "user"
    assert agent.conversation_history[1]["content"] == "Explain planning"
    assert agent.conversation_history[2]["role"] == "assistant"
    assert agent.conversation_history[2]["content"] == "Planning defines goals."


def test_system_prompt_remains_exactly_once(memory_session_store):
    sess = memory_session_store.create_session(session_id="sess-once")
    memory_session_store.add_message("sess-once", "user", "Hello")

    agent = AgentCore(session_store=memory_session_store, session_id="sess-once")
    system_msgs = [m for m in agent.conversation_history if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert agent.conversation_history[0]["role"] == "system"


@pytest.mark.asyncio
async def test_user_message_persistence(memory_session_store, mock_groq_client):
    agent = AgentCore(session_store=memory_session_store, session_id="user-persist")
    await agent.process_intent("What is entrepreneurship?")

    messages = memory_session_store.get_messages("user-persist")
    assert len(messages) >= 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is entrepreneurship?"


@pytest.mark.asyncio
async def test_assistant_message_persistence(memory_session_store, mock_groq_client):
    agent = AgentCore(session_store=memory_session_store, session_id="asst-persist")
    display, spoken, action = await agent.process_intent("Tell me about strategy")

    messages = memory_session_store.get_messages("asst-persist")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert "planning is the fundamental process" in messages[1]["content"]


@pytest.mark.asyncio
async def test_multiple_turns_survive_reconstruction(memory_session_store, mock_groq_client):
    sess_id = "multi-turn-sess"
    agent_turn1 = AgentCore(session_store=memory_session_store, session_id=sess_id)
    await agent_turn1.process_intent("Turn 1 question")

    # Reconstruct new AgentCore instance for turn 2 with same session_id
    agent_turn2 = AgentCore(session_store=memory_session_store, session_id=sess_id)
    assert len(agent_turn2.conversation_history) == 3  # System + User1 + Asst1

    await agent_turn2.process_intent("Turn 2 question")

    # Verify 4 messages persisted (User1, Asst1, User2, Asst2)
    persisted = memory_session_store.get_messages(sess_id)
    assert len(persisted) == 4
    assert [m["role"] for m in persisted] == ["user", "assistant", "user", "assistant"]


@pytest.mark.asyncio
async def test_tool_call_metadata_survives_persistence_and_reload(memory_session_store, mock_groq_client):
    sess_id = "tool-sess"
    store = memory_session_store
    store.create_session(session_id=sess_id)

    # Persist tool calls payload
    tc_json = json.dumps([{"id": "call_999", "type": "function", "function": {"name": "system_diagnostics", "arguments": "{}"}}])
    store.add_message(sess_id, "assistant", "Running diagnostics...", tool_calls_json=tc_json)
    store.add_message(sess_id, "tool", '{"cpu_percent": 12.5}', tool_calls_json=json.dumps({"tool_call_id": "call_999", "name": "system_diagnostics"}))

    agent = AgentCore(session_store=store, session_id=sess_id)
    assert len(agent.conversation_history) == 3
    assert agent.conversation_history[1]["role"] == "assistant"
    assert agent.conversation_history[1]["tool_calls"][0]["function"]["name"] == "system_diagnostics"
    assert agent.conversation_history[2]["role"] == "tool"
    assert agent.conversation_history[2]["name"] == "system_diagnostics"


def test_context_trimming_remains_compatible(memory_session_store):
    agent = AgentCore(session_store=memory_session_store, session_id="trim-sess")

    # Add multiple long messages
    for i in range(15):
        agent.conversation_history.append({"role": "user", "content": f"Turn {i} " + "data " * 100})
        agent.conversation_history.append({"role": "assistant", "content": f"Response {i} " + "info " * 100})

    agent._trim_context()
    assert agent.conversation_history[0]["role"] == "system"
    assert agent.governor.estimate_tokens(agent.conversation_history) <= 3000


def test_clear_context_consistency(memory_session_store):
    sess_id = "clear-sess"
    agent = AgentCore(session_store=memory_session_store, session_id=sess_id)
    memory_session_store.add_message(sess_id, "user", "Pre-clear message")
    memory_session_store.add_message(sess_id, "assistant", "Pre-clear response")

    assert len(memory_session_store.get_messages(sess_id)) == 2

    # Clear context
    agent.clear_context()

    # In-memory history reset
    assert len(agent.conversation_history) == 1
    assert agent.conversation_history[0]["role"] == "system"

    # Persisted session history reset
    assert len(memory_session_store.get_messages(sess_id)) == 0


def test_missing_session_handling(memory_session_store):
    missing_id = "auto-created-session"
    assert memory_session_store.get_session(missing_id) is None

    agent = AgentCore(session_store=memory_session_store, session_id=missing_id)
    assert agent.session_id == missing_id
    assert memory_session_store.get_session(missing_id) is not None


def test_database_reopen_persistence(tmp_path):
    db_file = str(tmp_path / "agent_reopen.db")
    store1 = SessionStore(db_file)
    sess = store1.create_session("Reopen Agent Session")
    store1.add_message(sess["id"], "user", "Message before reopen")
    store1.close()

    store2 = SessionStore(db_file)
    agent = AgentCore(session_store=store2, session_id=sess["id"])
    assert len(agent.conversation_history) == 2
    assert agent.conversation_history[1]["content"] == "Message before reopen"
    store2.close()


@pytest.mark.asyncio
async def test_persistence_failure_does_not_corrupt_in_memory_history(mock_groq_client):
    failing_store = MagicMock()
    failing_store.get_session.return_value = {"id": "fail-sess", "title": "Fail"}
    failing_store.get_messages.return_value = []
    failing_store.add_message.side_effect = sqlite3.OperationalError("Database disk image is malformed")

    agent = AgentCore(session_store=failing_store, session_id="fail-sess")
    display, spoken, action = await agent.process_intent("Will this crash?")

    assert display is not None
    assert spoken is not None
    assert len(agent.conversation_history) == 3
    assert agent.conversation_history[1]["content"] == "Will this crash?"


@pytest.mark.asyncio
async def test_existing_phase45_display_formatting_intact(memory_session_store, mock_groq_client):
    agent = AgentCore(session_store=memory_session_store, session_id="fmt-sess")
    display, spoken, action = await agent.process_intent("Test formatting")

    # Display formatting must not contain markdown headings, bold labels, etc.
    assert "##" not in display
    assert "**" not in display
    assert spoken is not None


@pytest.mark.asyncio
async def test_existing_api_response_contract_intact(memory_session_store, mock_groq_client):
    agent = AgentCore(session_store=memory_session_store, session_id="contract-sess")
    result = await agent.process_intent("Contract test")

    assert isinstance(result, tuple)
    assert len(result) == 3
    display, spoken, action = result
    assert isinstance(display, str)
    assert isinstance(spoken, str)
    assert action is None or isinstance(action, dict)
