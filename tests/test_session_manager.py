import json
import sqlite3
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agent.core import AgentCore
from backend.agent.session_manager import SessionManager
from backend.storage.session_store import SessionStore
from backend.tools.base import BaseTool, PermissionLevel


class DummyConfirmedTool(BaseTool):
    @property
    def name(self) -> str:
        return "manager_dummy_action"

    @property
    def description(self) -> str:
        return "Tool requiring confirmation"

    @property
    def schema(self) -> dict:
        return {"type": "object", "properties": {}}

    @property
    def permission(self) -> PermissionLevel:
        return PermissionLevel.CONFIRM_REQUIRED

    def execute(self, **kwargs) -> str:
        return "Action executed"


@pytest.fixture
def mock_groq():
    with patch("backend.agent.core.AsyncGroq") as mock_groq_cls:
        client_instance = MagicMock()
        mock_groq_cls.return_value = client_instance

        def make_completion(content_text="Manager mock response."):
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
            resp_text = getattr(client_instance, "_custom_response", "Manager mock response.")
            return make_completion(resp_text)

        client_instance.chat.completions.create = AsyncMock(side_effect=async_create)
        yield client_instance


def set_mock_response(client_instance, text: str):
    client_instance._custom_response = text


def test_session_manager_initialization(tmp_path):
    db_file = str(tmp_path / "mgr_init.db")
    mgr = SessionManager(db_path=db_file)
    assert mgr.session_store is not None
    assert len(mgr._active_agents) == 0
    mgr.close()


def test_new_session_creation(tmp_path):
    db_file = str(tmp_path / "mgr_create.db")
    mgr = SessionManager(db_path=db_file)

    agent = mgr.create_session(title="Test Chat")
    assert isinstance(agent, AgentCore)
    assert agent.session_id is not None

    meta = mgr.session_store.get_session(agent.session_id)
    assert meta is not None
    assert meta["title"] == "Test Chat"
    mgr.close()


def test_session_id_propagation(tmp_path):
    db_file = str(tmp_path / "mgr_prop.db")
    mgr = SessionManager(db_path=db_file)

    agent = mgr.create_session(session_id="custom-sid-123")
    assert agent.session_id == "custom-sid-123"
    assert agent.session_store == mgr.session_store
    mgr.close()


def test_session_store_persistence(tmp_path):
    db_file = str(tmp_path / "mgr_persist.db")
    mgr = SessionManager(db_path=db_file)

    agent = mgr.create_session(session_id="sess-persist-check")
    assert mgr.session_store.get_session("sess-persist-check") is not None
    mgr.close()


def test_get_existing_session(tmp_path):
    db_file = str(tmp_path / "mgr_get.db")
    mgr = SessionManager(db_path=db_file)

    created_agent = mgr.create_session(session_id="sess-get-check")
    retrieved_agent = mgr.get_session("sess-get-check")

    assert retrieved_agent is not None
    assert retrieved_agent.session_id == "sess-get-check"
    mgr.close()


def test_agentcore_instance_reuse(tmp_path):
    db_file = str(tmp_path / "mgr_reuse.db")
    mgr = SessionManager(db_path=db_file)

    agent1 = mgr.create_session(session_id="sess-reuse")
    agent2 = mgr.get_session("sess-reuse")

    # Must return exact same object instance from memory cache
    assert agent1 is agent2
    mgr.close()


def test_different_sessions_different_instances(tmp_path):
    db_file = str(tmp_path / "mgr_diff.db")
    mgr = SessionManager(db_path=db_file)

    agent_a = mgr.create_session(session_id="sess-alpha")
    agent_b = mgr.create_session(session_id="sess-beta")

    assert agent_a is not agent_b
    assert agent_a.session_id == "sess-alpha"
    assert agent_b.session_id == "sess-beta"
    mgr.close()


@pytest.mark.asyncio
async def test_conversation_isolation(tmp_path, mock_groq):
    db_file = str(tmp_path / "mgr_iso.db")
    mgr = SessionManager(db_path=db_file)

    agent_a = mgr.create_session(session_id="sess-iso-a")
    agent_b = mgr.create_session(session_id="sess-iso-b")

    set_mock_response(mock_groq, "Response for A")
    await agent_a.process_intent("Prompt for A")

    set_mock_response(mock_groq, "Response for B")
    await agent_b.process_intent("Prompt for B")

    assert len(agent_a.conversation_history) == 3
    assert agent_a.conversation_history[1]["content"] == "Prompt for A"

    assert len(agent_b.conversation_history) == 3
    assert agent_b.conversation_history[1]["content"] == "Prompt for B"

    mgr.close()


def test_missing_invalid_session_handling(tmp_path):
    db_file = str(tmp_path / "mgr_missing.db")
    mgr = SessionManager(db_path=db_file)

    # Missing session ID must return None
    assert mgr.get_session("non_existent_session_id") is None
    mgr.close()


def test_get_or_create_session(tmp_path):
    db_file = str(tmp_path / "mgr_goc.db")
    mgr = SessionManager(db_path=db_file)

    # 1. Calling get_or_create_session without ID creates a new session
    agent1 = mgr.get_or_create_session(title="Auto Session")
    sid = agent1.session_id
    assert sid is not None

    # 2. Calling get_or_create_session with existing ID returns existing instance
    agent2 = mgr.get_or_create_session(session_id=sid)
    assert agent1 is agent2
    mgr.close()


def test_remove_session(tmp_path):
    db_file = str(tmp_path / "mgr_remove.db")
    mgr = SessionManager(db_path=db_file)

    agent = mgr.create_session(session_id="sess-to-delete")
    assert mgr.get_session("sess-to-delete") is not None

    res = mgr.remove_session("sess-to-delete")
    assert res is True
    assert mgr.get_session("sess-to-delete") is None
    assert mgr.session_store.get_session("sess-to-delete") is None
    mgr.close()


@pytest.mark.asyncio
async def test_reconstruct_session_after_manager_recreation(tmp_path, mock_groq):
    db_file = str(tmp_path / "mgr_reconstruct.db")

    # Manager 1 writes conversation
    mgr1 = SessionManager(db_path=db_file)
    agent1 = mgr1.create_session(session_id="persistent-sess")
    set_mock_response(mock_groq, "Persisted in Manager 1")
    await agent1.process_intent("Input in Manager 1")
    mgr1.close()

    # Manager 2 reopens DB file and retrieves session
    mgr2 = SessionManager(db_path=db_file)
    agent2 = mgr2.get_session("persistent-sess")

    assert agent2 is not None
    assert len(agent2.conversation_history) == 3
    assert agent2.conversation_history[1]["content"] == "Input in Manager 1"
    assert agent2.conversation_history[2]["content"] == "Persisted in Manager 1"
    mgr2.close()


@pytest.mark.asyncio
async def test_state_consistency_under_cache_clearing(tmp_path, mock_groq):
    db_file = str(tmp_path / "mgr_cache_clear.db")
    mgr = SessionManager(db_path=db_file)

    agent1 = mgr.create_session(session_id="cache-clear-sess")
    set_mock_response(mock_groq, "Response before cache clear")
    await agent1.process_intent("Prompt before cache clear")

    # Flush memory cache
    mgr.clear_session_cache("cache-clear-sess")
    assert "cache-clear-sess" not in mgr._active_agents

    # Re-retrieve session -> reconstitutes fresh AgentCore with restored SQLite state
    agent2 = mgr.get_session("cache-clear-sess")
    assert agent2 is not agent1
    assert len(agent2.conversation_history) == 3
    assert agent2.conversation_history[1]["content"] == "Prompt before cache clear"
    mgr.close()


@pytest.mark.asyncio
async def test_phase45_formatting_unaffected(tmp_path, mock_groq):
    db_file = str(tmp_path / "mgr_fmt.db")
    mgr = SessionManager(db_path=db_file)

    agent = mgr.create_session(session_id="fmt-sess")
    set_mock_response(mock_groq, "## Heading\n- Item 1\n**Bold text**")

    display, spoken, _ = await agent.process_intent("Explain planning")
    assert "##" not in display
    assert "**" not in display
    assert spoken is not None
    mgr.close()


@pytest.mark.asyncio
async def test_confirmation_safety_unaffected(tmp_path, mock_groq):
    db_file = str(tmp_path / "mgr_conf.db")
    mgr = SessionManager(db_path=db_file)

    agent_a = mgr.create_session(session_id="conf-sess-a")
    agent_b = mgr.create_session(session_id="conf-sess-b")

    agent_a.tool_registry.register(DummyConfirmedTool())
    agent_b.tool_registry.register(DummyConfirmedTool())

    mock_tc = MagicMock()
    mock_tc.id = "tc_mgr_1"
    mock_tc.function.name = "manager_dummy_action"
    mock_tc.function.arguments = json.dumps({})

    tool_comp = MagicMock()
    tool_choice = MagicMock()
    tool_msg = MagicMock()
    tool_msg.content = None
    tool_msg.tool_calls = [mock_tc]
    tool_choice.message = tool_msg
    tool_comp.choices = [tool_choice]
    tool_comp.usage = MagicMock(prompt_tokens=20, total_tokens=50)

    mock_groq.chat.completions.create.side_effect = AsyncMock(return_value=tool_comp)

    _, _, action = await agent_a.process_intent("Trigger confirmation")
    assert action is not None
    conf_id = action["confirmation_id"]

    # Session B attempting to execute Session A's confirmation ID must be rejected
    display_b, _, _ = await agent_b.handle_confirmation(conf_id, approved=True)
    assert "Confirmation failed: Unknown, expired, or already used" in display_b

    mgr.close()
