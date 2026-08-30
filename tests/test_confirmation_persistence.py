import asyncio
import json
import sqlite3
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agent.core import AgentCore, PendingConfirmation
from backend.storage.session_store import SessionStore
from backend.tools.base import BaseTool, PermissionLevel
from backend.tools.registry import ConfirmationRequiredException


class MockPrivilegedTool(BaseTool):
    def __init__(self, execution_counter):
        self.execution_counter = execution_counter

    @property
    def name(self) -> str:
        return "privileged_action"

    @property
    def description(self) -> str:
        return "Requires confirmation"

    @property
    def schema(self) -> dict:
        return {"type": "object", "properties": {"target": {"type": "string"}}}

    @property
    def permission(self) -> PermissionLevel:
        return PermissionLevel.CONFIRM_REQUIRED

    def execute(self, **kwargs) -> str:
        self.execution_counter["count"] += 1
        return f"Executed action on {kwargs.get('target', 'unknown')}"


@pytest.fixture
def memory_store():
    store = SessionStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def mock_groq():
    with patch("backend.agent.core.AsyncGroq") as mock_groq_cls:
        client_instance = MagicMock()
        mock_groq_cls.return_value = client_instance

        def make_tool_completion(conf_id="tool_call_1"):
            mock_comp = MagicMock()
            mock_choice = MagicMock()
            mock_msg = MagicMock()
            mock_msg.content = None
            mock_tc = MagicMock()
            mock_tc.id = conf_id
            mock_tc.function.name = "privileged_action"
            mock_tc.function.arguments = json.dumps({"target": "production_server"})
            mock_msg.tool_calls = [mock_tc]
            mock_choice.message = mock_msg
            mock_comp.choices = [mock_choice]
            mock_comp.usage = MagicMock(prompt_tokens=20, total_tokens=50)
            return mock_comp

        async def async_create(**kwargs):
            return make_tool_completion()

        client_instance.chat.completions.create = AsyncMock(side_effect=async_create)
        yield client_instance


@pytest.mark.asyncio
async def test_confirmation_belongs_to_session(memory_store, mock_groq):
    counter = {"count": 0}
    agent = AgentCore(session_store=memory_store, session_id="sess-alpha")
    agent.tool_registry.register(MockPrivilegedTool(counter))

    display, spoken, action = await agent.process_intent("Run privileged action")
    assert action is not None
    conf_id = action["confirmation_id"]

    pending = agent.pending_confirmations[conf_id]
    assert pending.session_id == "sess-alpha"


@pytest.mark.asyncio
async def test_cross_session_confirmation_rejection(memory_store, mock_groq):
    counter = {"count": 0}

    # Session Alpha creates pending confirmation
    agent_alpha = AgentCore(session_store=memory_store, session_id="sess-alpha")
    agent_alpha.tool_registry.register(MockPrivilegedTool(counter))
    display_a, spoken_a, action_a = await agent_alpha.process_intent("Action in Alpha")

    conf_id = action_a["confirmation_id"]

    # Session Beta attempts to execute Alpha's confirmation ID
    agent_beta = AgentCore(session_store=memory_store, session_id="sess-beta")
    agent_beta.tool_registry.register(MockPrivilegedTool(counter))

    # Share pending confirmations reference to test cross-session rejection logic
    agent_beta.pending_confirmations = agent_alpha.pending_confirmations

    display_b, spoken_b, action_b = await agent_beta.handle_confirmation(conf_id, approved=True)

    # Must be rejected because pending.session_id ("sess-alpha") != agent_beta.session_id ("sess-beta")
    assert counter["count"] == 0
    assert "Confirmation failed: Unknown, expired, or already used" in display_b


@pytest.mark.asyncio
async def test_confirmation_reconstruction_safety(memory_store, mock_groq):
    counter = {"count": 0}
    session_id = "reconstruct-conf-sess"

    agent_a = AgentCore(session_store=memory_store, session_id=session_id)
    agent_a.tool_registry.register(MockPrivilegedTool(counter))
    _, _, action = await agent_a.process_intent("Trigger confirmation")
    conf_id = action["confirmation_id"]

    # Destroy Agent A
    del agent_a

    # Instantiate Agent B (fresh in-memory pending confirmations)
    agent_b = AgentCore(session_store=memory_session_store if False else memory_store, session_id=session_id)
    agent_b.tool_registry.register(MockPrivilegedTool(counter))

    # Attempting to execute conf_id on reconstructed instance B must fail safely
    display, spoken, action_res = await agent_b.handle_confirmation(conf_id, approved=True)
    assert counter["count"] == 0
    assert "Confirmation failed: Unknown, expired, or already used" in display


@pytest.mark.asyncio
async def test_consumed_confirmation_cannot_replay(memory_store, mock_groq):
    counter = {"count": 0}
    agent = AgentCore(session_store=memory_store, session_id="replay-sess")
    agent.tool_registry.register(MockPrivilegedTool(counter))

    _, _, action = await agent.process_intent("Run tool")
    conf_id = action["confirmation_id"]

    # First approval -> succeeds
    display1, spoken1, _ = await agent.handle_confirmation(conf_id, approved=True)
    assert counter["count"] == 1

    # Second approval attempt -> fails safely
    display2, spoken2, _ = await agent.handle_confirmation(conf_id, approved=True)
    assert counter["count"] == 1
    assert "Confirmation failed: Unknown, expired, or already used" in display2


@pytest.mark.asyncio
async def test_rejected_confirmation_cannot_replay(memory_store, mock_groq):
    counter = {"count": 0}
    agent = AgentCore(session_store=memory_store, session_id="reject-sess")
    agent.tool_registry.register(MockPrivilegedTool(counter))

    _, _, action = await agent.process_intent("Run tool")
    conf_id = action["confirmation_id"]

    # First call -> Rejection
    display1, spoken1, _ = await agent.handle_confirmation(conf_id, approved=False)
    assert counter["count"] == 0

    # Second call -> Replay fails
    display2, spoken2, _ = await agent.handle_confirmation(conf_id, approved=True)
    assert counter["count"] == 0
    assert "Confirmation failed: Unknown, expired, or already used" in display2


@pytest.mark.asyncio
async def test_expired_confirmation_cannot_execute(memory_store, mock_groq):
    counter = {"count": 0}
    agent = AgentCore(session_store=memory_store, session_id="expire-sess")
    agent.tool_registry.register(MockPrivilegedTool(counter))

    _, _, action = await agent.process_intent("Run tool")
    conf_id = action["confirmation_id"]

    # Force expiration timestamp
    agent.pending_confirmations[conf_id].expires_at = time.time() - 10.0

    display, spoken, _ = await agent.handle_confirmation(conf_id, approved=True)
    assert counter["count"] == 0


@pytest.mark.asyncio
async def test_forged_confirmation_cannot_execute(memory_store, mock_groq):
    counter = {"count": 0}
    agent = AgentCore(session_store=memory_store, session_id="forged-sess")
    agent.tool_registry.register(MockPrivilegedTool(counter))

    display, spoken, _ = await agent.handle_confirmation("forged_id_9999", approved=True)
    assert counter["count"] == 0
    assert "Confirmation failed: Unknown, expired, or already used" in display


@pytest.mark.asyncio
async def test_clear_context_invalidates_confirmation(memory_store, mock_groq):
    counter = {"count": 0}
    agent = AgentCore(session_store=memory_store, session_id="clear-conf-sess")
    agent.tool_registry.register(MockPrivilegedTool(counter))

    _, _, action = await agent.process_intent("Run tool")
    conf_id = action["confirmation_id"]

    # Clear context
    agent.clear_context()

    # Attempting handle_confirmation after clear_context must fail
    display, spoken, _ = await agent.handle_confirmation(conf_id, approved=True)
    assert counter["count"] == 0
    assert "Confirmation failed: Unknown, expired, or already used" in display


@pytest.mark.asyncio
async def test_persistence_failure_safety(mock_groq):
    counter = {"count": 0}
    failing_store = MagicMock()
    failing_store.get_session.return_value = {"id": "fail-sess", "title": "Test"}
    failing_store.get_messages.return_value = []
    failing_store.add_message.side_effect = sqlite3.OperationalError("Disk full")

    agent = AgentCore(session_store=failing_store, session_id="fail-sess")
    agent.tool_registry.register(MockPrivilegedTool(counter))

    _, _, action = await agent.process_intent("Run tool")
    conf_id = action["confirmation_id"]

    display, spoken, _ = await agent.handle_confirmation(conf_id, approved=True)
    assert counter["count"] == 1
    assert display is not None


@pytest.mark.asyncio
async def test_exact_once_execution_under_concurrent_confirmation_attempts(memory_store, mock_groq):
    counter = {"count": 0}
    agent = AgentCore(session_store=memory_store, session_id="concurrent-conf-sess")
    agent.tool_registry.register(MockPrivilegedTool(counter))

    _, _, action = await agent.process_intent("Run tool")
    conf_id = action["confirmation_id"]

    # Fire 5 concurrent handle_confirmation calls
    tasks = [agent.handle_confirmation(conf_id, approved=True) for _ in range(5)]
    results = await asyncio.gather(*tasks)

    # Exactly 1 tool execution must occur
    assert counter["count"] == 1


@pytest.mark.asyncio
async def test_existing_phase45_formatting_remains_clean(memory_store, mock_groq):
    counter = {"count": 0}
    agent = AgentCore(session_store=memory_store, session_id="fmt-conf-sess")
    agent.tool_registry.register(MockPrivilegedTool(counter))

    _, _, action = await agent.process_intent("Run tool")
    conf_id = action["confirmation_id"]

    display, spoken, _ = await agent.handle_confirmation(conf_id, approved=True)
    assert "##" not in display
    assert "**" not in display
    assert spoken is not None
