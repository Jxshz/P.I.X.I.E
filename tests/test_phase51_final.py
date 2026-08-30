import asyncio
import json
import sqlite3
import threading
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agent.core import AgentCore, PendingConfirmation
from backend.agent.personality import SYSTEM_PROMPT
from backend.storage.session_store import SessionStore
from backend.tools.base import BaseTool, PermissionLevel


class DummyActionTool(BaseTool):
    def __init__(self, execution_counter):
        self.execution_counter = execution_counter

    @property
    def name(self) -> str:
        return "dummy_action"

    @property
    def description(self) -> str:
        return "Action tool requiring confirmation"

    @property
    def schema(self) -> dict:
        return {"type": "object", "properties": {"item": {"type": "string"}}}

    @property
    def permission(self) -> PermissionLevel:
        return PermissionLevel.CONFIRM_REQUIRED

    def execute(self, **kwargs) -> str:
        self.execution_counter["count"] += 1
        return f"Executed item {kwargs.get('item', 'none')}"


@pytest.fixture
def mock_groq():
    with patch("backend.agent.core.AsyncGroq") as mock_groq_cls:
        client_instance = MagicMock()
        mock_groq_cls.return_value = client_instance

        def make_completion(content_text="Default response."):
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
            resp_text = getattr(client_instance, "_custom_response", "Mock response.")
            return make_completion(resp_text)

        client_instance.chat.completions.create = AsyncMock(side_effect=async_create)
        yield client_instance


def set_mock_response(client_instance, text: str):
    client_instance._custom_response = text


@pytest.mark.asyncio
async def test_full_persistence_round_trip(tmp_path, mock_groq):
    db_file = str(tmp_path / "roundtrip.db")
    store = SessionStore(db_file)
    session_id = "sess-rt"

    agent = AgentCore(session_store=store, session_id=session_id)
    set_mock_response(mock_groq, "Response turn 1")
    await agent.process_intent("Prompt turn 1")

    set_mock_response(mock_groq, "Response turn 2")
    await agent.process_intent("Prompt turn 2")

    messages = store.get_messages(session_id)
    assert len(messages) == 4
    assert messages[0]["content"] == "Prompt turn 1"
    assert messages[1]["content"] == "Response turn 1"
    assert messages[2]["content"] == "Prompt turn 2"
    assert messages[3]["content"] == "Response turn 2"
    store.close()


@pytest.mark.asyncio
async def test_agentcore_destruction_reconstruction(tmp_path, mock_groq):
    db_file = str(tmp_path / "reconstruct_final.db")
    store = SessionStore(db_file)
    session_id = "sess-reconstruct-final"

    # Instance A
    agent_a = AgentCore(session_store=store, session_id=session_id)
    set_mock_response(mock_groq, "Memory state persisted.")
    await agent_a.process_intent("Test input")
    del agent_a

    # Instance B
    agent_b = AgentCore(session_store=store, session_id=session_id)
    assert len(agent_b.conversation_history) == 3
    assert agent_b.conversation_history[0]["role"] == "system"
    assert agent_b.conversation_history[1]["content"] == "Test input"
    assert agent_b.conversation_history[2]["content"] == "Memory state persisted."
    store.close()


@pytest.mark.asyncio
async def test_long_conversation_persistence(tmp_path, mock_groq):
    db_file = str(tmp_path / "long_conv.db")
    store = SessionStore(db_file)
    session_id = "sess-long-conv"

    agent = AgentCore(session_store=store, session_id=session_id)
    for i in range(12):
        set_mock_response(mock_groq, f"Answer {i} " + "detail " * 20)
        await agent.process_intent(f"Question {i} " + "context " * 20)

    persisted = store.get_messages(session_id)
    assert len(persisted) == 24
    store.close()


def test_trimming_vs_database_history(tmp_path):
    db_file = str(tmp_path / "trim_vs_db.db")
    store = SessionStore(db_file)
    session_id = "sess-trim-vs-db"

    store.create_session(session_id=session_id)
    for i in range(25):
        store.add_message(session_id, "user", f"User prompt {i} " + "data " * 40)
        store.add_message(session_id, "assistant", f"Assistant reply {i} " + "info " * 40)

    agent = AgentCore(session_store=store, session_id=session_id)
    agent._trim_context()

    # Active working memory is bounded
    assert len(agent.conversation_history) < 51
    assert agent.conversation_history[0]["role"] == "system"

    # SQLite long-term records remain 100% complete (50 records)
    db_messages = store.get_messages(session_id)
    assert len(db_messages) == 50
    store.close()


def test_malformed_persisted_messages(tmp_path):
    db_file = str(tmp_path / "malformed_final.db")
    store = SessionStore(db_file)
    session_id = "sess-malformed-final"

    store.create_session(session_id=session_id)
    store.add_message(session_id, "user", "Healthy message")
    store.add_message(session_id, "assistant", "Corrupt payload", tool_calls_json="{invalid: json,")

    agent = AgentCore(session_store=store, session_id=session_id)
    assert len(agent.conversation_history) == 3
    assert agent.conversation_history[1]["content"] == "Healthy message"
    assert agent.conversation_history[2]["content"] == "Corrupt payload"
    store.close()


@pytest.mark.asyncio
async def test_persistence_failure_recovery(mock_groq):
    failing_store = MagicMock()
    failing_store.get_session.return_value = {"id": "fail-recovery", "title": "Test"}
    failing_store.get_messages.return_value = []
    failing_store.add_message.side_effect = sqlite3.OperationalError("Database disk image is malformed")

    agent = AgentCore(session_store=failing_store, session_id="fail-recovery")
    set_mock_response(mock_groq, "Recovery response")

    display, spoken, _ = await agent.process_intent("Input during DB error")
    assert display is not None
    assert spoken is not None
    assert len(agent.conversation_history) == 3


def test_clear_context_consistency(tmp_path):
    db_file = str(tmp_path / "clear_consistency.db")
    store = SessionStore(db_file)
    session_id = "sess-clear-cons"

    agent = AgentCore(session_store=store, session_id=session_id)
    store.add_message(session_id, "user", "Message before clear")

    agent.clear_context()
    assert len(agent.conversation_history) == 1
    assert agent.conversation_history[0]["content"] == SYSTEM_PROMPT

    db_messages = store.get_messages(session_id)
    assert len(db_messages) == 0
    store.close()


@pytest.mark.asyncio
async def test_confirmation_invalidation(tmp_path, mock_groq):
    db_file = str(tmp_path / "conf_inval.db")
    store = SessionStore(db_file)
    session_id = "sess-conf-inval"

    counter = {"count": 0}
    agent = AgentCore(session_store=store, session_id=session_id)
    agent.tool_registry.register(DummyActionTool(counter))

    # Mock tool call completion
    mock_tc = MagicMock()
    mock_tc.id = "tc_conf_1"
    mock_tc.function.name = "dummy_action"
    mock_tc.function.arguments = json.dumps({"item": "test_box"})

    tool_comp = MagicMock()
    tool_choice = MagicMock()
    tool_msg = MagicMock()
    tool_msg.content = None
    tool_msg.tool_calls = [mock_tc]
    tool_choice.message = tool_msg
    tool_comp.choices = [tool_choice]
    tool_comp.usage = MagicMock(prompt_tokens=20, total_tokens=50)

    mock_groq.chat.completions.create.side_effect = AsyncMock(return_value=tool_comp)

    display, spoken, action = await agent.process_intent("Execute dummy action")
    assert action is not None
    conf_id = action["confirmation_id"]

    # Clear context invalidates pending confirmation for session
    agent.clear_context()

    display_res, _, _ = await agent.handle_confirmation(conf_id, approved=True)
    assert counter["count"] == 0
    assert "Confirmation failed: Unknown, expired, or already used" in display_res
    store.close()


def test_tool_metadata_round_trip(tmp_path):
    db_file = str(tmp_path / "tool_rt.db")
    store = SessionStore(db_file)
    session_id = "sess-tool-rt"
    store.create_session(session_id=session_id)

    tc_json = json.dumps([{"id": "call_123", "type": "function", "function": {"name": "dummy_action", "arguments": '{"item":"A"}'}}])
    store.add_message(session_id, "assistant", "Calling tool...", tool_calls_json=tc_json)
    store.add_message(session_id, "tool", '{"result":"Executed A"}', tool_calls_json=json.dumps({"tool_call_id": "call_123", "name": "dummy_action"}))

    agent = AgentCore(session_store=store, session_id=session_id)
    assert len(agent.conversation_history) == 3
    assert agent.conversation_history[1]["tool_calls"][0]["id"] == "call_123"
    assert agent.conversation_history[2]["tool_call_id"] == "call_123"
    assert agent.conversation_history[2]["name"] == "dummy_action"
    store.close()


def test_concurrent_persistence_access(tmp_path):
    db_file = str(tmp_path / "concurrent_store.db")
    store = SessionStore(db_file)
    session_id = "sess-concurrent-store"
    store.create_session(session_id=session_id)

    errors = []

    def worker(worker_id):
        try:
            for i in range(10):
                store.add_message(session_id, "user", f"Worker {worker_id} msg {i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    messages = store.get_messages(session_id)
    assert len(messages) == 50
    store.close()


@pytest.mark.asyncio
async def test_unicode_large_payload_safety(tmp_path, mock_groq):
    db_file = str(tmp_path / "unicode_large.db")
    store = SessionStore(db_file)
    session_id = "sess-unicode-large"

    agent1 = AgentCore(session_store=store, session_id=session_id)
    unicode_q = "Multi-script test: 🚀 こんにちは, Привет, 🤖. " + "payload " * 500
    unicode_a = "Multi-script reply: ✅ 💬 ⚡️. " + "content " * 500

    set_mock_response(mock_groq, unicode_a)
    await agent1.process_intent(unicode_q)

    agent2 = AgentCore(session_store=store, session_id=session_id)
    assert agent2.conversation_history[1]["content"] == unicode_q
    assert agent2.conversation_history[2]["content"] == unicode_a
    store.close()


@pytest.mark.asyncio
async def test_phase45_formatting_regression(tmp_path, mock_groq):
    db_file = str(tmp_path / "fmt_reg.db")
    store = SessionStore(db_file)
    session_id = "sess-fmt-reg"

    agent = AgentCore(session_store=store, session_id=session_id)
    set_mock_response(mock_groq, "## Heading\n- Item 1\n- Item 2\n**Bold text**")

    display, spoken, _ = await agent.process_intent("Give me formatted text")
    assert "##" not in display
    assert "**" not in display
    assert spoken is not None
    store.close()
