import pytest
import asyncio
import json
import time
import threading
from backend.agent.core import AgentCore
from backend.tools.base import BaseTool
from backend.tools.permissions import PermissionLevel
from backend.tools.registry import ConfirmationRequiredException
from backend.agent.core import PendingConfirmation
from backend.main import ConfirmRequest

class DummyConfirmTool(BaseTool):
    @property
    def name(self) -> str: return "dummy_confirm"
    @property
    def description(self) -> str: return "Requires confirmation"
    @property
    def permission(self) -> PermissionLevel: return PermissionLevel.CONFIRM_REQUIRED
    @property
    def schema(self) -> dict: return {"type": "object", "properties": {"msg": {"type": "string"}}}
    def execute(self, msg: str = ""):
        return f"Executed: {msg}"

class DummySafeTool(BaseTool):
    @property
    def name(self) -> str: return "dummy_safe"
    @property
    def description(self) -> str: return "Safe execution"
    @property
    def permission(self) -> PermissionLevel: return PermissionLevel.SAFE
    @property
    def schema(self) -> dict: return {"type": "object", "properties": {"msg": {"type": "string"}}}
    def execute(self, msg: str = ""):
        return f"Safe executed: {msg}"

@pytest.fixture
def agent():
    agent = AgentCore(db_path=":memory:")
    agent.tool_registry.register(DummyConfirmTool())
    agent.tool_registry.register(DummySafeTool())
    return agent

@pytest.mark.asyncio
async def test_confirmation_exception():
    agent = AgentCore(db_path=":memory:")
    tool = DummyConfirmTool()
    agent.tool_registry.register(tool)

    with pytest.raises(ConfirmationRequiredException):
        agent.tool_registry.execute_tool("dummy_confirm", '{"msg": "test"}', "call_1")

@pytest.mark.asyncio
async def test_agentcore_pause(agent, monkeypatch):
    class MockMessage:
        content = None
        class MockToolCall:
            id = "call_1"
            class MockFunction:
                name = "dummy_confirm"
                arguments = '{"msg": "test"}'
            function = MockFunction()
        tool_calls = [MockToolCall()]

    class MockChoice:
        message = MockMessage()

    class MockCompletion:
        choices = [MockChoice()]
        usage = None

    async def mock_create(*args, **kwargs):
        return MockCompletion()

    monkeypatch.setattr(agent.client.chat.completions, "create", mock_create)

    resp, spoken, action = await agent.process_intent("do it")

    assert action is not None
    assert action["tool_name"] == "dummy_confirm"
    assert action["arguments"] == {"msg": "test"}
    assert "confirmation_id" in action
    assert action["confirmation_id"] in agent.pending_confirmations

    # Verify placeholder in history
    assert "[PENDING_CONFIRMATION_" in agent.conversation_history[-1]["content"]

@pytest.mark.asyncio
async def test_agentcore_approval(agent, monkeypatch):
    conf_id = "test_conf_id_1"
    agent.pending_confirmations[conf_id] = PendingConfirmation(
        confirmation_id=conf_id,
        tool_call_id="call_1",
        tool_name="dummy_confirm",
        arguments_json='{"msg": "test"}',
        created_at=time.time(),
        expires_at=time.time() + 300
    )

    agent.conversation_history.append({
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "dummy_confirm",
        "content": f"[PENDING_CONFIRMATION_{conf_id}]"
    })

    class MockMessage:
        content = "Done."
        tool_calls = None

    class MockChoice:
        message = MockMessage()

    class MockCompletion:
        choices = [MockChoice()]
        usage = None

    async def mock_create(*args, **kwargs):
        return MockCompletion()

    monkeypatch.setattr(agent.client.chat.completions, "create", mock_create)

    resp, spoken, action = await agent.handle_confirmation(conf_id, True)

    assert conf_id not in agent.pending_confirmations
    assert resp == "Done."
    assert "Executed: test" in agent.conversation_history[-2]["content"]

@pytest.mark.asyncio
async def test_agentcore_rejection(agent, monkeypatch):
    conf_id = "test_conf_id_2"
    agent.pending_confirmations[conf_id] = PendingConfirmation(
        confirmation_id=conf_id,
        tool_call_id="call_1",
        tool_name="dummy_confirm",
        arguments_json='{"msg": "test"}',
        created_at=time.time(),
        expires_at=time.time() + 300
    )

    agent.conversation_history.append({
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "dummy_confirm",
        "content": f"[PENDING_CONFIRMATION_{conf_id}]"
    })

    class MockMessage:
        content = "Understood."
        tool_calls = None

    class MockChoice:
        message = MockMessage()

    class MockCompletion:
        choices = [MockChoice()]
        usage = None

    async def mock_create(*args, **kwargs):
        return MockCompletion()

    monkeypatch.setattr(agent.client.chat.completions, "create", mock_create)

    resp, spoken, action = await agent.handle_confirmation(conf_id, False)

    assert conf_id not in agent.pending_confirmations
    assert "aborted" in agent.conversation_history[-2]["content"].lower()

@pytest.mark.asyncio
async def test_agentcore_expiration(agent, monkeypatch):
    conf_id = "test_conf_id_expired"
    agent.pending_confirmations[conf_id] = PendingConfirmation(
        confirmation_id=conf_id,
        tool_call_id="call_1",
        tool_name="dummy_confirm",
        arguments_json='{"msg": "test"}',
        created_at=time.time() - 400,
        expires_at=time.time() - 100
    )

    agent.conversation_history.append({
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "dummy_confirm",
        "content": f"[PENDING_CONFIRMATION_{conf_id}]"
    })

    class MockMessage:
        content = "Expired."
        tool_calls = None

    class MockChoice:
        message = MockMessage()

    class MockCompletion:
        choices = [MockChoice()]
        usage = None

    async def mock_create(*args, **kwargs):
        return MockCompletion()

    monkeypatch.setattr(agent.client.chat.completions, "create", mock_create)

    resp, spoken, action = await agent.handle_confirmation(conf_id, True)

    assert conf_id not in agent.pending_confirmations
    assert "expired" in agent.conversation_history[-2]["content"].lower()

@pytest.mark.asyncio
async def test_replay_protection(agent, monkeypatch):
    conf_id = "test_conf_id_replay"
    agent.pending_confirmations[conf_id] = PendingConfirmation(
        confirmation_id=conf_id,
        tool_call_id="call_1",
        tool_name="dummy_confirm",
        arguments_json='{"msg": "test"}',
        created_at=time.time(),
        expires_at=time.time() + 300
    )

    agent.conversation_history.append({
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "dummy_confirm",
        "content": f"[PENDING_CONFIRMATION_{conf_id}]"
    })

    class MockMessage:
        content = "Done."
        tool_calls = None
    class MockChoice:
        message = MockMessage()
    class MockCompletion:
        choices = [MockChoice()]
        usage = None

    async def mock_create(*args, **kwargs):
        return MockCompletion()
    monkeypatch.setattr(agent.client.chat.completions, "create", mock_create)

    resp, _, _ = await agent.handle_confirmation(conf_id, True)
    assert resp == "Done."

    resp2, _, _ = await agent.handle_confirmation(conf_id, True)
    assert "Confirmation failed" in resp2

@pytest.mark.asyncio
async def test_multi_tool_request(agent, monkeypatch):
    class MockMessage:
        content = None
        class MockToolCall1:
            id = "call_1"
            class MockFunction:
                name = "dummy_confirm"
                arguments = '{"msg": "test"}'
            function = MockFunction()

        class MockToolCall2:
            id = "call_2"
            class MockFunction:
                name = "dummy_safe"
                arguments = '{"msg": "safe"}'
            function = MockFunction()

        tool_calls = [MockToolCall1(), MockToolCall2()]

    class MockChoice:
        message = MockMessage()

    class MockCompletion:
        choices = [MockChoice()]
        usage = None

    async def mock_create(*args, **kwargs):
        return MockCompletion()

    monkeypatch.setattr(agent.client.chat.completions, "create", mock_create)

    resp, spoken, action = await agent.process_intent("do it")

    assert action is not None
    assert action["tool_name"] == "dummy_confirm"

    assert "[PENDING_CONFIRMATION_" in agent.conversation_history[-2]["content"]
    assert "Execution skipped" in agent.conversation_history[-1]["content"]

@pytest.mark.asyncio
async def test_concurrent_resolution(agent, monkeypatch):
    conf_id = "test_conf_id_concurrent"
    agent.pending_confirmations[conf_id] = PendingConfirmation(
        confirmation_id=conf_id,
        tool_call_id="call_1",
        tool_name="dummy_confirm",
        arguments_json='{"msg": "test"}',
        created_at=time.time(),
        expires_at=time.time() + 300
    )
    agent.conversation_history.append({
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "dummy_confirm",
        "content": f"[PENDING_CONFIRMATION_{conf_id}]"
    })

    class MockMessage:
        content = "Done."
        tool_calls = None
    class MockChoice:
        message = MockMessage()
    class MockCompletion:
        choices = [MockChoice()]
        usage = None

    async def mock_create(*args, **kwargs):
        return MockCompletion()
    monkeypatch.setattr(agent.client.chat.completions, "create", mock_create)

    task1 = asyncio.create_task(agent.handle_confirmation(conf_id, True))
    task2 = asyncio.create_task(agent.handle_confirmation(conf_id, True))

    results = await asyncio.gather(task1, task2)

    succeeded = [r for r in results if "Done" in r[0]]
    failed = [r for r in results if "failed" in r[0].lower()]

    assert len(succeeded) == 1
    assert len(failed) == 1

@pytest.mark.asyncio
async def test_tampering_is_impossible(agent):
    conf_id = "test_conf_id_tamper"
    agent.pending_confirmations[conf_id] = PendingConfirmation(
        confirmation_id=conf_id,
        tool_call_id="call_1",
        tool_name="dummy_confirm",
        arguments_json='{"msg": "ORIGINAL"}',
        created_at=time.time(),
        expires_at=time.time() + 300
    )

    fields = ConfirmRequest.model_fields
    assert "arguments" not in fields
    assert "tool_name" not in fields
    assert "tool_call_id" not in fields
    assert "permission" not in fields
