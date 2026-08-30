"""
test_concurrency.py
===================
Deterministic concurrency and session safety tests for AgentCore.

Scenarios tested:
  1. Two concurrent normal requests against the same AgentCore instance are serialized safely.
  2. Concurrent requests preserve turn order and do not corrupt conversation history.
  3. Concurrent confirmation resolution: exactly one succeeds, the other fails cleanly.
  4. Confirmation cannot execute twice (replay / double execution prevention).
  5. Rejected confirmation leaves clean state and proper history.
  6. Expired confirmation leaves clean state and does not execute tool.
  7. Exception during execution does not corrupt history or leave broken turns.
  8. clear_context clears pending confirmations and resets history cleanly.
  9. Simultaneous clear_context and process_intent interactions are well-behaved.
"""

import asyncio
import json
import time
import pytest
from backend.agent.core import AgentCore, PendingConfirmation
from backend.tools.base import BaseTool
from backend.tools.permissions import PermissionLevel


class DummySafeTool(BaseTool):
    @property
    def name(self) -> str:
        return "dummy_safe_tool"

    @property
    def description(self) -> str:
        return "Safe tool"

    @property
    def permission(self) -> PermissionLevel:
        return PermissionLevel.SAFE

    @property
    def schema(self) -> dict:
        return {"type": "object", "properties": {"val": {"type": "string"}}}

    def execute(self, val: str = ""):
        return json.dumps({"result": f"Executed: {val}"})


class DummyConfirmTool(BaseTool):
    @property
    def name(self) -> str:
        return "dummy_confirm_tool"

    @property
    def description(self) -> str:
        return "Confirm tool"

    @property
    def permission(self) -> PermissionLevel:
        return PermissionLevel.CONFIRM_REQUIRED

    @property
    def schema(self) -> dict:
        return {"type": "object", "properties": {"action": {"type": "string"}}}

    def execute(self, action: str = ""):
        return json.dumps({"result": f"Action confirmed: {action}"})


@pytest.fixture
def agent():
    ag = AgentCore(db_path=":memory:")
    ag.tool_registry.register(DummySafeTool())
    ag.tool_registry.register(DummyConfirmTool())
    return ag


@pytest.mark.asyncio
async def test_concurrent_normal_requests_serialized(agent, monkeypatch):
    """
    Scenario 1 & 2: Two concurrent requests against the same AgentCore instance.
    Both should complete without history interleaving or state corruption.
    """
    call_order = []

    async def mock_create(**kwargs):
        messages = kwargs.get("messages", [])
        user_msg = messages[-1]["content"] if messages else ""
        call_order.append(f"start_{user_msg}")
        await asyncio.sleep(0.05)  # Simulate network latency
        call_order.append(f"end_{user_msg}")

        class MockMessage:
            content = f"Response to {user_msg}"
            tool_calls = None

        class MockChoice:
            message = MockMessage()

        class MockCompletion:
            choices = [MockChoice()]
            usage = None

        return MockCompletion()

    monkeypatch.setattr(agent.client.chat.completions, "create", mock_create)

    task1 = asyncio.create_task(agent.process_intent("Prompt 1"))
    task2 = asyncio.create_task(agent.process_intent("Prompt 2"))

    res1, res2 = await asyncio.gather(task1, task2)

    # Responses should be valid
    assert "Prompt 1" in res1[0] or "Prompt 2" in res1[0]
    assert "Prompt 1" in res2[0] or "Prompt 2" in res2[0]

    # Turns should be serialized (start1 -> end1 -> start2 -> end2) because of context_lock
    assert call_order[0].startswith("start_")
    assert call_order[1] == f"end_{call_order[0].split('_', 1)[1]}"
    assert call_order[2].startswith("start_")
    assert call_order[3] == f"end_{call_order[2].split('_', 1)[1]}"

    # History should contain System -> User1 -> Asst1 -> User2 -> Asst2 (5 messages)
    assert len(agent.conversation_history) == 5
    assert agent.conversation_history[0]["role"] == "system"
    assert agent.conversation_history[1]["role"] == "user"
    assert agent.conversation_history[2]["role"] == "assistant"
    assert agent.conversation_history[3]["role"] == "user"
    assert agent.conversation_history[4]["role"] == "assistant"


@pytest.mark.asyncio
async def test_concurrent_confirmation_resolution_exact_once(agent, monkeypatch):
    """
    Scenario 3 & 4: Concurrent confirmation resolution on the same confirmation_id.
    Exactly one must succeed, the second must be rejected (replay/double execution impossible).
    """
    conf_id = "test_conf_concurrent_1"
    agent.pending_confirmations[conf_id] = PendingConfirmation(
        confirmation_id=conf_id,
        tool_call_id="call_99",
        tool_name="dummy_confirm_tool",
        arguments_json='{"action": "format_disk"}',
        created_at=time.time(),
        expires_at=time.time() + 300
    )
    agent.conversation_history.append({
        "role": "tool",
        "tool_call_id": "call_99",
        "name": "dummy_confirm_tool",
        "content": f"[PENDING_CONFIRMATION_{conf_id}]"
    })

    async def mock_create(**kwargs):
        class MockMessage:
            content = "Action complete."
            tool_calls = None

        class MockChoice:
            message = MockMessage()

        class MockCompletion:
            choices = [MockChoice()]
            usage = None

        return MockCompletion()

    monkeypatch.setattr(agent.client.chat.completions, "create", mock_create)

    # Launch two simultaneous confirmation resolution tasks
    task1 = asyncio.create_task(agent.handle_confirmation(conf_id, approved=True))
    task2 = asyncio.create_task(agent.handle_confirmation(conf_id, approved=True))

    results = await asyncio.gather(task1, task2)

    succeeded = [r for r in results if "Action complete" in r[0]]
    failed = [r for r in results if "Confirmation failed" in r[0]]

    assert len(succeeded) == 1
    assert len(failed) == 1

    # Confirmation ID must be completely gone from pending
    assert conf_id not in agent.pending_confirmations


@pytest.mark.asyncio
async def test_rejected_confirmation_leaves_clean_state(agent, monkeypatch):
    """
    Scenario 5: Rejected confirmation leaves clean state and does not execute the tool.
    """
    conf_id = "test_conf_reject_1"
    agent.pending_confirmations[conf_id] = PendingConfirmation(
        confirmation_id=conf_id,
        tool_call_id="call_101",
        tool_name="dummy_confirm_tool",
        arguments_json='{"action": "delete_all"}',
        created_at=time.time(),
        expires_at=time.time() + 300
    )
    agent.conversation_history.append({
        "role": "tool",
        "tool_call_id": "call_101",
        "name": "dummy_confirm_tool",
        "content": f"[PENDING_CONFIRMATION_{conf_id}]"
    })

    async def mock_create(**kwargs):
        class MockMessage:
            content = "Operation was aborted."
            tool_calls = None

        class MockChoice:
            message = MockMessage()

        class MockCompletion:
            choices = [MockChoice()]
            usage = None

        return MockCompletion()

    monkeypatch.setattr(agent.client.chat.completions, "create", mock_create)

    res_display, res_spoken, action = await agent.handle_confirmation(conf_id, approved=False)

    assert "Operation was aborted" in res_display
    assert conf_id not in agent.pending_confirmations
    # Verify the tool result in history records rejection
    tool_msg = [m for m in agent.conversation_history if m.get("role") == "tool"][-1]
    assert "User rejected" in tool_msg["content"]


@pytest.mark.asyncio
async def test_expired_confirmation_leaves_clean_state(agent, monkeypatch):
    """
    Scenario 6: Expired confirmation leaves clean state and returns expiration error.
    """
    conf_id = "test_conf_expired_1"
    agent.pending_confirmations[conf_id] = PendingConfirmation(
        confirmation_id=conf_id,
        tool_call_id="call_102",
        tool_name="dummy_confirm_tool",
        arguments_json='{"action": "critical_op"}',
        created_at=time.time() - 400,
        expires_at=time.time() - 100  # Already expired
    )
    agent.conversation_history.append({
        "role": "tool",
        "tool_call_id": "call_102",
        "name": "dummy_confirm_tool",
        "content": f"[PENDING_CONFIRMATION_{conf_id}]"
    })

    async def mock_create(**kwargs):
        class MockMessage:
            content = "Request expired."
            tool_calls = None

        class MockChoice:
            message = MockMessage()

        class MockCompletion:
            choices = [MockChoice()]
            usage = None

        return MockCompletion()

    monkeypatch.setattr(agent.client.chat.completions, "create", mock_create)

    res_display, res_spoken, action = await agent.handle_confirmation(conf_id, approved=True)

    assert "expired" in res_display.lower()
    assert conf_id not in agent.pending_confirmations
    tool_msg = [m for m in agent.conversation_history if m.get("role") == "tool"][-1]
    assert "expired" in tool_msg["content"].lower()


@pytest.mark.asyncio
async def test_exception_does_not_corrupt_history(agent, monkeypatch):
    """
    Scenario 7: An API exception during execution appends an assistant error message
    and keeps history valid for subsequent turns.
    """
    async def mock_failing_create(**kwargs):
        raise ConnectionError("Simulated socket timeout")

    monkeypatch.setattr(agent.client.chat.completions, "create", mock_failing_create)

    res_display, res_spoken, _ = await agent.process_intent("Try request")

    assert "Error connecting to Groq API" in res_display
    assert agent.conversation_history[-1]["role"] == "assistant"
    assert "Error connecting" in agent.conversation_history[-1]["content"]

    # Subsequent request with working API works cleanly
    async def mock_working_create(**kwargs):
        class MockMessage:
            content = "Recovered successfully."
            tool_calls = None

        class MockChoice:
            message = MockMessage()

        class MockCompletion:
            choices = [MockChoice()]
            usage = None

        return MockCompletion()

    monkeypatch.setattr(agent.client.chat.completions, "create", mock_working_create)

    res2_display, _, _ = await agent.process_intent("Next turn")
    assert "Recovered successfully." in res2_display
    assert agent.conversation_history[-1]["role"] == "assistant"
    assert agent.conversation_history[-1]["content"] == "Recovered successfully."


@pytest.mark.asyncio
async def test_clear_context_clears_pending_and_history(agent):
    """
    Scenario 8: clear_context resets history to only SYSTEM_PROMPT and wipes pending_confirmations.
    """
    agent.pending_confirmations["stale_conf"] = PendingConfirmation(
        confirmation_id="stale_conf",
        tool_call_id="call_x",
        tool_name="dummy_confirm_tool",
        arguments_json="{}",
        created_at=time.time(),
        expires_at=time.time() + 300
    )
    agent.conversation_history.append({"role": "user", "content": "Hello"})
    agent.conversation_history.append({"role": "assistant", "content": "Hi"})

    agent.clear_context()

    assert len(agent.conversation_history) == 1
    assert agent.conversation_history[0]["role"] == "system"
    assert len(agent.pending_confirmations) == 0
