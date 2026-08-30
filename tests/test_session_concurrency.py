import asyncio
import json
import sqlite3
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agent.core import AgentCore, RateLimitException
from backend.agent.session_manager import SessionManager
from backend.storage.session_store import SessionStore
from backend.tools.base import BaseTool, PermissionLevel


class SlowActionTool(BaseTool):
    def __init__(self, start_event: asyncio.Event, release_event: asyncio.Event):
        self.start_event = start_event
        self.release_event = release_event

    @property
    def name(self) -> str:
        return "slow_action"

    @property
    def description(self) -> str:
        return "Tool for testing concurrency blocking"

    @property
    def schema(self) -> dict:
        return {"type": "object", "properties": {}}

    @property
    def permission(self) -> PermissionLevel:
        return PermissionLevel.SAFE

    def execute(self, **kwargs) -> str:
        # Signalled when tool starts
        return "Slow action complete"


@pytest.fixture
def mock_groq():
    with patch("backend.agent.core.AsyncGroq") as mock_groq_cls:
        client_instance = MagicMock()
        mock_groq_cls.return_value = client_instance

        def make_completion(content_text="Concurrency response."):
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
            resp_text = getattr(client_instance, "_custom_response", "Concurrency response.")
            return make_completion(resp_text)

        client_instance.chat.completions.create = AsyncMock(side_effect=async_create)
        yield client_instance


def set_mock_response(client_instance, text: str):
    client_instance._custom_response = text


@pytest.mark.asyncio
async def test_same_session_turns_serialized(tmp_path, mock_groq):
    db_file = str(tmp_path / "conc_same.db")
    mgr = SessionManager(db_path=db_file)
    sid = "sess-same-serial"

    execution_order = []

    # Custom mock completion that tracks execution order
    async def delayed_create(**kwargs):
        messages = kwargs.get("messages", [])
        last_input = messages[-1].get("content", "")
        execution_order.append(f"start:{last_input}")
        await asyncio.sleep(0.02)
        execution_order.append(f"finish:{last_input}")
        mock_comp = MagicMock()
        mock_choice = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = f"Response to {last_input}"
        mock_msg.tool_calls = None
        mock_choice.message = mock_msg
        mock_comp.choices = [mock_choice]
        mock_comp.usage = MagicMock(prompt_tokens=10, total_tokens=30)
        return mock_comp

    mock_groq.chat.completions.create.side_effect = delayed_create

    # Launch two turns concurrently on the SAME session
    t1 = asyncio.create_task(mgr.process_intent(sid, "Turn 1"))
    t2 = asyncio.create_task(mgr.process_intent(sid, "Turn 2"))

    await asyncio.gather(t1, t2)

    # Must be strictly serialized: start:Turn 1 -> finish:Turn 1 -> start:Turn 2 -> finish:Turn 2
    assert execution_order == [
        "start:Turn 1",
        "finish:Turn 1",
        "start:Turn 2",
        "finish:Turn 2",
    ]
    mgr.close()


@pytest.mark.asyncio
async def test_different_sessions_execute_concurrently(tmp_path, mock_groq):
    db_file = str(tmp_path / "conc_diff.db")
    mgr = SessionManager(db_path=db_file)

    a_started = asyncio.Event()
    a_can_finish = asyncio.Event()
    b_finished = asyncio.Event()

    async def custom_create(**kwargs):
        messages = kwargs.get("messages", [])
        last_input = messages[-1].get("content", "")

        if "Session A" in last_input:
            a_started.set()
            await a_can_finish.wait()
        elif "Session B" in last_input:
            b_finished.set()

        mock_comp = MagicMock()
        mock_choice = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = f"Done {last_input}"
        mock_msg.tool_calls = None
        mock_choice.message = mock_msg
        mock_comp.choices = [mock_choice]
        mock_comp.usage = MagicMock(prompt_tokens=10, total_tokens=30)
        return mock_comp

    mock_groq.chat.completions.create.side_effect = custom_create

    task_a = asyncio.create_task(mgr.process_intent("sess-a", "Session A Turn"))

    # Wait until Session A has acquired its lock and entered API call
    await a_started.wait()

    # Session B executes while Session A is still holding its turn API call
    task_b = asyncio.create_task(mgr.process_intent("sess-b", "Session B Turn"))

    # Session B completes completely while Session A is paused
    await b_finished.wait()
    assert task_b.done()
    assert not task_a.done()

    # Allow Session A to finish
    a_can_finish.set()
    await task_a

    mgr.close()


@pytest.mark.asyncio
async def test_three_or_more_sessions_concurrent(tmp_path, mock_groq):
    db_file = str(tmp_path / "conc_three.db")
    mgr = SessionManager(db_path=db_file)

    active_count = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def concurrency_tracker(**kwargs):
        nonlocal active_count, max_concurrent
        async with lock:
            active_count += 1
            if active_count > max_concurrent:
                max_concurrent = active_count

        await asyncio.sleep(0.01)

        async with lock:
            active_count -= 1

        mock_comp = MagicMock()
        mock_choice = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "Done"
        mock_msg.tool_calls = None
        mock_choice.message = mock_msg
        mock_comp.choices = [mock_choice]
        mock_comp.usage = MagicMock(prompt_tokens=10, total_tokens=30)
        return mock_comp

    mock_groq.chat.completions.create.side_effect = concurrency_tracker

    tasks = [
        asyncio.create_task(mgr.process_intent(f"sess-{i}", f"Turn {i}"))
        for i in range(5)
    ]
    await asyncio.gather(*tasks)

    # All 5 independent sessions overlap concurrently
    assert max_concurrent > 1
    mgr.close()


@pytest.mark.asyncio
async def test_same_session_execution_order_deterministic(tmp_path, mock_groq):
    db_file = str(tmp_path / "conc_order.db")
    mgr = SessionManager(db_path=db_file)
    sid = "sess-order-det"

    set_mock_response(mock_groq, "OK")

    results = []
    for i in range(5):
        disp, _, _ = await mgr.process_intent(sid, f"Step {i}")
        results.append(disp)

    agent = mgr.get_session(sid)
    user_msgs = [m["content"] for m in agent.conversation_history if m["role"] == "user"]
    assert user_msgs == ["Step 0", "Step 1", "Step 2", "Step 3", "Step 4"]
    mgr.close()


@pytest.mark.asyncio
async def test_lock_released_after_success(tmp_path, mock_groq):
    db_file = str(tmp_path / "conc_rel_succ.db")
    mgr = SessionManager(db_path=db_file)
    sid = "sess-rel-succ"

    set_mock_response(mock_groq, "Success response")
    await mgr.process_intent(sid, "First turn")

    lock = mgr.get_session_lock(sid)
    assert not lock.locked()
    mgr.close()


@pytest.mark.asyncio
async def test_lock_released_after_exception(tmp_path, mock_groq):
    db_file = str(tmp_path / "conc_rel_exc.db")
    mgr = SessionManager(db_path=db_file)
    sid = "sess-rel-exc"

    agent = mgr.get_or_create_session(sid)
    with patch.object(agent.governor, "preflight", return_value=(False, "Rate limited", None)):
        with pytest.raises(RateLimitException):
            await mgr.process_intent(sid, "Turn that crashes")

    lock = mgr.get_session_lock(sid)
    assert not lock.locked()
    mgr.close()


@pytest.mark.asyncio
async def test_lock_released_after_error_turn(tmp_path, mock_groq):
    db_file = str(tmp_path / "conc_rel_err.db")
    mgr = SessionManager(db_path=db_file)
    sid = "sess-rel-err"

    mock_groq.chat.completions.create.side_effect = Exception("Connection Timeout")

    disp, spoken, action = await mgr.process_intent(sid, "Failing request")
    assert "Error connecting to Groq API" in disp

    lock = mgr.get_session_lock(sid)
    assert not lock.locked()
    mgr.close()


@pytest.mark.asyncio
async def test_one_blocked_session_does_not_block_another(tmp_path, mock_groq):
    db_file = str(tmp_path / "conc_no_block.db")
    mgr = SessionManager(db_path=db_file)

    a_block_event = asyncio.Event()

    async def slow_create_a(**kwargs):
        messages = kwargs.get("messages", [])
        if "Slow Turn" in messages[-1].get("content", ""):
            await a_block_event.wait()
        mock_comp = MagicMock()
        mock_choice = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "Finished"
        mock_msg.tool_calls = None
        mock_choice.message = mock_msg
        mock_comp.choices = [mock_choice]
        mock_comp.usage = MagicMock(prompt_tokens=10, total_tokens=30)
        return mock_comp

    mock_groq.chat.completions.create.side_effect = slow_create_a

    task_a = asyncio.create_task(mgr.process_intent("sess-blocked", "Slow Turn"))

    # Session B executes while Session A is blocked indefinitely on a_block_event
    disp_b, _, _ = await mgr.process_intent("sess-free", "Fast Turn")
    assert disp_b is not None

    # Unblock A
    a_block_event.set()
    await task_a
    mgr.close()


def test_session_lock_isolation(tmp_path):
    db_file = str(tmp_path / "conc_lock_iso.db")
    mgr = SessionManager(db_path=db_file)

    lock_a = mgr.get_session_lock("sess-a")
    lock_b = mgr.get_session_lock("sess-b")

    assert lock_a is not lock_b
    mgr.close()


@pytest.mark.asyncio
async def test_concurrent_same_session_history_consistent(tmp_path, mock_groq):
    db_file = str(tmp_path / "conc_hist_cons.db")
    mgr = SessionManager(db_path=db_file)
    sid = "sess-hist-cons"

    set_mock_response(mock_groq, "Response")

    tasks = [
        asyncio.create_task(mgr.process_intent(sid, f"Msg {i}"))
        for i in range(4)
    ]
    await asyncio.gather(*tasks)

    agent = mgr.get_session(sid)
    user_msgs = [m for m in agent.conversation_history if m["role"] == "user"]
    assert len(user_msgs) == 4

    persisted = mgr.session_store.get_messages(sid)
    assert len(persisted) == 8
    mgr.close()


@pytest.mark.asyncio
async def test_concurrent_different_session_histories_isolated(tmp_path, mock_groq):
    db_file = str(tmp_path / "conc_hist_iso.db")
    mgr = SessionManager(db_path=db_file)

    set_mock_response(mock_groq, "Response")

    t_a = asyncio.create_task(mgr.process_intent("sess-a", "Prompt A"))
    t_b = asyncio.create_task(mgr.process_intent("sess-b", "Prompt B"))
    await asyncio.gather(t_a, t_b)

    agent_a = mgr.get_session("sess-a")
    agent_b = mgr.get_session("sess-b")

    assert [m["content"] for m in agent_a.conversation_history if m["role"] == "user"] == ["Prompt A"]
    assert [m["content"] for m in agent_b.conversation_history if m["role"] == "user"] == ["Prompt B"]
    mgr.close()


@pytest.mark.asyncio
async def test_concurrent_confirmation_attempts_exact_once(tmp_path, mock_groq):
    db_file = str(tmp_path / "conc_conf_exact.db")
    mgr = SessionManager(db_path=db_file)
    sid = "sess-conf-exact"

    agent = mgr.create_session(session_id=sid)

    # Register mock tool requiring confirmation
    counter = {"count": 0}

    class ConfTool(BaseTool):
        @property
        def name(self) -> str: return "conf_action"
        @property
        def description(self) -> str: return "Conf tool"
        @property
        def schema(self) -> dict: return {}
        @property
        def permission(self) -> PermissionLevel: return PermissionLevel.CONFIRM_REQUIRED
        def execute(self, **kwargs) -> str:
            counter["count"] += 1
            return "Executed"

    agent.tool_registry.register(ConfTool())

    mock_tc = MagicMock()
    mock_tc.id = "tc_exact_1"
    mock_tc.function.name = "conf_action"
    mock_tc.function.arguments = "{}"

    tool_comp = MagicMock()
    tool_choice = MagicMock()
    tool_msg = MagicMock()
    tool_msg.content = None
    tool_msg.tool_calls = [mock_tc]
    tool_choice.message = tool_msg
    tool_comp.choices = [tool_choice]
    tool_comp.usage = MagicMock(prompt_tokens=20, total_tokens=50)

    mock_groq.chat.completions.create.side_effect = AsyncMock(return_value=tool_comp)

    _, _, action = await agent.process_intent("Run tool")
    conf_id = action["confirmation_id"]

    # Concurrent confirmation resolutions on Manager
    tasks = [
        asyncio.create_task(mgr.handle_confirmation(sid, conf_id, approved=True))
        for _ in range(4)
    ]
    await asyncio.gather(*tasks)

    # Tool executed EXACTLY ONCE
    assert counter["count"] == 1
    mgr.close()


@pytest.mark.asyncio
async def test_cross_session_confirmation_rejected_under_concurrency(tmp_path, mock_groq):
    db_file = str(tmp_path / "conc_cross_conf.db")
    mgr = SessionManager(db_path=db_file)

    agent_a = mgr.create_session(session_id="sess-conf-a")
    agent_b = mgr.create_session(session_id="sess-conf-b")

    counter = {"count": 0}

    class CrossTool(BaseTool):
        @property
        def name(self) -> str: return "cross_action"
        @property
        def description(self) -> str: return "Cross tool"
        @property
        def schema(self) -> dict: return {}
        @property
        def permission(self) -> PermissionLevel: return PermissionLevel.CONFIRM_REQUIRED
        def execute(self, **kwargs) -> str:
            counter["count"] += 1
            return "Executed"

    agent_a.tool_registry.register(CrossTool())
    agent_b.tool_registry.register(CrossTool())

    mock_tc = MagicMock()
    mock_tc.id = "tc_cross_1"
    mock_tc.function.name = "cross_action"
    mock_tc.function.arguments = "{}"

    tool_comp = MagicMock()
    tool_choice = MagicMock()
    tool_msg = MagicMock()
    tool_msg.content = None
    tool_msg.tool_calls = [mock_tc]
    tool_choice.message = tool_msg
    tool_comp.choices = [tool_choice]
    tool_comp.usage = MagicMock(prompt_tokens=20, total_tokens=50)

    mock_groq.chat.completions.create.side_effect = AsyncMock(return_value=tool_comp)

    _, _, action = await agent_a.process_intent("Run tool in A")
    conf_id = action["confirmation_id"]

    disp_b, _, _ = await mgr.handle_confirmation("sess-conf-b", conf_id, approved=True)
    assert counter["count"] == 0
    assert "Confirmation failed: Unknown, expired, or already used" in disp_b
    mgr.close()


def test_session_removal_cleans_up_concurrency_state(tmp_path):
    db_file = str(tmp_path / "conc_remove.db")
    mgr = SessionManager(db_path=db_file)

    mgr.create_session(session_id="sess-remove-lock")
    assert "sess-remove-lock" in mgr._session_locks

    mgr.remove_session("sess-remove-lock")
    assert "sess-remove-lock" not in mgr._session_locks
    assert "sess-remove-lock" not in mgr._active_agents
    mgr.close()


def test_recreating_removed_session_fresh_lock(tmp_path):
    db_file = str(tmp_path / "conc_recreate.db")
    mgr = SessionManager(db_path=db_file)

    lock1 = mgr.get_session_lock("sess-recreate")
    mgr.remove_session("sess-recreate")

    lock2 = mgr.get_session_lock("sess-recreate")
    assert lock1 is not lock2
    mgr.close()


@pytest.mark.asyncio
async def test_phase45_formatting_regression_under_concurrency(tmp_path, mock_groq):
    db_file = str(tmp_path / "conc_fmt.db")
    mgr = SessionManager(db_path=db_file)

    set_mock_response(mock_groq, "## Title\n- Bullet 1\n**Bold text**")

    t1 = asyncio.create_task(mgr.process_intent("sess-1", "Prompt 1"))
    t2 = asyncio.create_task(mgr.process_intent("sess-2", "Prompt 2"))

    res1, res2 = await asyncio.gather(t1, t2)

    for disp, _, _ in (res1, res2):
        assert "##" not in disp
        assert "**" not in disp

    mgr.close()


def test_session_store_persistence_tests_green(tmp_path):
    db_file = str(tmp_path / "conc_store_green.db")
    store = SessionStore(db_file)

    store.create_session(session_id="green-sess")
    store.add_message("green-sess", "user", "Hello")
    msgs = store.get_messages("green-sess")
    assert len(msgs) == 1
    store.close()
