import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.main import app
from backend.agent.session_manager import SessionManager
from backend.storage.session_store import SessionStore
from backend.tools.base import BaseTool, PermissionLevel


class RouteTestTool(BaseTool):
    def __init__(self, execution_counter):
        self.execution_counter = execution_counter

    @property
    def name(self) -> str:
        return "route_action"

    @property
    def description(self) -> str:
        return "Route tool"

    @property
    def schema(self) -> dict:
        return {"type": "object", "properties": {}}

    @property
    def permission(self) -> PermissionLevel:
        return PermissionLevel.CONFIRM_REQUIRED

    def execute(self, **kwargs) -> str:
        self.execution_counter["count"] += 1
        return "Executed"


@pytest.fixture
def client(tmp_path):
    db_file = str(tmp_path / "test_routing.db")
    store = SessionStore(db_file)
    test_mgr = SessionManager(session_store=store)

    with patch("backend.main.session_manager", test_mgr):
        with TestClient(app) as test_client:
            yield test_client

    test_mgr.close()


@pytest.fixture
def mock_groq():
    with patch("backend.agent.core.AsyncGroq") as mock_groq_cls:
        client_instance = MagicMock()
        mock_groq_cls.return_value = client_instance

        def make_completion(content_text="Mocked completion response."):
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
            resp_text = getattr(client_instance, "_custom_response", "Mocked completion response.")
            return make_completion(resp_text)

        client_instance.chat.completions.create = AsyncMock(side_effect=async_create)
        yield client_instance


def set_mock_response(client_instance, text: str):
    client_instance._custom_response = text


def test_chat_routes_to_requested_session(client, mock_groq):
    s_res = client.post("/sessions", json={"title": "Session 1"}).json()
    sid = s_res["id"]

    set_mock_response(mock_groq, "Response in Session 1")
    chat_res = client.post("/chat", json={"message": "Hello Session 1", "session_id": sid})
    assert chat_res.status_code == 200
    data = chat_res.json()
    assert data["response"] == "Response in Session 1"


def test_session_a_and_b_have_isolated_histories(client, mock_groq):
    sid_a = client.post("/sessions", json={"title": "A"}).json()["id"]
    sid_b = client.post("/sessions", json={"title": "B"}).json()["id"]

    set_mock_response(mock_groq, "Response A")
    client.post("/chat", json={"message": "Prompt A", "session_id": sid_a})

    set_mock_response(mock_groq, "Response B")
    client.post("/chat", json={"message": "Prompt B", "session_id": sid_b})

    agent_a = main_module.session_manager.get_session(sid_a)
    agent_b = main_module.session_manager.get_session(sid_b)

    assert [m["content"] for m in agent_a.conversation_history if m["role"] == "user"] == ["Prompt A"]
    assert [m["content"] for m in agent_b.conversation_history if m["role"] == "user"] == ["Prompt B"]


def test_unknown_session_returns_404(client):
    response = client.post("/chat", json={"message": "Hello", "session_id": "non_existent_sid_999"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


def test_existing_session_reuses_same_agentcore(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Reuse Test"}).json()["id"]

    client.post("/chat", json={"message": "Turn 1", "session_id": sid})
    agent1 = main_module.session_manager.get_session(sid)

    client.post("/chat", json={"message": "Turn 2", "session_id": sid})
    agent2 = main_module.session_manager.get_session(sid)

    assert agent1 is agent2


def test_chat_uses_session_manager(client, mock_groq):
    sid = client.post("/sessions", json={"title": "SM Call Test"}).json()["id"]

    with patch.object(main_module.session_manager, "process_intent", wraps=main_module.session_manager.process_intent) as spy:
        client.post("/chat", json={"message": "Testing SM route", "session_id": sid})
        spy.assert_called_once_with(sid, "Testing SM route")


def test_concurrent_same_session_requests_are_serialized(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Concurrent Serial Test"}).json()["id"]

    execution_order = []

    async def delayed_create(**kwargs):
        messages = kwargs.get("messages", [])
        last_input = messages[-1].get("content", "")
        execution_order.append(f"start:{last_input}")
        await asyncio.sleep(0.02)
        execution_order.append(f"finish:{last_input}")
        mock_comp = MagicMock()
        mock_choice = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = f"Done {last_input}"
        mock_msg.tool_calls = None
        mock_choice.message = mock_msg
        mock_comp.choices = [mock_choice]
        mock_comp.usage = MagicMock(prompt_tokens=10, total_tokens=30)
        return mock_comp

    mock_groq.chat.completions.create.side_effect = delayed_create

    # Process turns under session lock via SessionManager
    async def run_turns():
        t1 = asyncio.create_task(main_module.session_manager.process_intent(sid, "Input 1"))
        t2 = asyncio.create_task(main_module.session_manager.process_intent(sid, "Input 2"))
        await asyncio.gather(t1, t2)

    asyncio.run(run_turns())

    assert execution_order == [
        "start:Input 1",
        "finish:Input 1",
        "start:Input 2",
        "finish:Input 2",
    ]


def test_concurrent_different_sessions_run_concurrently(client, mock_groq):
    sid_a = client.post("/sessions", json={"title": "Session A"}).json()["id"]
    sid_b = client.post("/sessions", json={"title": "Session B"}).json()["id"]

    a_started = asyncio.Event()
    a_can_finish = asyncio.Event()
    b_finished = asyncio.Event()

    async def custom_create(**kwargs):
        messages = kwargs.get("messages", [])
        last_input = messages[-1].get("content", "")

        if "Task A" in last_input:
            a_started.set()
            await a_can_finish.wait()
        elif "Task B" in last_input:
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

    async def run_concurrency():
        task_a = asyncio.create_task(main_module.session_manager.process_intent(sid_a, "Task A"))
        await a_started.wait()

        task_b = asyncio.create_task(main_module.session_manager.process_intent(sid_b, "Task B"))
        await b_finished.wait()
        assert task_b.done()
        assert not task_a.done()

        a_can_finish.set()
        await task_a

    asyncio.run(run_concurrency())


def test_confirmation_routes_to_correct_session(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Conf Session"}).json()["id"]
    agent = main_module.session_manager.get_session(sid)

    counter = {"count": 0}
    agent.tool_registry.register(RouteTestTool(counter))

    mock_tc = MagicMock()
    mock_tc.id = "tc_route_1"
    mock_tc.function.name = "route_action"
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

    res1 = client.post("/chat", json={"message": "Run tool", "session_id": sid}).json()
    assert res1["action_required"] is not None
    conf_id = res1["action_required"]["confirmation_id"]

    res2 = client.post("/confirm", json={"confirmation_id": conf_id, "approved": True, "session_id": sid})
    assert res2.status_code == 200
    assert counter["count"] == 1


def test_cross_session_confirmation_rejected(client, mock_groq):
    sid_a = client.post("/sessions", json={"title": "Session A"}).json()["id"]
    sid_b = client.post("/sessions", json={"title": "Session B"}).json()["id"]

    agent_a = main_module.session_manager.get_session(sid_a)
    agent_b = main_module.session_manager.get_session(sid_b)

    counter = {"count": 0}
    agent_a.tool_registry.register(RouteTestTool(counter))
    agent_b.tool_registry.register(RouteTestTool(counter))

    mock_tc = MagicMock()
    mock_tc.id = "tc_cross_route_1"
    mock_tc.function.name = "route_action"
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

    res1 = client.post("/chat", json={"message": "Run tool in A", "session_id": sid_a}).json()
    conf_id = res1["action_required"]["confirmation_id"]

    # Attempting to resolve A's confirmation in Session B must be rejected
    res2 = client.post("/confirm", json={"confirmation_id": conf_id, "approved": True, "session_id": sid_b})
    assert res2.status_code == 200
    assert "Confirmation failed: Unknown, expired, or already used" in res2.json()["response"]
    assert counter["count"] == 0


def test_confirmation_exact_once_preserved(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Exact Once"}).json()["id"]
    agent = main_module.session_manager.get_session(sid)

    counter = {"count": 0}
    agent.tool_registry.register(RouteTestTool(counter))

    mock_tc = MagicMock()
    mock_tc.id = "tc_exact_once_1"
    mock_tc.function.name = "route_action"
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

    res1 = client.post("/chat", json={"message": "Run tool", "session_id": sid}).json()
    conf_id = res1["action_required"]["confirmation_id"]

    # First call -> Approved
    c1 = client.post("/confirm", json={"confirmation_id": conf_id, "approved": True, "session_id": sid})
    assert counter["count"] == 1

    # Second call -> Replay rejected
    c2 = client.post("/confirm", json={"confirmation_id": conf_id, "approved": True, "session_id": sid})
    assert counter["count"] == 1
    assert "Confirmation failed: Unknown, expired, or already used" in c2.json()["response"]


def test_persistent_history_available_after_routing(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Persist Route"}).json()["id"]

    set_mock_response(mock_groq, "Persisted response")
    client.post("/chat", json={"message": "Persistent question", "session_id": sid})

    # Flush RAM cache
    main_module.session_manager.clear_session_cache(sid)

    # Subsequent chat call re-loads persisted history
    set_mock_response(mock_groq, "Follow-up response")
    res = client.post("/chat", json={"message": "Follow-up question", "session_id": sid})
    assert res.status_code == 200

    agent = main_module.session_manager.get_session(sid)
    assert len(agent.conversation_history) == 5
    assert agent.conversation_history[1]["content"] == "Persistent question"


def test_phase45_display_response_contract_unchanged(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Contract Test"}).json()["id"]

    set_mock_response(mock_groq, "## Heading\n- Bullet 1\n**Bold text**")
    res = client.post("/chat", json={"message": "Give formatting", "session_id": sid})
    assert res.status_code == 200

    data = res.json()
    assert "##" not in data["response"]
    assert "**" not in data["response"]


def test_phase45_spoken_response_contract_unchanged(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Spoken Contract"}).json()["id"]

    set_mock_response(mock_groq, "Hello there! How can I help?")
    res = client.post("/chat", json={"message": "Hi", "session_id": sid})
    assert res.status_code == 200

    data = res.json()
    assert data["spoken_response"] is not None
    assert "Hello there" in data["spoken_response"]


def test_existing_non_session_endpoint_unchanged(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_no_direct_agentcore_construction_in_routes():
    with open("/Users/novus/Documents/P.I.X.I.E/backend/main.py", "r") as f:
        content = f.read()

    assert "AgentCore(" not in content
    assert "SessionStore(" not in content
