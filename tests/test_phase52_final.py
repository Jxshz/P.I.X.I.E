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


class FinalAuditTool(BaseTool):
    def __init__(self, execution_counter):
        self.execution_counter = execution_counter

    @property
    def name(self) -> str:
        return "audit_action"

    @property
    def description(self) -> str:
        return "Audit test tool"

    @property
    def schema(self) -> dict:
        return {"type": "object", "properties": {}}

    @property
    def permission(self) -> PermissionLevel:
        return PermissionLevel.CONFIRM_REQUIRED

    def execute(self, **kwargs) -> str:
        self.execution_counter["count"] += 1
        return "Audit tool executed successfully"


@pytest.fixture
def client(tmp_path):
    db_file = str(tmp_path / "test_phase52_final.db")
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

        def make_completion(content_text="Final audit mocked response."):
            mock_comp = MagicMock()
            mock_choice = MagicMock()
            mock_msg = MagicMock()
            mock_msg.content = content_text
            mock_msg.tool_calls = None
            mock_choice.message = mock_msg
            mock_comp.choices = [mock_choice]
            mock_comp.usage = MagicMock(prompt_tokens=12, total_tokens=36)
            return mock_comp

        async def async_create(**kwargs):
            resp_text = getattr(client_instance, "_custom_response", "Final audit mocked response.")
            return make_completion(resp_text)

        client_instance.chat.completions.create = AsyncMock(side_effect=async_create)
        yield client_instance


def set_mock_response(client_instance, text: str):
    client_instance._custom_response = text


# ==========================================
# A. SESSION ISOLATION TESTS
# ==========================================

def test_multi_session_history_isolation(client, mock_groq):
    s1 = client.post("/sessions", json={"title": "Session Alpha"}).json()["id"]
    s2 = client.post("/sessions", json={"title": "Session Beta"}).json()["id"]

    set_mock_response(mock_groq, "Alpha turn 1 response")
    client.post("/chat", json={"message": "Alpha turn 1", "session_id": s1})

    set_mock_response(mock_groq, "Beta turn 1 response")
    client.post("/chat", json={"message": "Beta turn 1", "session_id": s2})

    agent1 = main_module.session_manager.get_session(s1)
    agent2 = main_module.session_manager.get_session(s2)

    user_msgs1 = [m["content"] for m in agent1.conversation_history if m["role"] == "user"]
    user_msgs2 = [m["content"] for m in agent2.conversation_history if m["role"] == "user"]

    assert user_msgs1 == ["Alpha turn 1"]
    assert user_msgs2 == ["Beta turn 1"]


def test_session_scoped_confirmation_isolation(client, mock_groq):
    s1 = client.post("/sessions", json={"title": "Session A"}).json()["id"]
    s2 = client.post("/sessions", json={"title": "Session B"}).json()["id"]

    agent1 = main_module.session_manager.get_session(s1)
    counter = {"count": 0}
    agent1.tool_registry.register(FinalAuditTool(counter))

    mock_tc = MagicMock()
    mock_tc.id = "tc_iso_1"
    mock_tc.function.name = "audit_action"
    mock_tc.function.arguments = "{}"

    tool_comp = MagicMock()
    tool_choice = MagicMock()
    tool_msg = MagicMock()
    tool_msg.content = None
    tool_msg.tool_calls = [mock_tc]
    tool_choice.message = tool_msg
    tool_comp.choices = [tool_choice]
    tool_comp.usage = MagicMock(prompt_tokens=15, total_tokens=40)

    mock_groq.chat.completions.create.side_effect = AsyncMock(return_value=tool_comp)

    res1 = client.post("/chat", json={"message": "Trigger audit tool", "session_id": s1}).json()
    conf_id = res1["action_required"]["confirmation_id"]

    # Try resolving via Session B -> Must be rejected
    res2 = client.post("/confirm", json={"confirmation_id": conf_id, "approved": True, "session_id": s2})
    assert res2.status_code == 200
    assert "Confirmation failed: Unknown, expired, or already used" in res2.json()["response"]
    assert counter["count"] == 0


def test_session_metadata_association(client):
    s = client.post("/sessions", json={"title": "Meta Session"}).json()
    sid = s["id"]

    get_res = client.get(f"/sessions/{sid}")
    assert get_res.status_code == 200
    meta = get_res.json()
    assert meta["id"] == sid
    assert meta["title"] == "Meta Session"
    assert meta["created_at"] > 0
    assert meta["updated_at"] >= meta["created_at"]


def test_reconstructed_session_restoration(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Reconstruct Test"}).json()["id"]

    set_mock_response(mock_groq, "Response before eviction")
    client.post("/chat", json={"message": "Prompt before eviction", "session_id": sid})

    # Clear RAM cache
    main_module.session_manager.clear_session_cache(sid)

    # Reload session
    agent = main_module.session_manager.get_session(sid)
    assert agent is not None
    assert any(m["content"] == "Prompt before eviction" for m in agent.conversation_history)


# ==========================================
# B. CONCURRENCY TESTS
# ==========================================

def test_concurrent_same_session_serialization(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Serial Lock Test"}).json()["id"]
    order = []

    async def custom_create(**kwargs):
        msgs = kwargs.get("messages", [])
        prompt = msgs[-1].get("content", "")
        order.append(f"start:{prompt}")
        await asyncio.sleep(0.02)
        order.append(f"finish:{prompt}")
        comp = MagicMock()
        choice = MagicMock()
        msg = MagicMock()
        msg.content = f"Resp {prompt}"
        msg.tool_calls = None
        choice.message = msg
        comp.choices = [choice]
        comp.usage = MagicMock(prompt_tokens=10, total_tokens=30)
        return comp

    mock_groq.chat.completions.create.side_effect = custom_create

    async def run_concurrent_turns():
        t1 = asyncio.create_task(main_module.session_manager.process_intent(sid, "Turn 1"))
        t2 = asyncio.create_task(main_module.session_manager.process_intent(sid, "Turn 2"))
        await asyncio.gather(t1, t2)

    asyncio.run(run_concurrent_turns())

    assert order == ["start:Turn 1", "finish:Turn 1", "start:Turn 2", "finish:Turn 2"]


def test_concurrent_multi_session_concurrency(client, mock_groq):
    sids = [client.post("/sessions", json={"title": f"S{i}"}).json()["id"] for i in range(5)]
    finished_count = {"count": 0}

    async def custom_create(**kwargs):
        await asyncio.sleep(0.01)
        finished_count["count"] += 1
        comp = MagicMock()
        choice = MagicMock()
        msg = MagicMock()
        msg.content = "Multi-session parallel resp"
        msg.tool_calls = None
        choice.message = msg
        comp.choices = [choice]
        comp.usage = MagicMock(prompt_tokens=10, total_tokens=30)
        return comp

    mock_groq.chat.completions.create.side_effect = custom_create

    async def run_parallel_sessions():
        tasks = [
            asyncio.create_task(main_module.session_manager.process_intent(sid, f"Prompt {sid}"))
            for sid in sids
        ]
        await asyncio.gather(*tasks)

    asyncio.run(run_parallel_sessions())
    assert finished_count["count"] == 5


def test_session_removal_purges_locks_and_agents(client):
    sid = client.post("/sessions", json={"title": "Purge Test"}).json()["id"]
    main_module.session_manager.get_session(sid)

    assert sid in main_module.session_manager._active_agents
    assert sid in main_module.session_manager._session_locks

    client.delete(f"/sessions/{sid}")
    assert sid not in main_module.session_manager._active_agents
    assert sid not in main_module.session_manager._session_locks


# ==========================================
# C. REST API TESTS
# ==========================================

def test_sessions_rest_crud_complete(client):
    # Create
    created = client.post("/sessions", json={"title": "CRUD Session"}).json()
    sid = created["id"]
    assert created["title"] == "CRUD Session"

    # List
    list_res = client.get("/sessions").json()
    assert any(s["id"] == sid for s in list_res)

    # Get
    get_res = client.get(f"/sessions/{sid}").json()
    assert get_res["title"] == "CRUD Session"

    # Patch
    patch_res = client.patch(f"/sessions/{sid}", json={"title": "CRUD Session Renamed"}).json()
    assert patch_res["title"] == "CRUD Session Renamed"

    # Delete
    del_res = client.delete(f"/sessions/{sid}").json()
    assert del_res["status"] == "deleted"

    # Verify 404 after delete
    assert client.get(f"/sessions/{sid}").status_code == 404


def test_sessions_api_http_statuses(client):
    # Duplicate creation -> 400
    s1 = client.post("/sessions", json={"title": "Dup Test", "session_id": "dup_sid_123"}).json()
    dup_res = client.post("/sessions", json={"title": "Dup Test 2", "session_id": "dup_sid_123"})
    assert dup_res.status_code == 400

    # Non-existent session -> 404
    assert client.get("/sessions/non_existent_999").status_code == 404
    assert client.patch("/sessions/non_existent_999", json={"title": "New"}).status_code == 404
    assert client.delete("/sessions/non_existent_999").status_code == 404


# ==========================================
# D. ROUTING TESTS
# ==========================================

def test_routing_exclusive_to_session_manager(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Route Exclusive"}).json()["id"]

    with patch.object(main_module.session_manager, "process_intent", wraps=main_module.session_manager.process_intent) as spy:
        client.post("/chat", json={"message": "Chat msg", "session_id": sid})
        spy.assert_called_once_with(sid, "Chat msg")


def test_routing_explicit_unknown_session_returns_404(client):
    assert client.post("/chat", json={"message": "Hi", "session_id": "unknown_sid"}).status_code == 404
    assert client.post("/voice", json={"message": "Hi", "session_id": "unknown_sid"}).status_code == 404
    assert client.post("/confirm", json={"confirmation_id": "c1", "approved": True, "session_id": "unknown_sid"}).status_code == 404
    assert client.post("/api/clear", json={"session_id": "unknown_sid"}).status_code == 404


def test_routing_unspecified_session_defaults_to_default(client, mock_groq):
    set_mock_response(mock_groq, "Default session response")
    res = client.post("/chat", json={"message": "No session_id provided"})
    assert res.status_code == 200
    assert res.json()["response"] == "Default session response"

    default_agent = main_module.session_manager.get_session("default")
    assert default_agent is not None
    assert any(m["content"] == "No session_id provided" for m in default_agent.conversation_history)


def test_no_direct_agentcore_or_sessionstore_in_main():
    with open("/Users/novus/Documents/P.I.X.I.E/backend/main.py", "r") as f:
        content = f.read()

    assert "AgentCore(" not in content
    assert "SessionStore(" not in content


# ==========================================
# E. PERSISTENCE TESTS
# ==========================================

def test_conversation_survives_agentcore_reconstruction(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Survive Test"}).json()["id"]

    set_mock_response(mock_groq, "Turn 1 answer")
    client.post("/chat", json={"message": "Turn 1 prompt", "session_id": sid})

    # Evict agent from RAM
    main_module.session_manager.clear_session_cache(sid)

    # Next call reloads history from SQLite
    set_mock_response(mock_groq, "Turn 2 answer")
    res = client.post("/chat", json={"message": "Turn 2 prompt", "session_id": sid})
    assert res.status_code == 200

    agent = main_module.session_manager.get_session(sid)
    history_prompts = [m["content"] for m in agent.conversation_history if m["role"] == "user"]
    assert history_prompts == ["Turn 1 prompt", "Turn 2 prompt"]


def test_context_trimming_preserves_db_history(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Trim Test"}).json()["id"]

    long_text = "This is a long prompt containing lots of text. " * 40

    for i in range(8):
        set_mock_response(mock_groq, f"Resp {i} " + ("word " * 40))
        client.post("/chat", json={"message": f"Prompt {i} {long_text}", "session_id": sid})

    agent = main_module.session_manager.get_session(sid)

    # RAM history has been trimmed to stay under max_context_tokens
    assert len(agent.conversation_history) < 17

    # Database history retains all 8 user turns
    db_msgs = main_module.session_manager.session_store.get_messages(sid)
    user_db_msgs = [m for m in db_msgs if m["role"] == "user"]
    assert len(user_db_msgs) == 8


# ==========================================
# F. CONFIRMATION SAFETY TESTS
# ==========================================

def test_confirmation_exact_once_execution(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Exact Once Final"}).json()["id"]
    agent = main_module.session_manager.get_session(sid)

    counter = {"count": 0}
    agent.tool_registry.register(FinalAuditTool(counter))

    mock_tc = MagicMock()
    mock_tc.id = "tc_final_1"
    mock_tc.function.name = "audit_action"
    mock_tc.function.arguments = "{}"

    tool_comp = MagicMock()
    tool_choice = MagicMock()
    tool_msg = MagicMock()
    tool_msg.content = None
    tool_msg.tool_calls = [mock_tc]
    tool_choice.message = tool_msg
    tool_comp.choices = [tool_choice]
    tool_comp.usage = MagicMock(prompt_tokens=15, total_tokens=40)

    mock_groq.chat.completions.create.side_effect = AsyncMock(return_value=tool_comp)

    res = client.post("/chat", json={"message": "Run audit tool", "session_id": sid}).json()
    conf_id = res["action_required"]["confirmation_id"]

    # Execution 1
    c1 = client.post("/confirm", json={"confirmation_id": conf_id, "approved": True, "session_id": sid})
    assert c1.status_code == 200
    assert counter["count"] == 1

    # Replay -> Rejected
    c2 = client.post("/confirm", json={"confirmation_id": conf_id, "approved": True, "session_id": sid})
    assert c2.status_code == 200
    assert counter["count"] == 1
    assert "Confirmation failed: Unknown, expired, or already used" in c2.json()["response"]


def test_clear_context_invalidates_session_confirmations(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Clear Invalidate"}).json()["id"]
    agent = main_module.session_manager.get_session(sid)

    counter = {"count": 0}
    agent.tool_registry.register(FinalAuditTool(counter))

    mock_tc = MagicMock()
    mock_tc.id = "tc_clear_1"
    mock_tc.function.name = "audit_action"
    mock_tc.function.arguments = "{}"

    tool_comp = MagicMock()
    tool_choice = MagicMock()
    tool_msg = MagicMock()
    tool_msg.content = None
    tool_msg.tool_calls = [mock_tc]
    tool_choice.message = tool_msg
    tool_comp.choices = [tool_choice]
    tool_comp.usage = MagicMock(prompt_tokens=15, total_tokens=40)

    mock_groq.chat.completions.create.side_effect = AsyncMock(return_value=tool_comp)

    res = client.post("/chat", json={"message": "Run audit tool", "session_id": sid}).json()
    conf_id = res["action_required"]["confirmation_id"]

    # Clear context for sid
    client.post("/api/clear", json={"session_id": sid})

    # Confirmation resolution after clear context must be rejected
    c = client.post("/confirm", json={"confirmation_id": conf_id, "approved": True, "session_id": sid})
    assert c.status_code == 200
    assert "Confirmation failed: Unknown, expired, or already used" in c.json()["response"]
    assert counter["count"] == 0


# ==========================================
# G. PHASE 4.5 REGRESSION TESTS
# ==========================================

def test_phase45_display_response_formatting(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Formatting Test"}).json()["id"]
    set_mock_response(mock_groq, "# Title\n**bold text**\n- item 1\n---")

    res = client.post("/chat", json={"message": "Formatting prompt", "session_id": sid})
    assert res.status_code == 200

    display_text = res.json()["response"]
    assert "# Title" not in display_text
    assert "**bold text**" not in display_text
    assert "---" not in display_text


def test_phase45_spoken_response_formatting(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Spoken Test"}).json()["id"]
    set_mock_response(mock_groq, "Here is your requested answer.")

    res = client.post("/chat", json={"message": "Spoken prompt", "session_id": sid})
    assert res.status_code == 200

    spoken_text = res.json()["spoken_response"]
    assert spoken_text is not None
    assert "Here is your requested answer" in spoken_text


def test_phase45_token_governor_telemetry(client):
    status_res = client.get("/status")
    assert status_res.status_code == 200
    data = status_res.json()
    assert data["status"] == "online"
    assert "requests_minute" in data
    assert "tokens_minute" in data
    assert "rpm_limit" in data
    assert "tpm_limit" in data
