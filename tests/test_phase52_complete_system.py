import asyncio
import json
import sqlite3
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.main import app
from backend.agent.session_manager import SessionManager
from backend.storage.session_store import SessionStore
from backend.tools.base import BaseTool, PermissionLevel


class SystemValidationTool(BaseTool):
    def __init__(self, execution_counter):
        self.execution_counter = execution_counter

    @property
    def name(self) -> str:
        return "system_val_action"

    @property
    def description(self) -> str:
        return "System validation tool"

    @property
    def schema(self) -> dict:
        return {"type": "object", "properties": {}}

    @property
    def permission(self) -> PermissionLevel:
        return PermissionLevel.CONFIRM_REQUIRED

    def execute(self, **kwargs) -> str:
        self.execution_counter["count"] += 1
        return "System validation tool executed successfully"


@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "complete_system.db")


@pytest.fixture
def client(temp_db_path):
    store = SessionStore(temp_db_path)
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

        def make_completion(content_text="Complete system validation response."):
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
            resp_text = getattr(client_instance, "_custom_response", "Complete system validation response.")
            return make_completion(resp_text)

        client_instance.chat.completions.create = AsyncMock(side_effect=async_create)
        yield client_instance


def set_mock_response(client_instance, text: str):
    client_instance._custom_response = text


# ==========================================
# 1. COMPLETE SESSION LIFECYCLE
# ==========================================

def test_1_complete_session_lifecycle(client, mock_groq):
    # CREATE
    create_res = client.post("/sessions", json={"title": "Lifecycle Session"}).json()
    sid = create_res["id"]
    assert create_res["title"] == "Lifecycle Session"

    # GET
    get_res = client.get(f"/sessions/{sid}").json()
    assert get_res["id"] == sid

    # CHAT
    set_mock_response(mock_groq, "Lifecycle response 1")
    chat_res = client.post("/chat", json={"message": "Lifecycle turn 1", "session_id": sid}).json()
    assert chat_res["response"] == "Lifecycle response 1"

    # PERSIST
    msgs = main_module.session_manager.session_store.get_messages(sid)
    assert len(msgs) == 2

    # LIST
    list_res = client.get("/sessions").json()
    assert any(s["id"] == sid for s in list_res)

    # RECONSTRUCT (clear RAM cache)
    main_module.session_manager.clear_session_cache(sid)

    # CHAT AGAIN
    set_mock_response(mock_groq, "Lifecycle response 2")
    chat_res2 = client.post("/chat", json={"message": "Lifecycle turn 2", "session_id": sid}).json()
    assert chat_res2["response"] == "Lifecycle response 2"

    # UPDATE TITLE
    patch_res = client.patch(f"/sessions/{sid}", json={"title": "Renamed Lifecycle"}).json()
    assert patch_res["title"] == "Renamed Lifecycle"

    # DELETE
    del_res = client.delete(f"/sessions/{sid}").json()
    assert del_res["status"] == "deleted"

    # VERIFY DELETION
    assert client.get(f"/sessions/{sid}").status_code == 404
    assert len(main_module.session_manager.session_store.get_messages(sid)) == 0


# ==========================================
# 2. MULTI-SESSION ISOLATION
# ==========================================

def test_2_multi_session_isolation(client, mock_groq):
    s_a = client.post("/sessions", json={"title": "Session A"}).json()["id"]
    s_b = client.post("/sessions", json={"title": "Session B"}).json()["id"]
    s_c = client.post("/sessions", json={"title": "Session C"}).json()["id"]

    # Turns in A
    set_mock_response(mock_groq, "Resp A")
    client.post("/chat", json={"message": "Turn A1", "session_id": s_a})
    client.post("/chat", json={"message": "Turn A2", "session_id": s_a})

    # Turns in B
    set_mock_response(mock_groq, "Resp B")
    client.post("/chat", json={"message": "Turn B1", "session_id": s_b})

    # Turns in C
    set_mock_response(mock_groq, "Resp C")
    client.post("/chat", json={"message": "Turn C1", "session_id": s_c})

    agent_a = main_module.session_manager.get_session(s_a)
    agent_b = main_module.session_manager.get_session(s_b)
    agent_c = main_module.session_manager.get_session(s_c)

    # Distinct AgentCore instances
    assert agent_a is not agent_b
    assert agent_b is not agent_c

    # History isolation
    assert [m["content"] for m in agent_a.conversation_history if m["role"] == "user"] == ["Turn A1", "Turn A2"]
    assert [m["content"] for m in agent_b.conversation_history if m["role"] == "user"] == ["Turn B1"]
    assert [m["content"] for m in agent_c.conversation_history if m["role"] == "user"] == ["Turn C1"]

    # Delete A -> B and C unaffected
    client.delete(f"/sessions/{s_a}")
    assert client.get(f"/sessions/{s_b}").status_code == 200
    assert client.get(f"/sessions/{s_c}").status_code == 200


# ==========================================
# 3. REAL CONCURRENCY SCENARIO
# ==========================================

def test_3_real_concurrency_scenario(client, mock_groq):
    s_a = client.post("/sessions", json={"title": "Session A"}).json()["id"]
    s_b = client.post("/sessions", json={"title": "Session B"}).json()["id"]
    s_c = client.post("/sessions", json={"title": "Session C"}).json()["id"]

    a_order = []

    async def custom_create(**kwargs):
        msgs = kwargs.get("messages", [])
        last_input = msgs[-1].get("content", "")
        if "A" in last_input:
            a_order.append(f"start:{last_input}")
            await asyncio.sleep(0.02)
            a_order.append(f"finish:{last_input}")

        comp = MagicMock()
        choice = MagicMock()
        msg = MagicMock()
        msg.content = f"Done {last_input}"
        msg.tool_calls = None
        choice.message = msg
        comp.choices = [choice]
        comp.usage = MagicMock(prompt_tokens=10, total_tokens=30)
        return comp

    mock_groq.chat.completions.create.side_effect = custom_create

    async def run_scenario():
        # A1, A2, B1, B2, C1 concurrent tasks
        t_a1 = asyncio.create_task(main_module.session_manager.process_intent(s_a, "Turn A1"))
        t_a2 = asyncio.create_task(main_module.session_manager.process_intent(s_a, "Turn A2"))
        t_b1 = asyncio.create_task(main_module.session_manager.process_intent(s_b, "Turn B1"))
        t_b2 = asyncio.create_task(main_module.session_manager.process_intent(s_b, "Turn B2"))
        t_c1 = asyncio.create_task(main_module.session_manager.process_intent(s_c, "Turn C1"))

        await asyncio.gather(t_a1, t_a2, t_b1, t_b2, t_c1)

    asyncio.run(run_scenario())

    # A requests serialized cleanly with A
    assert a_order == ["start:Turn A1", "finish:Turn A1", "start:Turn A2", "finish:Turn A2"]

    # History intact
    agent_a = main_module.session_manager.get_session(s_a)
    agent_b = main_module.session_manager.get_session(s_b)
    agent_c = main_module.session_manager.get_session(s_c)

    assert len([m for m in agent_a.conversation_history if m["role"] == "user"]) == 2
    assert len([m for m in agent_b.conversation_history if m["role"] == "user"]) == 2
    assert len([m for m in agent_c.conversation_history if m["role"] == "user"]) == 1


# ==========================================
# 4. PERSISTENCE / RESTART TEST
# ==========================================

def test_4_persistence_restart_test(temp_db_path, mock_groq):
    # Step 1: Create SessionManager and store history
    sm1 = SessionManager(db_path=temp_db_path)
    agent1 = sm1.create_session(title="Restart Test", session_id="restart_sid_1")

    set_mock_response(mock_groq, "Restart response 1")
    asyncio.run(sm1.process_intent("restart_sid_1", "Restart turn 1"))

    # Step 2: Simulate complete application shutdown
    sm1.close()

    # Step 3: Recreate SessionManager with same SQLite DB path
    sm2 = SessionManager(db_path=temp_db_path)
    agent2 = sm2.get_session("restart_sid_1")

    assert agent2 is not None
    user_msgs = [m for m in agent2.conversation_history if m["role"] == "user"]
    assert user_msgs[0]["content"] == "Restart turn 1"

    # Continue conversation
    set_mock_response(mock_groq, "Restart response 2")
    asyncio.run(sm2.process_intent("restart_sid_1", "Restart turn 2"))

    assert len(agent2.conversation_history) == 5  # System + 2*(user+assistant)
    sm2.close()


# ==========================================
# 5. REST -> SESSIONMANAGER -> AGENTCORE -> SESSIONSTORE INTEGRATION
# ==========================================

def test_5_full_rest_path_integration(client, mock_groq):
    # POST /sessions
    s_data = client.post("/sessions", json={"title": "REST Path Session"}).json()
    sid = s_data["id"]

    # GET /sessions
    assert any(s["id"] == sid for s in client.get("/sessions").json())

    # GET /sessions/{id}
    assert client.get(f"/sessions/{sid}").json()["id"] == sid

    # POST /chat
    set_mock_response(mock_groq, "REST Chat Response")
    chat_res = client.post("/chat", json={"message": "REST Chat Prompt", "session_id": sid})
    assert chat_res.status_code == 200

    # POST /voice
    set_mock_response(mock_groq, "REST Voice Response")
    voice_res = client.post("/voice", json={"message": "REST Voice Prompt", "session_id": sid})
    assert voice_res.status_code == 200

    # PATCH /sessions/{id}
    patch_res = client.patch(f"/sessions/{sid}", json={"title": "Updated Title"})
    assert patch_res.json()["title"] == "Updated Title"

    # DELETE /sessions/{id}
    del_res = client.delete(f"/sessions/{sid}")
    assert del_res.status_code == 200


# ==========================================
# 6. UNKNOWN SESSION SAFETY
# ==========================================

def test_6_unknown_session_safety(client):
    bad_sid = "non_existent_session_9999"

    assert client.post("/chat", json={"message": "Hi", "session_id": bad_sid}).status_code == 404
    assert client.post("/voice", json={"message": "Hi", "session_id": bad_sid}).status_code == 404
    assert client.post("/confirm", json={"confirmation_id": "c1", "approved": True, "session_id": bad_sid}).status_code == 404
    assert client.get(f"/sessions/{bad_sid}").status_code == 404
    assert client.patch(f"/sessions/{bad_sid}", json={"title": "Title"}).status_code == 404
    assert client.delete(f"/sessions/{bad_sid}").status_code == 404


# ==========================================
# 7. CONFIRMATION END-TO-END SAFETY
# ==========================================

def test_7_confirmation_end_to_end_safety(client, mock_groq):
    s_a = client.post("/sessions", json={"title": "Conf Session A"}).json()["id"]
    s_b = client.post("/sessions", json={"title": "Conf Session B"}).json()["id"]

    agent_a = main_module.session_manager.get_session(s_a)
    agent_b = main_module.session_manager.get_session(s_b)

    counter = {"count": 0}
    agent_a.tool_registry.register(SystemValidationTool(counter))
    agent_b.tool_registry.register(SystemValidationTool(counter))

    mock_tc = MagicMock()
    mock_tc.id = "tc_sys_val_1"
    mock_tc.function.name = "system_val_action"
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

    # Generate confirmation in Session A
    res_a = client.post("/chat", json={"message": "Trigger tool in A", "session_id": s_a}).json()
    conf_id = res_a["action_required"]["confirmation_id"]

    # 3. Attempt from Session B -> Must NOT execute
    res_b = client.post("/confirm", json={"confirmation_id": conf_id, "approved": True, "session_id": s_b}).json()
    assert "Confirmation failed" in res_b["response"]
    assert counter["count"] == 0

    # 6. Forged confirmation ID -> Must NOT execute
    res_forged = client.post("/confirm", json={"confirmation_id": "forged_id_999", "approved": True, "session_id": s_a}).json()
    assert "Confirmation failed" in res_forged["response"]
    assert counter["count"] == 0

    # 1. Approve from Session A -> Should execute exactly once
    res_approve = client.post("/confirm", json={"confirmation_id": conf_id, "approved": True, "session_id": s_a}).json()
    assert res_approve["response"] is not None
    assert counter["count"] == 1

    # 2. Replay from Session A -> Must NOT execute again
    res_replay = client.post("/confirm", json={"confirmation_id": conf_id, "approved": True, "session_id": s_a}).json()
    assert "Confirmation failed" in res_replay["response"]
    assert counter["count"] == 1


# ==========================================
# 8. CONTEXT TRIMMING VS PERSISTENCE
# ==========================================

def test_8_context_trimming_vs_persistence(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Trimming Vs Persist"}).json()["id"]

    long_prompt = "Heavy prompt text for token budget test. " * 40

    for i in range(8):
        set_mock_response(mock_groq, f"Answer {i} " + ("word " * 40))
        client.post("/chat", json={"message": f"Question {i} {long_prompt}", "session_id": sid})

    agent = main_module.session_manager.get_session(sid)

    # 1. RAM history length vs SQLite total history length
    assert len(agent.conversation_history) < 17

    db_msgs = main_module.session_manager.session_store.get_messages(sid)
    user_db_msgs = [m for m in db_msgs if m["role"] == "user"]
    assert len(user_db_msgs) == 8

    # 2. Destroy agent in memory & reconstruct
    main_module.session_manager.clear_session_cache(sid)
    reconstructed_agent = main_module.session_manager.get_session(sid)

    # 3. System prompt appears exactly once
    system_msgs = [m for m in reconstructed_agent.conversation_history if m["role"] == "system"]
    assert len(system_msgs) == 1


# ==========================================
# 9. FAILURE RECOVERY
# ==========================================

def test_9_failure_recovery(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Failure Recovery"}).json()["id"]

    # Exception during Groq call
    mock_groq.chat.completions.create.side_effect = Exception("API connection failure")
    try:
        client.post("/chat", json={"message": "Will fail", "session_id": sid})
    except Exception:
        pass

    # Reset mock to normal async completion function
    async def async_create(**kwargs):
        resp_text = getattr(mock_groq, "_custom_response", "Recovered response")
        mock_comp = MagicMock()
        mock_choice = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = resp_text
        mock_msg.tool_calls = None
        mock_choice.message = mock_msg
        mock_comp.choices = [mock_choice]
        mock_comp.usage = MagicMock(prompt_tokens=10, total_tokens=30)
        return mock_comp

    mock_groq.chat.completions.create.side_effect = async_create
    set_mock_response(mock_groq, "Recovered response")

    res = client.post("/chat", json={"message": "Recover prompt", "session_id": sid})
    assert res.status_code == 200
    assert res.json()["response"] == "Recovered response"


# ==========================================
# 10. PHASE 4.5 REGRESSION
# ==========================================

def test_10_phase45_regression(client, mock_groq):
    sid = client.post("/sessions", json={"title": "Formatting Regression"}).json()["id"]

    set_mock_response(mock_groq, "# Heading\n- Bullet 1\n| Header | Header |\n| --- | --- |")
    res = client.post("/chat", json={"message": "Formatting test", "session_id": sid}).json()

    assert "# Heading" not in res["response"]
    assert "---" not in res["response"]
    assert "|" not in res["spoken_response"]
    assert res["spoken_response"] is not None


# ==========================================
# 11. ARCHITECTURE INTEGRITY
# ==========================================

def test_11_architecture_integrity():
    with open("/Users/novus/Documents/P.I.X.I.E/backend/main.py", "r") as f:
        content = f.read()

    assert "AgentCore(" not in content
    assert "SessionStore(" not in content


# ==========================================
# 12. DATABASE INTEGRITY
# ==========================================

def test_12_database_integrity(temp_db_path):
    store = SessionStore(temp_db_path)
    conn = store._get_connection()
    cursor = conn.cursor()

    # Foreign keys enabled check in SessionStore connections
    cursor.execute("PRAGMA foreign_keys;")
    fk_status = cursor.fetchone()[0]
    assert fk_status == 1

    # CASCADE deletion check
    store.create_session(title="DB Test", session_id="s_db_1")
    store.add_message("s_db_1", "user", "Hello")

    assert len(store.get_messages("s_db_1")) == 1

    store.delete_session("s_db_1")
    assert len(store.get_messages("s_db_1")) == 0

    store.close()
