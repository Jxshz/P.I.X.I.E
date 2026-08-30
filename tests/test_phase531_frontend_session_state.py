import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.main import app
from backend.agent.session_manager import SessionManager
from backend.storage.session_store import SessionStore


@pytest.fixture
def client(tmp_path):
    db_file = str(tmp_path / "test_phase531.db")
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

        def make_completion(content_text="Mocked response for 5.3.1 test."):
            mock_comp = MagicMock()
            mock_choice = MagicMock()
            mock_msg = MagicMock()
            mock_msg.content = content_text
            mock_msg.tool_calls = None
            mock_choice.message = mock_msg
            mock_comp.choices = [mock_choice]
            mock_comp.usage = MagicMock(prompt_tokens=10, total_tokens=30)
            return mock_comp

        async def async_create(**kwargs):
            resp_text = getattr(client_instance, "_custom_response", "Mocked response for 5.3.1 test.")
            return make_completion(resp_text)

        client_instance.chat.completions.create = AsyncMock(side_effect=async_create)
        yield client_instance


def set_mock_response(client_instance, text: str):
    client_instance._custom_response = text


# 1. Test currentSessionId initialisation & Session REST API helper endpoints
def test_fetch_sessions_and_create_session(client):
    # Initial list is empty
    list_res = client.get("/sessions")
    assert list_res.status_code == 200
    assert list_res.json() == []

    # Create session helper endpoint
    created = client.post("/sessions", json={"title": "New Chat"}).json()
    assert "id" in created
    assert created["title"] == "New Chat"

    # Fetch sessions now returns created session
    sessions = client.get("/sessions?limit=50").json()
    assert len(sessions) == 1
    assert sessions[0]["id"] == created["id"]


# 2. Test existing session selection
def test_existing_session_selection(client):
    s1 = client.post("/sessions", json={"title": "Session 1"}).json()
    s2 = client.post("/sessions", json={"title": "Session 2"}).json()

    sessions = client.get("/sessions").json()
    # First returned session can be selected as active currentSessionId
    selected_id = sessions[0]["id"]
    get_res = client.get(f"/sessions/{selected_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == selected_id


# 3. Test automatic session creation when no sessions exist
def test_auto_session_creation_when_none_exist(client):
    sessions = client.get("/sessions").json()
    assert len(sessions) == 0

    # Frontend creates session via POST /sessions
    new_s = client.post("/sessions", json={"title": "New Chat"}).json()
    assert new_s["id"] is not None


# 4. Test chat request includes session_id
def test_chat_request_includes_session_id(client, mock_groq):
    s = client.post("/sessions", json={"title": "Chat Session"}).json()
    sid = s["id"]

    set_mock_response(mock_groq, "Mocked response for 5.3.1 test.")
    res = client.post("/chat", json={"message": "Hello P.I.X.I.E.", "session_id": sid})
    assert res.status_code == 200
    assert res.json()["response"] == "Mocked response for 5.3.1 test."

    agent = main_module.session_manager.get_session(sid)
    assert any(m["content"] == "Hello P.I.X.I.E." for m in agent.conversation_history)


# 5. Test voice request includes session_id
def test_voice_request_includes_session_id(client, mock_groq):
    s = client.post("/sessions", json={"title": "Voice Session"}).json()
    sid = s["id"]

    res = client.post("/voice", json={"message": "Voice command", "session_id": sid})
    assert res.status_code == 200

    agent = main_module.session_manager.get_session(sid)
    assert any(m["content"] == "Voice command" for m in agent.conversation_history)


# 6. Test confirmation request includes session_id
def test_confirmation_request_includes_session_id(client, mock_groq):
    s = client.post("/sessions", json={"title": "Confirm Session"}).json()
    sid = s["id"]

    # Trying to resolve non-existent confirmation returns safe error under sid
    res = client.post("/confirm", json={"confirmation_id": "c1", "approved": True, "session_id": sid})
    assert res.status_code == 200
    assert "Confirmation failed" in res.json()["response"]


# 7. Test fetchSessions() helper
def test_fetch_sessions_helper(client):
    client.post("/sessions", json={"title": "S1"})
    client.post("/sessions", json={"title": "S2"})

    res = client.get("/sessions?limit=50")
    assert res.status_code == 200
    assert len(res.json()) == 2


# 8. Test createSession() helper
def test_create_session_helper(client):
    res = client.post("/sessions", json={"title": "Custom Session"})
    assert res.status_code == 201
    assert res.json()["title"] == "Custom Session"


# 9. Test getSession() helper
def test_get_session_helper(client):
    created = client.post("/sessions", json={"title": "Target Session"}).json()
    sid = created["id"]

    res = client.get(f"/sessions/{sid}")
    assert res.status_code == 200
    assert res.json()["id"] == sid


# 10. Test deleteSession() helper
def test_delete_session_helper(client):
    created = client.post("/sessions", json={"title": "Delete Me"}).json()
    sid = created["id"]

    del_res = client.delete(f"/sessions/{sid}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    get_res = client.get(f"/sessions/{sid}")
    assert get_res.status_code == 404


# 11. Test invalid session handling
def test_invalid_session_handling(client):
    res = client.post("/chat", json={"message": "Test", "session_id": "invalid_sid_999"})
    assert res.status_code == 404
    assert res.json()["detail"] == "Session not found"


# 12. Test strict browser persistence boundary in frontend files (only pixie_active_session_id)
def test_no_browser_persistence_mechanisms():
    with open("/Users/novus/Documents/P.I.X.I.E/frontend/app.js", "r") as f:
        app_js_content = f.read()

    assert "pixie_active_session_id" in app_js_content
    assert "sessionStorage" not in app_js_content
    assert "indexedDB" not in app_js_content
    assert "document.cookie" not in app_js_content


# 13. Test existing Phase 4.5 behavior remains intact
def test_existing_phase45_behavior_intact(client, mock_groq):
    s = client.post("/sessions", json={"title": "Phase 4.5 Check"}).json()
    sid = s["id"]

    mock_groq._custom_response = "# Heading\n**bold**"
    res = client.post("/chat", json={"message": "Format test", "session_id": sid}).json()

    assert "# Heading" not in res["response"]
    assert "**bold**" not in res["response"]
    assert res["spoken_response"] is not None
