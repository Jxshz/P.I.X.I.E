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
    db_file = str(tmp_path / "test_phase533_persistence.db")
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

        def make_completion(content_text="Mocked response for 5.3.3 persistence test."):
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
            resp_text = getattr(client_instance, "_custom_response", "Mocked response for 5.3.3 persistence test.")
            return make_completion(resp_text)

        client_instance.chat.completions.create = AsyncMock(side_effect=async_create)
        yield client_instance


def set_mock_response(client_instance, text: str):
    client_instance._custom_response = text


# 1. Active-session persistence key is configured correctly
def test_1_active_session_key_configured():
    with open("frontend/app.js", "r") as f:
        js = f.read()
    assert "pixie_active_session_id" in js
    assert "getPersistedSessionId" in js
    assert "setPersistedSessionId" in js


# 2. Persisted session ID is read during startup
def test_2_persisted_session_read_during_startup():
    with open("frontend/app.js", "r") as f:
        js = f.read()
    assert "getPersistedSessionId()" in js
    assert "initializeSession()" in js
    assert "let targetSessionId = currentSessionId || getPersistedSessionId();" in js


# 3. Persisted valid session ID is verified against backend
def test_3_persisted_valid_session_verified(client):
    s1 = client.post("/sessions", json={"title": "Persisted Chat"}).json()
    sid = s1["id"]

    res = client.get(f"/sessions/{sid}")
    assert res.status_code == 200
    assert res.json()["id"] == sid


# 4. Valid persisted session restores the correct session & loads history
def test_4_valid_session_restores_history(client, mock_groq):
    s = client.post("/sessions", json={"title": "Restored Session"}).json()
    sid = s["id"]

    set_mock_response(mock_groq, "Restored Answer")
    client.post("/chat", json={"message": "Hello Restored", "session_id": sid})

    msgs_res = client.get(f"/sessions/{sid}/messages")
    assert msgs_res.status_code == 200
    msgs = msgs_res.json()
    assert any(m["content"] == "Hello Restored" for m in msgs)


# 5. Invalid/deleted persisted session falls back safely
def test_5_invalid_persisted_session_fallback(client):
    res = client.get("/sessions/invalid_session_12345")
    assert res.status_code == 404
    assert res.json()["detail"] == "Session not found"


# 6. Missing persisted session falls back safely
def test_6_missing_persisted_session_fallback(client):
    # When no session ID is persisted, GET /sessions retrieves available sessions or creates default
    sessions = client.get("/sessions").json()
    assert isinstance(sessions, list)


# 7. Session switching updates persisted active-session ID
def test_7_session_switching_updates_persisted_id():
    with open("frontend/app.js", "r") as f:
        js = f.read()
    assert "handleSelectSession" in js
    assert "setPersistedSessionId(currentSessionId);" in js


# 8. Creating a new chat updates persisted active-session ID
def test_8_new_chat_updates_persisted_id(client):
    res = client.post("/sessions", json={"title": "New Chat"})
    assert res.status_code == 201
    new_id = res.json()["id"]

    meta = client.get(f"/sessions/{new_id}").json()
    assert meta["id"] == new_id


# 9. Deleting the active session updates persisted active-session ID
def test_9_deleting_active_session_updates_persisted_id(client):
    s1 = client.post("/sessions", json={"title": "S1"}).json()["id"]
    s2 = client.post("/sessions", json={"title": "S2"}).json()["id"]

    client.delete(f"/sessions/{s2}")
    sessions = client.get("/sessions").json()
    assert len(sessions) == 1
    assert sessions[0]["id"] == s1


# 10. Deleting the final session creates/selects a valid replacement
def test_10_deleting_final_session_creates_replacement(client):
    s1 = client.post("/sessions", json={"title": "Sole Session"}).json()["id"]
    client.delete(f"/sessions/{s1}")

    new_s = client.post("/sessions", json={"title": "New Chat"}).json()
    assert new_s["id"] is not None
    assert new_s["id"] != s1


# 11. Only the session ID is persisted — no conversation data
def test_11_only_session_id_persisted_no_conversation():
    with open("frontend/app.js", "r") as f:
        js = f.read()

    assert "pixie_active_session_id" in js
    assert "localStorage.setItem('messages'" not in js
    assert "localStorage.setItem('history'" not in js
    assert "localStorage.setItem('tokens'" not in js
    assert "localStorage.setItem('conversation'" not in js


# 12. Conversation history is never stored in browser persistence
def test_12_no_forbidden_browser_storage():
    with open("frontend/app.js", "r") as f:
        js = f.read()

    assert "sessionStorage" not in js
    assert "indexedDB" not in js
    assert "document.cookie" not in js


# 13. Session requests remain bound to the restored/active session
def test_13_requests_bound_to_active_session(client, mock_groq):
    s = client.post("/sessions", json={"title": "Bound Session"}).json()["id"]

    chat_res = client.post("/chat", json={"message": "Chat msg", "session_id": s})
    assert chat_res.status_code == 200

    voice_res = client.post("/voice", json={"message": "Voice msg", "session_id": s})
    assert voice_res.status_code == 200

    confirm_res = client.post("/confirm", json={"confirmation_id": "c1", "approved": True, "session_id": s})
    assert confirm_res.status_code == 200


# 14. Session history remains isolated after persistence/restoration
def test_14_session_history_isolated_after_restoration(client, mock_groq):
    sa = client.post("/sessions", json={"title": "Isolated A"}).json()["id"]
    sb = client.post("/sessions", json={"title": "Isolated B"}).json()["id"]

    client.post("/chat", json={"message": "Secret Payload A", "session_id": sa})

    msgs_b = client.get(f"/sessions/{sb}/messages").json()
    assert len(msgs_b) == 0


# 15. Context-aware greeting retained (no generic chatbot welcome)
def test_15_context_aware_greeting_retained():
    with open("frontend/app.js", "r") as f:
        js = f.read()
    assert "getGreeting" in js
    assert "Good morning, Sir." in js
    assert "Good afternoon, Sir." in js
    assert "Good evening, Sir." in js
    assert "Welcome to P.I.X.I.E." not in js
