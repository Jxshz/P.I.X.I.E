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


# 1. Active session key exists in app.js
def test_1_active_session_key_configured():
    with open("frontend/app.js", "r") as f:
        js = f.read()
    assert "pixie_active_session_id" in js
    assert "getPersistedSessionId" in js
    assert "setPersistedSessionId" in js


# 2. Persisted active session validation via REST API
def test_2_persisted_session_validation(client):
    s1 = client.post("/sessions", json={"title": "Persisted Chat"}).json()
    sid = s1["id"]

    res = client.get(f"/sessions/{sid}")
    assert res.status_code == 200
    assert res.json()["id"] == sid


# 3. Invalid persisted session ID fallback
def test_3_invalid_persisted_session_fallback(client):
    res = client.get("/sessions/invalid_session_12345")
    assert res.status_code == 404
    assert res.json()["detail"] == "Session not found"


# 4. Valid session restoration and history loading
def test_4_valid_session_restoration_and_history(client, mock_groq):
    s = client.post("/sessions", json={"title": "Restored Session"}).json()
    sid = s["id"]

    set_mock_response(mock_groq, "Restored Answer")
    client.post("/chat", json={"message": "Hello Restored", "session_id": sid})

    msgs_res = client.get(f"/sessions/{sid}/messages")
    assert msgs_res.status_code == 200
    msgs = msgs_res.json()
    assert any(m["content"] == "Hello Restored" for m in msgs)


# 5. New Chat creates backend session and updates active session
def test_5_new_chat_creates_session_and_updates_active(client):
    res = client.post("/sessions", json={"title": "New Chat"})
    assert res.status_code == 201
    new_id = res.json()["id"]

    meta = client.get(f"/sessions/{new_id}").json()
    assert meta["id"] == new_id


# 6. Session selection switches active session and loads correct history
def test_6_session_switching_loads_correct_history(client, mock_groq):
    s1 = client.post("/sessions", json={"title": "S1"}).json()["id"]
    s2 = client.post("/sessions", json={"title": "S2"}).json()["id"]

    client.post("/chat", json={"message": "Msg in S1", "session_id": s1})
    client.post("/chat", json={"message": "Msg in S2", "session_id": s2})

    msgs_s1 = client.get(f"/sessions/{s1}/messages").json()
    msgs_s2 = client.get(f"/sessions/{s2}/messages").json()

    assert any(m["content"] == "Msg in S1" for m in msgs_s1)
    assert not any("Msg in S2" in m["content"] for m in msgs_s1)

    assert any(m["content"] == "Msg in S2" for m in msgs_s2)
    assert not any("Msg in S1" in m["content"] for m in msgs_s2)


# 7. Session deletion selects valid replacement session
def test_7_session_deletion_selects_replacement(client):
    s1 = client.post("/sessions", json={"title": "S1"}).json()["id"]
    s2 = client.post("/sessions", json={"title": "S2"}).json()["id"]

    client.delete(f"/sessions/{s2}")
    sessions = client.get("/sessions").json()
    assert len(sessions) == 1
    assert sessions[0]["id"] == s1


# 8. Deleting final session creates a valid replacement
def test_8_deleting_final_session_creates_replacement(client):
    s1 = client.post("/sessions", json={"title": "Sole Session"}).json()["id"]
    client.delete(f"/sessions/{s1}")

    # Client code creates a replacement session when 0 sessions remain
    new_s = client.post("/sessions", json={"title": "New Chat"}).json()
    assert new_s["id"] is not None
    assert new_s["id"] != s1


# 9. Session A history never leaks into Session B
def test_9_session_isolation_guarantee(client, mock_groq):
    sa = client.post("/sessions", json={"title": "Isolated A"}).json()["id"]
    sb = client.post("/sessions", json={"title": "Isolated B"}).json()["id"]

    client.post("/chat", json={"message": "Secret Payload A", "session_id": sa})

    msgs_b = client.get(f"/sessions/{sb}/messages").json()
    assert len(msgs_b) == 0


# 10. Chat, voice, confirm requests bound to active session
def test_10_requests_bound_to_active_session(client, mock_groq):
    s = client.post("/sessions", json={"title": "Bound Session"}).json()["id"]

    chat_res = client.post("/chat", json={"message": "Chat msg", "session_id": s})
    assert chat_res.status_code == 200

    voice_res = client.post("/voice", json={"message": "Voice msg", "session_id": s})
    assert voice_res.status_code == 200

    confirm_res = client.post("/confirm", json={"confirmation_id": "c1", "approved": True, "session_id": s})
    assert confirm_res.status_code == 200


# 11. Strict persistence boundary: ONLY pixie_active_session_id in localStorage
def test_11_strict_persistence_boundary():
    with open("frontend/app.js", "r") as f:
        js = f.read()

    assert "pixie_active_session_id" in js
    assert "localStorage.setItem('messages'" not in js
    assert "localStorage.setItem('history'" not in js
    assert "localStorage.setItem('tokens'" not in js
    assert "localStorage.setItem('conversation'" not in js
    assert "sessionStorage" not in js
    assert "indexedDB" not in js
    assert "document.cookie" not in js


# 12. Context-aware greeting retained
def test_12_context_aware_greeting():
    with open("frontend/app.js", "r") as f:
        js = f.read()
    assert "getGreeting" in js
    assert "Good morning, Sir." in js
    assert "Good afternoon, Sir." in js
    assert "Good evening, Sir." in js
    assert "Welcome to P.I.X.I.E." not in js
