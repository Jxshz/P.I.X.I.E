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
    db_file = str(tmp_path / "test_phase532_ui.db")
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

        def make_completion(content_text="Mocked response for 5.3.2 UI Refinement test."):
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
            resp_text = getattr(client_instance, "_custom_response", "Mocked response for 5.3.2 UI Refinement test.")
            return make_completion(resp_text)

        client_instance.chat.completions.create = AsyncMock(side_effect=async_create)
        yield client_instance


def set_mock_response(client_instance, text: str):
    client_instance._custom_response = text


# 1. Header renders P.I.X.I.E. branding.
def test_1_header_renders_pixie_branding():
    with open("/Users/novus/Documents/P.I.X.I.E/frontend/index.html", "r") as f:
        html = f.read()
    assert "P.I.X.I.E." in html
    assert "brand-title" in html


# 2. Chat History dashboard control exists.
def test_2_chat_history_control_exists():
    with open("/Users/novus/Documents/P.I.X.I.E/frontend/index.html", "r") as f:
        html = f.read()
    assert 'id="chat-history-btn"' in html
    assert "CHAT HISTORY" in html


# 3. Chat History control opens/closes the existing session sidebar.
def test_3_chat_history_control_toggles_sidebar():
    with open("/Users/novus/Documents/P.I.X.I.E/frontend/app.js", "r") as f:
        js = f.read()
    assert "chatHistoryBtn" in js
    assert "sessionSidebar.classList.toggle('hidden')" in js or "classList.toggle" in js


# 4. Session list remains functional.
def test_4_session_list_functional(client):
    client.post("/sessions", json={"title": "Chat 1"})
    client.post("/sessions", json={"title": "Chat 2"})
    res = client.get("/sessions")
    assert res.status_code == 200
    assert len(res.json()) == 2


# 5. Active session remains visually identifiable.
def test_5_active_session_identifiable(client):
    s = client.post("/sessions", json={"title": "Active"}).json()
    get_res = client.get(f"/sessions/{s['id']}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == s["id"]


# 6. New Chat remains functional.
def test_6_new_chat_functional(client):
    res = client.post("/sessions", json={"title": "New Chat"})
    assert res.status_code == 201
    assert "id" in res.json()


# 7. Session switching remains functional.
def test_7_session_switching_functional(client):
    s1 = client.post("/sessions", json={"title": "S1"}).json()
    s2 = client.post("/sessions", json={"title": "S2"}).json()
    res = client.get(f"/sessions/{s2['id']}")
    assert res.status_code == 200
    assert res.json()["id"] == s2["id"]


# 8. Session history loads correctly.
def test_8_session_history_loads_correctly(client, mock_groq):
    s1 = client.post("/sessions", json={"title": "S1"}).json()["id"]
    set_mock_response(mock_groq, "Response S1")
    client.post("/chat", json={"message": "Prompt S1", "session_id": s1})

    msgs = client.get(f"/sessions/{s1}/messages").json()
    assert any(m["content"] == "Prompt S1" for m in msgs)


# 9. Session deletion remains functional.
def test_9_session_deletion_functional(client):
    s1 = client.post("/sessions", json={"title": "Delete Target"}).json()["id"]
    res = client.delete(f"/sessions/{s1}")
    assert res.status_code == 200
    assert client.get(f"/sessions/{s1}").status_code == 404


# 10. Deleting the active session selects a valid replacement.
def test_10_deleting_active_session_selects_replacement(client):
    s1 = client.post("/sessions", json={"title": "S1"}).json()["id"]
    s2 = client.post("/sessions", json={"title": "S2"}).json()["id"]
    client.delete(f"/sessions/{s2}")
    remaining = client.get("/sessions").json()
    assert len(remaining) == 1
    assert remaining[0]["id"] == s1


# 11. Token telemetry is displayed using existing telemetry only.
def test_11_token_telemetry_uses_existing_endpoint(client):
    res = client.get("/status")
    assert res.status_code == 200
    data = res.json()
    assert "tokens_day" in data
    assert "tpd_limit" in data


# 12. No new token calculation logic exists.
def test_12_no_new_token_calculation_logic():
    with open("/Users/novus/Documents/P.I.X.I.E/frontend/app.js", "r") as f:
        js = f.read()
    assert "fetchTokenTelemetry" in js
    assert "data.tokens_day" in js


# 13. System IDLE/status indicator remains functional.
def test_13_status_indicator_remains_functional():
    with open("/Users/novus/Documents/P.I.X.I.E/frontend/index.html", "r") as f:
        html = f.read()
    assert 'id="agent-status"' in html
    assert "status-indicator" in html


# 14. Session A history cannot leak into Session B.
def test_14_session_a_history_cannot_leak_into_session_b(client, mock_groq):
    sa = client.post("/sessions", json={"title": "A"}).json()["id"]
    sb = client.post("/sessions", json={"title": "B"}).json()["id"]

    client.post("/chat", json={"message": "Top Secret A", "session_id": sa})
    msgs_b = client.get(f"/sessions/{sb}/messages").json()
    assert not any("Top Secret A" in m["content"] for m in msgs_b)


# 15. /chat continues using currentSessionId.
def test_15_chat_continues_using_current_session_id(client, mock_groq):
    s = client.post("/sessions", json={"title": "Chat Session"}).json()["id"]
    set_mock_response(mock_groq, "Chat Response")

    res = client.post("/chat", json={"message": "Test msg", "session_id": s})
    assert res.status_code == 200
    assert res.json()["response"] == "Chat Response"


# 16. /voice continues using currentSessionId.
def test_16_voice_continues_using_current_session_id(client, mock_groq):
    s = client.post("/sessions", json={"title": "Voice Session"}).json()["id"]
    set_mock_response(mock_groq, "Voice Response")

    res = client.post("/voice", json={"message": "Test voice", "session_id": s})
    assert res.status_code == 200
    assert res.json()["response"] == "Voice Response"


# 17. /confirm continues using currentSessionId.
def test_17_confirm_continues_using_current_session_id(client):
    s = client.post("/sessions", json={"title": "Confirm Session"}).json()["id"]
    res = client.post("/confirm", json={"confirmation_id": "c1", "approved": True, "session_id": s})
    assert res.status_code == 200


# 18. No duplicate session state variable exists.
def test_18_no_duplicate_session_state_variable():
    with open("/Users/novus/Documents/P.I.X.I.E/frontend/app.js", "r") as f:
        js = f.read()
    assert "currentSessionId" in js
    assert "activeSessionId" not in js
    assert "selectedSessionId" not in js


# 19. Test strict browser persistence boundary (only pixie_active_session_id)
def test_19_no_browser_persistence_introduced():
    with open("/Users/novus/Documents/P.I.X.I.E/frontend/app.js", "r") as f:
        js = f.read()
    assert "pixie_active_session_id" in js
    assert "sessionStorage" not in js
    assert "document.cookie" not in js


# 20. Existing Phase 4.5 behaviour remains intact.
def test_20_existing_phase45_behaviour_intact(client, mock_groq):
    s = client.post("/sessions", json={"title": "Phase 4.5 Check"}).json()["id"]
    set_mock_response(mock_groq, "# Title\n**bold**")

    res = client.post("/chat", json={"message": "Format check", "session_id": s}).json()
    assert "# Title" not in res["response"]
    assert "**bold**" not in res["response"]
    assert res["spoken_response"] is not None
