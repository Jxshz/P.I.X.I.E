import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from backend.main import app, session_manager
from backend.agent.session_manager import SessionManager
from backend.storage.session_store import SessionStore


@pytest.fixture
def client(tmp_path):
    db_file = str(tmp_path / "test_sessions_api.db")
    store = SessionStore(db_file)
    test_mgr = SessionManager(session_store=store)

    with patch("backend.main.session_manager", test_mgr):
        with TestClient(app) as test_client:
            yield test_client

    test_mgr.close()


def test_post_creates_session(client):
    response = client.post("/sessions", json={"title": "New Research"})
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["title"] == "New Research"
    assert "created_at" in data
    assert "updated_at" in data


def test_post_supports_custom_title(client):
    response = client.post("/sessions", json={"title": "Project Alpha", "session_id": "api-sess-alpha"})
    assert response.status_code == 201
    data = response.json()
    assert data["id"] == "api-sess-alpha"
    assert data["title"] == "Project Alpha"


def test_get_lists_sessions(client):
    s1 = client.post("/sessions", json={"title": "Session 1"}).json()
    s2 = client.post("/sessions", json={"title": "Session 2"}).json()

    response = client.get("/sessions?limit=50")
    assert response.status_code == 200
    sessions = response.json()
    assert isinstance(sessions, list)
    ids = [s["id"] for s in sessions]
    assert s1["id"] in ids
    assert s2["id"] in ids


def test_get_retrieves_existing_session(client):
    created = client.post("/sessions", json={"title": "Target Session"}).json()
    sid = created["id"]

    response = client.get(f"/sessions/{sid}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == sid
    assert data["title"] == "Target Session"


def test_get_unknown_session_returns_404(client):
    response = client.get("/sessions/unknown_non_existent_id_999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


def test_patch_renames_existing_session(client):
    created = client.post("/sessions", json={"title": "Original Title"}).json()
    sid = created["id"]

    response = client.patch(f"/sessions/{sid}", json={"title": "Updated Title"})
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == sid
    assert data["title"] == "Updated Title"

    # Verify GET returns updated title
    get_res = client.get(f"/sessions/{sid}").json()
    assert get_res["title"] == "Updated Title"


def test_patch_unknown_session_returns_404(client):
    response = client.patch("/sessions/unknown_id_999", json={"title": "New Title"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


def test_delete_removes_existing_session(client):
    created = client.post("/sessions", json={"title": "To Delete"}).json()
    sid = created["id"]

    response = client.delete(f"/sessions/{sid}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deleted"
    assert data["session_id"] == sid

    # Verify GET returns 404 after deletion
    get_res = client.get(f"/sessions/{sid}")
    assert get_res.status_code == 404


def test_delete_unknown_session_returns_404(client):
    response = client.delete("/sessions/unknown_id_999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


import backend.main as main_module


def test_deleted_session_messages_removed(client):
    created = client.post("/sessions", json={"title": "Message Cascade Test"}).json()
    sid = created["id"]

    # Add message directly to SessionStore
    main_module.session_manager.session_store.add_message(sid, "user", "Test message")
    assert len(main_module.session_manager.session_store.get_messages(sid)) == 1

    # Delete session
    client.delete(f"/sessions/{sid}")

    # Messages cascade deleted
    assert len(main_module.session_manager.session_store.get_messages(sid)) == 0


def test_deleted_session_removed_from_cache(client):
    created = client.post("/sessions", json={"title": "Cache Purge Test"}).json()
    sid = created["id"]

    # Access session to ensure it's in SessionManager cache
    main_module.session_manager.get_session(sid)
    assert sid in main_module.session_manager._active_agents

    client.delete(f"/sessions/{sid}")
    assert sid not in main_module.session_manager._active_agents
    assert sid not in main_module.session_manager._session_locks


def test_api_uses_same_application_level_session_manager(client):
    response = client.post("/sessions", json={"title": "App Instance Test"})
    sid = response.json()["id"]

    # Must be accessible via app's session_manager instance directly
    assert main_module.session_manager.get_session(sid) is not None


def test_concurrent_requests_different_sessions_isolated(client):
    s_a = client.post("/sessions", json={"title": "Session A"}).json()["id"]
    s_b = client.post("/sessions", json={"title": "Session B"}).json()["id"]

    agent_a = main_module.session_manager.get_session(s_a)
    agent_b = main_module.session_manager.get_session(s_b)

    assert agent_a is not agent_b
    assert agent_a.session_id == s_a
    assert agent_b.session_id == s_b


def test_existing_phase45_api_response_contracts_unchanged(client):
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model" in data
    assert "requests_minute" in data
    assert "tokens_minute" in data

    hist_res = client.get("/usage/history")
    assert hist_res.status_code == 200

    clear_res = client.post("/api/clear")
    assert clear_res.status_code == 200
    assert clear_res.json() == {"status": "Context cleared"}


def test_existing_non_session_endpoints_functional(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
