import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.agent.personality import SYSTEM_PROMPT, generate_spoken_response

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_chat_endpoint_structure():
    pass

@pytest.fixture
def mock_agent(monkeypatch):
    async def mock_process_intent(self, message: str) -> tuple[str, str]:
        return "Mocked response", "Mocked spoken"
    
    from backend.agent.core import AgentCore
    monkeypatch.setattr(AgentCore, "process_intent", mock_process_intent)

def test_chat_endpoint_success(mock_agent):
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    assert "response" in response.json()
    assert response.json()["response"] == "Mocked response"
    assert response.json()["spoken_response"] == "Mocked spoken"

def test_voice_endpoint_success(mock_agent):
    response = client.post("/voice", json={"message": "Hello from voice"})
    assert response.status_code == 200
    assert "response" in response.json()
    assert response.json()["response"] == "Mocked response"
    assert response.json()["spoken_response"] == "Mocked spoken"

def test_system_prompt_loading():
    assert "P.I.X.I.E." in SYSTEM_PROMPT
    assert "Sir" in SYSTEM_PROMPT

def test_speech_cleanup_bold():
    assert generate_spoken_response("Hello **Joshva**.") == "Hello Joshva."

def test_speech_cleanup_italic():
    assert generate_spoken_response("That is *important*.") == "That is important."

def test_speech_cleanup_heading():
    assert generate_spoken_response("## System Status") == "System Status"

def test_speech_cleanup_inline_code():
    assert generate_spoken_response("Run `python app.py`.") == "Run python app.py."

def test_speech_cleanup_markdown_link():
    assert generate_spoken_response("Open the [dashboard](https://example.com).") == "Open the dashboard."

def test_speech_cleanup_bullet():
    assert generate_spoken_response("- Open Safari") == "Open Safari"

def test_speech_cleanup_stray_markdown():
    assert generate_spoken_response("Hello **world** *again*.") == "Hello world again."

def test_speech_cleanup_contractions():
    assert generate_spoken_response("I'm ready and I don't have a problem.") == "I'm ready and I don't have a problem."

def test_speech_cleanup_pixie_pronunciation():
    spoken = generate_spoken_response("P.I.X.I.E. is here.")
    assert "Pixie" in spoken
    assert "P.I.X.I.E." not in spoken

def test_speech_cleanup_mixed_markdown():
    text = "## Status\n\nP.I.X.I.E. is **online**.\n\n- Voice is ready\n- Backend is healthy"
    spoken = generate_spoken_response(text)
    assert spoken == "Status Pixie is online. Voice is ready Backend is healthy"
