import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.agent.personality import SYSTEM_PROMPT, generate_spoken_response

client = TestClient(app)

@pytest.fixture(autouse=True)
def isolate_global_db(tmp_path):
    from backend.main import agent
    from backend.storage.usage_store import UsageStore
    agent.usage_store = UsageStore(db_path=str(tmp_path / "test_global.db"))

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_chat_endpoint_structure():
    pass

@pytest.fixture
def mock_agent(monkeypatch):
    async def mock_process_intent(self, message: str) -> tuple[str, str, dict | None]:
        return "Mocked response", "Mocked spoken", None

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
    assert spoken == "Status Pixie is online. Voice is ready, Backend is healthy."


import time
import asyncio
import pytest
from backend.agent.token_governor import TokenGovernor, Reservation

def test_governor_initializes():
    gov = TokenGovernor()
    assert gov.rpm_limit > 0
    assert gov.tpm_limit > 0
    assert gov.rpd_limit > 0
    assert gov.tpd_limit > 0

def test_governor_allowed():
    gov = TokenGovernor()
    is_allowed, msg, res = gov.preflight([{"role": "user", "content": "Hello"}])
    assert is_allowed is True
    assert msg == ""
    assert isinstance(res, Reservation)

def test_governor_rpm_limit(monkeypatch):
    gov = TokenGovernor()
    gov.rpm_limit = 2
    is_allowed, msg, res1 = gov.preflight([{"role": "user", "content": "Hello"}])
    gov.record_usage(res1)
    is_allowed, msg, res2 = gov.preflight([{"role": "user", "content": "Hello"}])
    gov.record_usage(res2)
    is_allowed, msg, res3 = gov.preflight([{"role": "user", "content": "Hello"}])
    assert is_allowed is False
    assert "quickly" in msg
    assert res3 is None

def test_governor_tpm_limit():
    gov = TokenGovernor()
    gov.tpm_limit = 100
    gov.max_completion_tokens = 10
    # Use a large string to exceed 90 input tokens. ~375 tokens + 10 completion = 385 tokens.
    is_allowed, msg, res = gov.preflight([{"role": "user", "content": "A"*1000}])
    assert is_allowed is False
    assert "short-term" in msg
    assert res is None

def test_governor_rpd_limit():
    gov = TokenGovernor()
    gov.rpd_limit = 2
    is_allowed, msg, res1 = gov.preflight([{"role": "user", "content": "Hello"}])
    gov.record_usage(res1)
    is_allowed, msg, res2 = gov.preflight([{"role": "user", "content": "Hello"}])
    gov.record_usage(res2)
    is_allowed, msg, res3 = gov.preflight([{"role": "user", "content": "Hello"}])
    assert is_allowed is False
    assert "today" in msg
    assert res3 is None

def test_governor_tpd_limit():
    gov = TokenGovernor()
    gov.tpd_limit = 50
    gov.max_completion_tokens = 5
    # ~37 tokens + 5 completion = 42 tokens
    is_allowed, msg, res1 = gov.preflight([{"role": "user", "content": "A"*100}])
    gov.record_usage(res1)
    is_allowed, msg, res2 = gov.preflight([{"role": "user", "content": "A"*100}])
    assert is_allowed is False
    assert "today" in msg
    assert res2 is None

def test_governor_usage_recorded():
    gov = TokenGovernor()
    is_allowed, msg, res = gov.preflight([{"role": "user", "content": "Hello"}])
    gov.record_usage(res)
    status = gov.get_status()
    assert status["requests_minute"] == 1
    assert status["tokens_minute"] == res.tokens

def test_governor_usage_from_groq_object():
    class MockUsage:
        total_tokens = 42

    gov = TokenGovernor()
    is_allowed, msg, res = gov.preflight([{"role": "user", "content": "Hello"}])
    gov.record_usage(res, MockUsage())
    assert gov.get_status()["tokens_minute"] == 42

def test_governor_missing_groq_usage():
    gov = TokenGovernor()
    is_allowed, msg, res = gov.preflight([{"role": "user", "content": "Hello"}])
    gov.record_usage(res, None)
    assert gov.get_status()["tokens_minute"] == res.tokens

def test_minute_window_resets(monkeypatch):
    gov = TokenGovernor()
    current_time = 1000.0
    monkeypatch.setattr(time, "time", lambda: current_time)

    is_allowed, msg, res = gov.preflight([{"role": "user", "content": "Hello"}])
    gov.record_usage(res)

    current_time = 1061.0 # 61 seconds later
    status = gov.get_status()
    assert status["requests_minute"] == 0
    assert status["tokens_minute"] == 0
    assert status["requests_day"] == 1
    assert status["tokens_day"] == res.tokens

def test_day_window_resets(monkeypatch):
    gov = TokenGovernor()
    current_time = 1000.0
    monkeypatch.setattr(time, "time", lambda: current_time)

    is_allowed, msg, res = gov.preflight([{"role": "user", "content": "Hello"}])
    gov.record_usage(res)

    current_time = 87401.0 # 1 day + 1 second later
    status = gov.get_status()
    assert status["requests_day"] == 0
    assert status["tokens_day"] == 0

@pytest.mark.asyncio
async def test_context_trimming():
    from backend.agent.core import AgentCore
    core = AgentCore(db_path=":memory:")
    # Fill context with many messages
    core.conversation_history = [{"role": "system", "content": "SYS"}]
    for i in range(20):
        core.conversation_history.append({"role": "user", "content": "A"*1000})
        core.conversation_history.append({"role": "assistant", "content": "B"*1000})

    class MockMessage:
        content = "Response"
        tool_calls = None
    class MockChoice:
        message = MockMessage()
    class MockCompletion:
        choices = [MockChoice()]
        usage = None
    class MockCreate:
        async def create(self, **kwargs):
            return MockCompletion()

    core.client.chat.completions = MockCreate()

    await core.process_intent("Hello")

    assert core.conversation_history[0]["role"] == "system"
    assert core.conversation_history[-2]["role"] == "user"
    assert core.conversation_history[-2]["content"] == "Hello"
    # Should have trimmed significantly
    assert len(core.conversation_history) < 41

@pytest.mark.asyncio
async def test_governor_denial_does_not_call_groq():
    from backend.agent.core import AgentCore
    core = AgentCore(db_path=":memory:")
    core.governor.rpm_limit = 0 # Force denial

    groq_called = False
    class MockCreate:
        async def create(self, **kwargs):
            nonlocal groq_called
            groq_called = True
            return None
    core.client.chat.completions = MockCreate()

    from backend.agent.core import RateLimitException
    with pytest.raises(RateLimitException) as excinfo:
        await core.process_intent("Hello")
    assert groq_called is False
    assert "quickly" in str(excinfo.value)

    # Ensure user message wasn't stuck in history
    assert len(core.conversation_history) == 1
    assert core.conversation_history[0]["role"] == "system"

@pytest.mark.asyncio
async def test_failed_groq_releases_reservation():
    from backend.agent.core import AgentCore
    core = AgentCore(db_path=":memory:")

    class MockCreateError:
        async def create(self, **kwargs):
            raise ValueError("Groq API Timeout")

    core.client.chat.completions = MockCreateError()

    resp, spoken, _ = await core.process_intent("Hello")
    assert "Error connecting" in resp

    # Check that reservation is released (usage should be 0)
    status = core.governor.get_status()
    assert status["requests_minute"] == 0
    assert status["tokens_minute"] == 0

@pytest.mark.asyncio
async def test_concurrent_requests_do_not_double_admit():
    from backend.agent.core import AgentCore
    core = AgentCore(db_path=":memory:")
    core.governor.rpm_limit = 1

    # We will simulate a slow Groq call to ensure concurrency gap is closed
    class SlowMockCreate:
        async def create(self, **kwargs):
            await asyncio.sleep(0.1)
            class MockMessage:
                content = "Response"
                tool_calls = None
            class MockChoice:
                message = MockMessage()
            class MockCompletion:
                choices = [MockChoice()]
                usage = None
            return MockCompletion()

    core.client.chat.completions = SlowMockCreate()

    # Fire two requests concurrently
    task1 = asyncio.create_task(core.process_intent("Hello 1"))
    task2 = asyncio.create_task(core.process_intent("Hello 2"))

    res1, res2 = await asyncio.gather(task1, task2, return_exceptions=True)

    from backend.agent.core import RateLimitException
    exceptions = [r for r in (res1, res2) if isinstance(r, RateLimitException)]
    successes = [r for r in (res1, res2) if not isinstance(r, Exception)]

    assert len(exceptions) == 1
    assert len(successes) == 1
    assert "quickly" in exceptions[0].message
    assert "Response" in successes[0][0]

@pytest.mark.asyncio
async def test_governor_failure_fails_closed():
    from backend.agent.core import AgentCore
    core = AgentCore(db_path=":memory:")

    # Force a failure inside the governor
    def failing_preflight(*args, **kwargs):
        raise ValueError("Governor crashed")

    core.governor.preflight = failing_preflight

    resp, spoken, _ = await core.process_intent("Hello")
    assert "experiencing issues" in resp


def test_status_endpoint_structure():
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "online"
    assert "model" in data
    assert "requests_minute" in data
    assert "tokens_minute" in data
    assert "requests_day" in data
    assert "tokens_day" in data
    assert "rpm_limit" in data
    assert "tpm_limit" in data
    assert "rpd_limit" in data
    assert "tpd_limit" in data
    assert "rpm_remaining" in data
    assert "tpm_remaining" in data
    assert "rpd_remaining" in data
    assert "tpd_remaining" in data

    # Ensure remaining calculates correctly
    assert data["rpm_remaining"] == max(0, data["rpm_limit"] - data["requests_minute"])
    assert data["tpm_remaining"] == max(0, data["tpm_limit"] - data["tokens_minute"])

    # Ensure no secrets leaked
    assert "api_key" not in str(data).lower()
    assert "groq" not in str(data).lower()

@pytest.mark.asyncio
async def test_status_updates_after_request():
    from backend.main import agent

    # Reset governor for clean test
    agent.governor._minute_window.clear()
    agent.governor._day_window.clear()

    class MockMessage:
        content = "Response"
        tool_calls = None
    class MockChoice:
        message = MockMessage()
    class MockUsage:
        total_tokens = 42
    class MockCompletion:
        choices = [MockChoice()]
        usage = MockUsage()
    class MockCreate:
        async def create(self, **kwargs):
            return MockCompletion()

    # Swap out the groq client
    agent.client.chat.completions = MockCreate()

    # Clear history so estimate is deterministic
    agent.conversation_history = [{"role": "system", "content": "SYS"}]

    # Initial status
    resp1 = client.get("/status")
    assert resp1.json()["requests_minute"] == 0
    assert resp1.json()["tokens_minute"] == 0

    # Make a chat request via the real endpoint (which calls real process_intent, using mocked Groq)
    chat_resp = client.post("/chat", json={"message": "Hello"})
    assert chat_resp.status_code == 200

    # Check status again
    resp2 = client.get("/status")
    data2 = resp2.json()
    assert data2["requests_minute"] == 1
    # 42 total tokens from the MockUsage
    assert data2["tokens_minute"] == 42
    assert data2["rpm_remaining"] == data2["rpm_limit"] - 1
    assert data2["tpm_remaining"] == data2["tpm_limit"] - 42

def test_chat_endpoint_429():
    from backend.main import agent
    # Force governor denial
    old_limit = agent.governor.rpm_limit
    agent.governor.rpm_limit = 0

    response = client.post("/chat", json={"message": "Hello"})

    agent.governor.rpm_limit = old_limit

    assert response.status_code == 429
    data = response.json()
    assert "response" in data
    assert "quickly" in data["response"]
