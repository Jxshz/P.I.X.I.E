import pytest
import asyncio
from backend.agent.core import AgentCore, RateLimitException

@pytest.fixture
def agent():
    return AgentCore(db_path=":memory:")

@pytest.mark.asyncio
async def test_max_iterations_exhaustion(agent, monkeypatch):
    """
    Test A: Mock the model so every inference returns a tool call.
    Verify the returned error is in conversation_history exactly once.
    """
    class MockMessage:
        content = None
        class MockToolCall:
            id = "call_123"
            class MockFunction:
                name = "system_diagnostics"
                arguments = "{}"
            function = MockFunction()
        tool_calls = [MockToolCall()]

    class MockChoice:
        message = MockMessage()

    class MockCompletion:
        choices = [MockChoice()]
        usage = None

    class MockCompletions:
        async def create(self, **kwargs):
            return MockCompletion()

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    agent.client = MockClient()
    
    # Bypass TokenGovernor for this test to avoid RateLimitException on the 4th/5th calls
    agent.governor.preflight = lambda history: (True, "", "mock_reservation")
    agent.governor.record_usage = lambda res, usage=None, failed=False: None
    
    # Run the loop
    res_text, res_spoken = await agent.process_intent("Hello")
    
    # Expected error message from max exhaustion
    expected_error = "I encountered an issue processing the tool results, taking too many steps."
    
    assert res_text == expected_error
    
    # Verify the error is present in history and is the last message
    last_message = agent.conversation_history[-1]
    assert last_message["role"] == "assistant"
    assert last_message["content"] == expected_error
    
    # Verify it appears exactly once
    error_count = sum(1 for msg in agent.conversation_history if msg.get("content") == expected_error)
    assert error_count == 1
    
    # Next-turn integrity verification
    res_text2, _ = await agent.process_intent("Next message")
    assert res_text2 == expected_error # Still loops endlessly with our mock
    
    # Check that previous error wasn't popped or mangled incorrectly
    # New history should have appended User -> [tool calls x3] -> Assistant error
    # We just ensure the history ends cleanly
    assert agent.conversation_history[-1]["role"] == "assistant"
    assert agent.conversation_history[-1]["content"] == expected_error

@pytest.mark.asyncio
async def test_api_exception(agent, monkeypatch):
    """
    Test B: Mock the API to raise an exception during processing.
    Verify the error response is appended exactly once.
    """
    class MockCompletions:
        async def create(self, **kwargs):
            raise ValueError("Simulated API failure")

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    agent.client = MockClient()

    res_text, res_spoken = await agent.process_intent("Hello")
    
    expected_error = "Error connecting to Groq API: Simulated API failure"
    assert res_text == expected_error
    
    last_message = agent.conversation_history[-1]
    assert last_message["role"] == "assistant"
    assert last_message["content"] == expected_error
    
    error_count = sum(1 for msg in agent.conversation_history if msg.get("content") == expected_error)
    assert error_count == 1
    
    # Next turn verification
    res_text2, _ = await agent.process_intent("Hello again")
    assert res_text2 == expected_error
    assert agent.conversation_history[-1]["role"] == "assistant"
    assert agent.conversation_history[-1]["content"] == expected_error

@pytest.mark.asyncio
async def test_normal_completion(agent, monkeypatch):
    """
    Test C: Verify normal completion appends exactly one assistant message.
    """
    class MockMessage:
        content = "Normal response"
        tool_calls = None

    class MockChoice:
        message = MockMessage()

    class MockCompletion:
        choices = [MockChoice()]
        usage = None

    class MockCompletions:
        async def create(self, **kwargs):
            return MockCompletion()

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    agent.client = MockClient()

    res_text, res_spoken = await agent.process_intent("Hello")
    assert res_text == "Normal response"
    
    last_message = agent.conversation_history[-1]
    assert last_message["role"] == "assistant"
    assert last_message["content"] == "Normal response"
    
    count = sum(1 for msg in agent.conversation_history if msg.get("content") == "Normal response")
    assert count == 1
    
    # Assert exact sequence: System -> User -> Assistant
    assert len(agent.conversation_history) == 3

@pytest.mark.asyncio
async def test_normal_tool_completion(agent, monkeypatch):
    """
    Test D: Verify the sequence is correct after one tool call is made,
    then followed by a normal response.
    """
    iteration = 0
    
    class MockCompletions:
        async def create(self, **kwargs):
            nonlocal iteration
            if iteration == 0:
                iteration += 1
                class MockMessage:
                    content = None
                    class MockToolCall:
                        id = "call_999"
                        class MockFunction:
                            name = "system_diagnostics"
                            arguments = "{}"
                        function = MockFunction()
                    tool_calls = [MockToolCall()]
                class MockChoice:
                    message = MockMessage()
                class MockCompletion:
                    choices = [MockChoice()]
                    usage = None
                return MockCompletion()
            else:
                class MockMessage:
                    content = "Final response"
                    tool_calls = None
                class MockChoice:
                    message = MockMessage()
                class MockCompletion:
                    choices = [MockChoice()]
                    usage = None
                return MockCompletion()

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    agent.client = MockClient()

    res_text, res_spoken = await agent.process_intent("Hello")
    assert res_text == "Final response"
    
    # System -> User -> Assistant(TC) -> Tool -> Assistant(Final)
    assert len(agent.conversation_history) == 5
    assert agent.conversation_history[1]["role"] == "user"
    assert agent.conversation_history[2]["role"] == "assistant"
    assert "tool_calls" in agent.conversation_history[2]
    assert agent.conversation_history[3]["role"] == "tool"
    assert agent.conversation_history[4]["role"] == "assistant"
    assert agent.conversation_history[4]["content"] == "Final response"
    
    # Ensure no duplicates
    count = sum(1 for msg in agent.conversation_history if msg.get("content") == "Final response")
    assert count == 1

@pytest.mark.asyncio
async def test_context_trimming_with_tools(agent, monkeypatch):
    """
    Test E: Verify that context trimming safely removes tool calls AND tool results
    together without leaving dangling orphaned tool results.
    """
    agent.governor.max_completion_tokens = 10
    # Artificially fill the context with tool call / tool result pairs
    agent.conversation_history = [{"role": "system", "content": "SYS"}]
    
    # 3 pairs of user + assistant(tools) + tool results
    for i in range(3):
        agent.conversation_history.append({"role": "user", "content": "U" * 2000}) # Big message
        agent.conversation_history.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "system_diagnostics", "arguments": "{}"}
                }
            ]
        })
        agent.conversation_history.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "name": "system_diagnostics",
            "content": "Tool result"
        })

    class MockMessage:
        content = "Final answer"
        tool_calls = None

    class MockChoice:
        message = MockMessage()

    class MockCompletion:
        choices = [MockChoice()]
        usage = None

    class MockCompletions:
        async def create(self, **kwargs):
            return MockCompletion()

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    agent.client = MockClient()

    # The next process_intent should trigger a trim
    res_text, _ = await agent.process_intent("U" * 100)
    assert res_text == "Final answer"
    
    # Verify no dangling tools
    # Ensure every tool message has a preceding assistant message with tool_calls
    for i, msg in enumerate(agent.conversation_history):
        if msg.get("role") == "tool":
            prev_msg = agent.conversation_history[i-1]
            assert prev_msg.get("role") == "assistant", "Dangling tool result without preceding assistant message!"
            assert "tool_calls" in prev_msg, "Assistant message preceding tool result has no tool_calls!"
