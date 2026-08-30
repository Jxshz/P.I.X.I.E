import pytest
from backend.agent.token_governor import TokenGovernor

def test_tool_call_payload_counted():
    gov = TokenGovernor()
    
    # Message with tool calls
    msg = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "system_diagnostics",
                    "arguments": "{\"verbose\": true, \"extra\": \"data data data data data data\"}"
                }
            }
        ]
    }
    
    # Length of stringified tool_calls is > 100 characters
    # 100 / 4 * 1.5 = 37.5 (approx)
    estimate = gov.estimate_tokens([msg])
    assert estimate > 10  # Ensure it is significantly larger than the minimum floor of 1

def test_empty_tool_calls():
    gov = TokenGovernor()
    msg = {
        "role": "assistant",
        "tool_calls": []
    }
    # Empty list stringified is "[]" (2 characters)
    # (2 / 4) * 1.5 = 0.75 -> int = 0 -> floored to 1
    estimate = gov.estimate_tokens([msg])
    assert estimate == 1

def test_normal_messages_remain_correct():
    gov = TokenGovernor()
    # 40 characters
    msg = {
        "role": "user",
        "content": "A" * 40
    }
    # 40 / 3.8 = 10
    estimate = gov.estimate_tokens([msg])
    assert estimate == 10

def test_missing_none_content_with_tool_calls():
    gov = TokenGovernor()
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "a", "type": "function", "function": {"name": "test", "arguments": "A"*100}}]
    }
    estimate = gov.estimate_tokens([msg])
    assert estimate > 20

def test_multiple_tool_calls():
    gov = TokenGovernor()
    tc = {"id": "a", "type": "function", "function": {"name": "test", "arguments": "A"*100}}
    msg1 = {
        "role": "assistant",
        "tool_calls": [tc]
    }
    msg2 = {
        "role": "assistant",
        "tool_calls": [tc, tc]
    }
    est1 = gov.estimate_tokens([msg1])
    est2 = gov.estimate_tokens([msg2])
    
    assert est2 > est1

def test_regression_zero_estimate_with_none_content():
    # Previous implementation failed: if content was None or missing, it only counted 0 characters 
    # resulting in the floor of 1. We must explicitly verify it is NOT 1.
    gov = TokenGovernor()
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"test": "data" * 50}]
    }
    estimate = gov.estimate_tokens([msg])
    assert estimate > 1


def test_multiturn_conversation_does_not_falsely_exhaust_governor():
    """
    Regression Test for Phase 4.5 Stability Rectification:
    Verifies that a 10-turn sequential conversation within a single minute window
    does NOT trigger false-positive short-term processing exhaustion.
    """
    from backend.agent.core import AgentCore
    agent = AgentCore(db_path=":memory:")

    turns = [
        ("Explain planning purposes", "Planning is defining goals and strategies to coordinate activities."),
        ("Explain it simply", "Planning is deciding what to do before doing it."),
        ("Give me an example", "An architect creating blueprints before building."),
        ("Why is planning important?", "It reduces uncertainty, minimizes waste, and sets standards."),
        ("Compare planning and organising", "Planning sets goals; organising assigns resources."),
        ("Explain in one sentence", "Planning is mapping the route to your destination."),
        ("Give another example", "A student preparing a study timetable for finals."),
        ("Explain for an exam", "Planning is the primary management function upon which others depend."),
        ("Make it simpler", "Choose where you want to go before you drive."),
        ("Summarise", "Planning aligns teams, clarifies targets, and improves execution.")
    ]

    for i, (prompt, response) in enumerate(turns, 1):
        agent.conversation_history.append({"role": "user", "content": prompt})
        agent._trim_context()

        is_allowed, err_msg, res = agent.governor.preflight(agent.conversation_history)
        assert is_allowed is True, f"Turn {i} ('{prompt}') falsely exhausted governor: {err_msg}"
        assert res is not None

        # Simulate Groq completion usage
        actual_prompt = int(sum(len(m.get("content", "")) for m in agent.conversation_history) / 4)
        actual_comp = int(len(response) / 4)

        class MockUsage:
            total_tokens = actual_prompt + actual_comp
            prompt_tokens = actual_prompt
            completion_tokens = actual_comp

        agent.governor.record_usage(res, MockUsage())
        agent.conversation_history.append({"role": "assistant", "content": response})

    # Verify total usage across all 10 turns is well within 8000 TPM limit
    req_m, tok_m, _, _ = agent.governor.get_status()["requests_minute"], agent.governor.get_status()["tokens_minute"], None, None
    assert req_m == 10
    assert tok_m < 8000


def test_20_turn_context_trimming_preserves_bounded_tokens():
    """
    Verifies that context trimming prevents unbounded token growth across 20 turns.
    """
    from backend.agent.core import AgentCore
    agent = AgentCore(db_path=":memory:")

    for i in range(20):
        agent.conversation_history.append({"role": "user", "content": f"Question {i}: " + "test " * 20})
        agent._trim_context()
        agent.conversation_history.append({"role": "assistant", "content": f"Answer {i}: " + "response " * 30})

    # System message must still be preserved at index 0
    assert agent.conversation_history[0]["role"] == "system"
    # Token count of trimmed history must not exceed max context limit
    assert agent.governor.estimate_tokens(agent.conversation_history) <= 3000


def test_reservation_invariant_no_leaked_reservations():
    """
    Verifies that preflight reservations are cleanly reconciled without leaking inflated tokens.
    """
    gov = TokenGovernor()
    messages = [{"role": "user", "content": "Hello world"}]

    # Preflight creates 1 reservation
    allowed, _, res = gov.preflight(messages)
    assert allowed is True
    assert len(gov._minute_window) == 1
    reserved_tokens = gov._minute_window[0].tokens

    # Failed request drops reservation completely
    gov.record_usage(res, failed=True)
    assert len(gov._minute_window) == 0

    # Successful request reconciles to actual tokens
    allowed, _, res2 = gov.preflight(messages)
    assert len(gov._minute_window) == 1

    class MockUsage:
        total_tokens = 50

    gov.record_usage(res2, MockUsage())
    assert len(gov._minute_window) == 1
    assert gov._minute_window[0].tokens == 50
    assert gov._minute_window[0].tokens < reserved_tokens


def test_legitimate_exhaustion_still_rejected():
    """
    Verifies that genuinely excessive requests exceeding limits are still rejected.
    """
    gov = TokenGovernor()
    # Create massive message exceeding TPM limit (8000 tokens * 3.8 = 30400 chars)
    massive_msg = [{"role": "user", "content": "X" * 35000}]

    allowed, err_msg, res = gov.preflight(massive_msg)
    assert allowed is False
    assert res is None
    assert "short-term processing limit" in err_msg
