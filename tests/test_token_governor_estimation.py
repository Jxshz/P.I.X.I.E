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
    # (40 / 4.0) * 1.5 = 15
    estimate = gov.estimate_tokens([msg])
    assert estimate == 15

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
