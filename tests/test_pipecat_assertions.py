from covenant.models import (
    AgentTrace, ToolCallTrace, ResponseWithinMs, ConversationFlowFollowed, NeverInterrupted
)
from covenant.assertions import evaluate

def test_response_within_ms_passes():
    trace = AgentTrace(final_response="Hello", duration_ms=2500.0)
    assertion = ResponseWithinMs(max_ms=5000)
    result = evaluate(assertion, trace)
    assert result.passed is True
    assert "passed" in result.message

def test_response_within_ms_fails():
    trace = AgentTrace(final_response="Hello", duration_ms=6000.0)
    assertion = ResponseWithinMs(max_ms=5000)
    result = evaluate(assertion, trace)
    assert result.passed is False
    assert "expected under 5000ms" in result.message

def test_never_interrupted_passes_when_not_interrupted_and_expected_false():
    trace = AgentTrace(final_response="Hello", interrupted_user=False)
    assertion = NeverInterrupted(expected=False)
    result = evaluate(assertion, trace)
    assert result.passed is True
    assert "did not interrupt" in result.message

def test_never_interrupted_fails_when_interrupted_and_expected_false():
    trace = AgentTrace(final_response="Hello", interrupted_user=True)
    assertion = NeverInterrupted(expected=False)
    result = evaluate(assertion, trace)
    assert result.passed is False
    assert "interrupted user mid-speech unexpectedly" in result.message

def test_never_interrupted_passes_when_interrupted_and_expected_true():
    trace = AgentTrace(final_response="Hello", interrupted_user=True)
    assertion = NeverInterrupted(expected=True)
    result = evaluate(assertion, trace)
    assert result.passed is True
    assert "interrupted" in result.message
    
def test_conversation_flow_followed():
    trace = AgentTrace(
        final_response="Done",
        tool_calls=[
            ToolCallTrace(tool_name="get_weather", arguments={}, result="Sunny", timestamp_ms=0),
            ToolCallTrace(tool_name="confirm_action", arguments={}, result="Yes", timestamp_ms=0)
        ]
    )
    assertion = ConversationFlowFollowed(states=["get_weather", "confirm_action"])
    result = evaluate(assertion, trace)
    assert result.passed is True
    assert "flow followed expected states" in result.message
    
    assertion_strict = ConversationFlowFollowed(states=["confirm_action", "get_weather"], strict=True)
    result_strict = evaluate(assertion_strict, trace)
    assert result_strict.passed is False
    assert "did not follow expected flow" in result_strict.message
