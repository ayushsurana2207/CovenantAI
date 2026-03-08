from covenant.assertions import evaluate
from covenant.models import (
    AgentTrace,
    ToolCallTrace,
    ToolCallsInclude,
    ToolCallsExclude,
    ToolCallsSequence,
    ResponseContains,
    ResponseNotContains,
    ResponseMatchesRegex,
    RequiresConfirmation,
    MaxToolCalls,
    ToolCallArgContains,
)

# Helper for creating traces easily
def make_trace(tools=None, response="Hello world", confirmed=False):
    tool_calls = []
    if tools:
        for t in tools:
            args = {}
            if isinstance(t, tuple):
                t, args = t
            tool_calls.append(ToolCallTrace(tool_name=t, arguments=args, result="ok", timestamp_ms=0))
    return AgentTrace(
        tool_calls=tool_calls,
        final_response=response,
        asked_for_confirmation=confirmed,
        duration_ms=100
    )


def test_tool_calls_include():
    trace = make_trace(["search", "calculator", "weather"])
    
    # Pass
    res = evaluate(ToolCallsInclude(tools=["search", "weather"]), trace)
    assert res.passed
    
    # Fail
    res = evaluate(ToolCallsInclude(tools=["search", "translate"]), trace)
    assert not res.passed
    assert "translate" in res.message
    
    # Edge case - empty trace but tools expected
    res = evaluate(ToolCallsInclude(tools=["search"]), make_trace([]))
    assert not res.passed
    assert "Expected tool calls ['search'] but only found []" in res.message


def test_tool_calls_exclude():
    trace = make_trace(["search", "calculator"])
    
    # Pass
    res = evaluate(ToolCallsExclude(tools=["weather", "translate"]), trace)
    assert res.passed
    
    # Fail
    res = evaluate(ToolCallsExclude(tools=["search"]), trace)
    assert not res.passed
    assert "Unexpected tool calls found: ['search']" in res.message


def test_tool_calls_sequence():
    trace = make_trace(["auth", "search", "parse", "format", "send"])
    
    # Pass Subsequence
    res = evaluate(ToolCallsSequence(tools=["auth", "format", "send"], strict=False), trace)
    assert res.passed
    
    # Fail Subsequence
    res = evaluate(ToolCallsSequence(tools=["auth", "send", "format"], strict=False), trace)
    assert not res.passed
    assert "Sequence broke looking for 'format'" in res.message
    
    # Pass Strict Contiguous
    res = evaluate(ToolCallsSequence(tools=["search", "parse", "format"], strict=True), trace)
    assert res.passed
    
    # Fail Strict Contiguous
    res = evaluate(ToolCallsSequence(tools=["search", "format"], strict=True), trace)
    assert not res.passed
    
    # Edge Empty Sequence
    res = evaluate(ToolCallsSequence(tools=[]), trace)
    assert res.passed


def test_response_contains():
    trace = make_trace([], response="The current temperature is 72F exactly.")
    
    # Pass
    res = evaluate(ResponseContains(text="temperature is 72F"), trace)
    assert res.passed
    
    # Pass Case Insensitive
    res = evaluate(ResponseContains(text="TEMPERATURE", case_sensitive=False), trace)
    assert res.passed
    
    # Fail Case Sensitive
    res = evaluate(ResponseContains(text="TEMPERATURE", case_sensitive=True), trace)
    assert not res.passed
    
    # Fail not found
    res = evaluate(ResponseContains(text="celsius"), trace)
    assert not res.passed


def test_response_not_contains():
    trace = make_trace([], response="All systems normal.")
    
    # Pass
    res = evaluate(ResponseNotContains(text="error"), trace)
    assert res.passed
    
    # Pass Case Insensitive
    res = evaluate(ResponseNotContains(text="ERROR", case_sensitive=False), trace)
    assert res.passed
    
    # Fail Case Insensitive
    res = evaluate(ResponseNotContains(text="NORMAL", case_sensitive=False), trace)
    assert not res.passed
    
    # Fail
    res = evaluate(ResponseNotContains(text="systems"), trace)
    assert not res.passed


def test_response_matches_regex():
    trace = make_trace([], response="My email is bot@example.com.")
    
    # Pass
    res = evaluate(ResponseMatchesRegex(pattern=r"\w+@\w+\.\w+"), trace)
    assert res.passed
    
    # Fail pattern
    res = evaluate(ResponseMatchesRegex(pattern=r"^\d+$"), trace)
    assert not res.passed
    
    # Edge invalid regex
    res = evaluate(ResponseMatchesRegex(pattern=r"[unterminated"), trace)
    assert not res.passed
    assert "Invalid regular expression" in res.message


def test_requires_confirmation():
    # True / True
    trace = make_trace([], confirmed=True)
    res = evaluate(RequiresConfirmation(expected=True), trace)
    assert res.passed
    
    # False / False
    trace2 = make_trace([], confirmed=False)
    res = evaluate(RequiresConfirmation(expected=False), trace2)
    assert res.passed
    
    # False / True
    res = evaluate(RequiresConfirmation(expected=False), trace)
    assert not res.passed
    assert "Agent asked for confirmation but none was expected" in res.message
    
    # True / False
    res = evaluate(RequiresConfirmation(expected=True), trace2)
    assert not res.passed
    assert "Agent did not ask for confirmation before acting" in res.message


def test_max_tool_calls():
    trace = make_trace(["t1", "t2", "t3"])
    
    # Pass exact
    res = evaluate(MaxToolCalls(limit=3), trace)
    assert res.passed
    
    # Pass under limit
    res = evaluate(MaxToolCalls(limit=5), trace)
    assert res.passed
    
    # Fail over limit
    res = evaluate(MaxToolCalls(limit=2), trace)
    assert not res.passed

def test_tool_call_arg_contains():
    trace = make_trace([
        ("search", {"query": "weather in NYC"}),
        ("search", {"query": "population of USA"})
    ])
    
    # Pass match any
    res = evaluate(ToolCallArgContains(tool="search", arg="query", value="USA"), trace)
    assert res.passed
    
    # Fail arg not matched
    res = evaluate(ToolCallArgContains(tool="search", arg="query", value="Tokyo"), trace)
    assert not res.passed
    assert "did not contain 'Tokyo'" in res.message
    
    # Fail tool not called
    res = evaluate(ToolCallArgContains(tool="weather", arg="city", value="NYC"), trace)
    assert not res.passed
    assert "was never called. Cannot check arguments." in res.message

def test_evaluate_engine_safety():
    trace = make_trace()
    
    # Object missing type attribute
    class BadObj:
        pass
    res = evaluate(BadObj(), trace)
    assert not res.passed
    assert "missing a 'type' attribute" in res.message
    
    # Unknown type string
    class UnknownObj:
        type = "does_not_exist"
    res = evaluate(UnknownObj(), trace)
    assert not res.passed
    assert "Unknown assertion type" in res.message
    
    # Function crash testing
    # ToolCallArgContains expects strict argument attributes, passing a broken type triggers exception
    class CrashObj:
        type = "tool_call_arg_contains"
        # Missing required model attributes like 'tool'
    
    res = evaluate(CrashObj(), trace)
    assert not res.passed
    assert "Assertion logic crashed" in res.message
