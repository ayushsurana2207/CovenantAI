"""
Assertion framework for CovenantAI testing.
"""
import re
from typing import Any, Callable, Dict
from covenant.models import (
    AgentTrace,
    AssertionResult,
    BaseAssertion,
    ToolCallsInclude,
    ToolCallsExclude,
    ToolCallsSequence,
    ResponseContains,
    ResponseNotContains,
    ResponseMatchesRegex,
    RequiresConfirmation,
    MaxToolCalls,
    ToolCallArgContains,
    ResponseWithinMs,
    ConversationFlowFollowed,
    NeverInterrupted,
    MultiTurnAssertion,
)

def _eval_tool_calls_include(assertion: ToolCallsInclude, trace: AgentTrace) -> AssertionResult:
    trace_tools = [tc.tool_name for tc in trace.tool_calls]
    missing = [t for t in assertion.tools if t not in trace_tools]
    
    passed = len(missing) == 0
    if passed:
        message = f"Found all expected tool calls: {assertion.tools}"
    else:
        message = f"Expected tool calls {assertion.tools} but only found {trace_tools}. Missing: {missing}"
        
    return AssertionResult(
        assertion_type=assertion.type,
        passed=passed,
        message=message,
        details={
            "assertion_type": assertion.type,
            "expected": assertion.tools,
            "actual": trace_tools
        }
    )

def _eval_tool_calls_exclude(assertion: ToolCallsExclude, trace: AgentTrace) -> AssertionResult:
    trace_tools = [tc.tool_name for tc in trace.tool_calls]
    unexpected = [t for t in assertion.tools if t in trace_tools]
    
    passed = len(unexpected) == 0
    if passed:
        message = f"None of the excluded tool calls {assertion.tools} were found."
    else:
        message = f"Unexpected tool calls found: {unexpected}"
        
    return AssertionResult(
        assertion_type=assertion.type,
        passed=passed,
        message=message,
        details={
            "assertion_type": assertion.type,
            "expected": assertion.tools,
            "actual": trace_tools
        }
    )

def _eval_tool_calls_sequence(assertion: ToolCallsSequence, trace: AgentTrace) -> AssertionResult:
    trace_tools = [tc.tool_name for tc in trace.tool_calls]
    expected = assertion.tools
    
    if not expected:
        return AssertionResult(
            assertion_type=assertion.type,
            passed=True,
            message="No tool calls expected in sequence.",
            details={
                "assertion_type": assertion.type,
                "expected": expected,
                "actual": trace_tools
            }
        )
        
    passed = False
    message = ""
    
    if assertion.strict:
        # Check strict contiguous sequence
        len_expected = len(expected)
        len_trace = len(trace_tools)
        for i in range(len_trace - len_expected + 1):
            if trace_tools[i:i + len_expected] == expected:
                passed = True
                message = f"Found strict tool call sequence: {expected}"
                break
        if not passed:
            message = f"Sequence {expected} not found contiguously. Actual list: {trace_tools}"
    else:
        # Check relative order (subsequence)
        curr_idx = 0
        for tool in trace_tools:
            if tool == expected[curr_idx]:
                curr_idx += 1
                if curr_idx == len(expected):
                    passed = True
                    break
        
        if passed:
            message = f"Found tool call sequence (non-strict): {expected}"
        else:
            missing_tool = expected[curr_idx]
            message = f"Sequence broke looking for '{missing_tool}'. Actual tools: {trace_tools}"
            
    return AssertionResult(
        assertion_type=assertion.type,
        passed=passed,
        message=message,
        details={
            "assertion_type": assertion.type,
            "expected": expected,
            "actual": trace_tools
        }
    )

def _eval_response_contains(assertion: ResponseContains, trace: AgentTrace) -> AssertionResult:
    resp = trace.final_response
    search_text = assertion.text
    
    if not assertion.case_sensitive:
        resp = resp.lower()
        search_text = search_text.lower()
        
    passed = search_text in resp
    snippet = trace.final_response[:200] + ("..." if len(trace.final_response) > 200 else "")
    
    if passed:
        message = f"Response contains expected text: '{assertion.text}'"
    else:
        message = f"Response did not contain expected text: '{assertion.text}'. Output snippet: '{snippet}'"
        
    return AssertionResult(
        assertion_type=assertion.type,
        passed=passed,
        message=message,
        details={
            "assertion_type": assertion.type,
            "expected": assertion.text,
            "actual": snippet
        }
    )

def _eval_response_not_contains(assertion: ResponseNotContains, trace: AgentTrace) -> AssertionResult:
    resp = trace.final_response
    search_text = assertion.text
    
    if not assertion.case_sensitive:
        resp = resp.lower()
        search_text = search_text.lower()
        
    passed = search_text not in resp
    snippet = trace.final_response[:200] + ("..." if len(trace.final_response) > 200 else "")
    
    if passed:
        message = f"Response successfully omitted text: '{assertion.text}'"
    else:
        message = f"Response contained forbidden text: '{assertion.text}'. Output snippet: '{snippet}'"
        
    return AssertionResult(
        assertion_type=assertion.type,
        passed=passed,
        message=message,
        details={
            "assertion_type": assertion.type,
            "expected": assertion.text,
            "actual": snippet
        }
    )

def _eval_response_matches_regex(assertion: ResponseMatchesRegex, trace: AgentTrace) -> AssertionResult:
    snippet = trace.final_response[:200] + ("..." if len(trace.final_response) > 200 else "")
    
    try:
        match = re.search(assertion.pattern, trace.final_response)
        passed = bool(match)
        if passed:
            message = f"Response matched pattern: {assertion.pattern}"
        else:
            message = f"Response did not match pattern: {assertion.pattern}. Snippet: '{snippet}'"
    except re.error as e:
        passed = False
        message = f"Invalid regular expression '{assertion.pattern}': {e}"
        
    return AssertionResult(
        assertion_type=assertion.type,
        passed=passed,
        message=message,
        details={
            "assertion_type": assertion.type,
            "expected": assertion.pattern,
            "actual": snippet
        }
    )

def _eval_requires_confirmation(assertion: RequiresConfirmation, trace: AgentTrace) -> AssertionResult:
    passed = trace.asked_for_confirmation == assertion.expected
    
    if passed:
        message = f"Agent {'asked' if assertion.expected else 'did not ask'} for confirmation as expected."
    else:
        if assertion.expected:
            message = "Agent did not ask for confirmation before acting"
        else:
            message = "Agent asked for confirmation but none was expected"
            
    return AssertionResult(
        assertion_type=assertion.type,
        passed=passed,
        message=message,
        details={
            "assertion_type": assertion.type,
            "expected": assertion.expected,
            "actual": trace.asked_for_confirmation
        }
    )

def _eval_max_tool_calls(assertion: MaxToolCalls, trace: AgentTrace) -> AssertionResult:
    actual_count = len(trace.tool_calls)
    passed = actual_count <= assertion.limit
    
    if passed:
        message = f"Agent made {actual_count} tool calls (limit {assertion.limit})"
    else:
        message = f"Agent made {actual_count} tool calls, limit is {assertion.limit}"
        
    return AssertionResult(
        assertion_type=assertion.type,
        passed=passed,
        message=message,
        details={
            "assertion_type": assertion.type,
            "expected": assertion.limit,
            "actual": actual_count
        }
    )

def _eval_tool_call_arg_contains(assertion: ToolCallArgContains, trace: AgentTrace) -> AssertionResult:
    matching_calls = [tc for tc in trace.tool_calls if tc.tool_name == assertion.tool]
    
    if not matching_calls:
        return AssertionResult(
            assertion_type=assertion.type,
            passed=False,
            message=f"Tool '{assertion.tool}' was never called. Cannot check arguments.",
            details={
                "assertion_type": assertion.type,
                "expected": f"Tool '{assertion.tool}' called with {assertion.arg} containing '{assertion.value}'",
                "actual": [tc.tool_name for tc in trace.tool_calls]
            }
        )
        
    passed = False
    actual_args_seen = []
    
    for tc in matching_calls:
        actual_args_seen.append(tc.arguments)
        if assertion.arg in tc.arguments:
            val_str = str(tc.arguments[assertion.arg])
            if assertion.value in val_str:
                passed = True
                break
                
    if passed:
        message = f"Tool '{assertion.tool}' called with argument '{assertion.arg}' containing '{assertion.value}'"
    else:
        message = f"Tool '{assertion.tool}' was called but '{assertion.arg}' did not contain '{assertion.value}'. Seen args: {actual_args_seen}"
        
    return AssertionResult(
        assertion_type=assertion.type,
        passed=passed,
        message=message,
        details={
            "assertion_type": assertion.type,
            "expected": f"Property '{assertion.arg}' contains '{assertion.value}'",
            "actual": actual_args_seen
        }
    )

def _eval_response_within_ms(assertion: ResponseWithinMs, trace: AgentTrace) -> AssertionResult:
    passed = trace.duration_ms <= assertion.max_ms
    if passed:
        message = f"Response took {trace.duration_ms:.1f}ms (passed <= {assertion.max_ms}ms)"
    else:
        message = f"Response took {trace.duration_ms:.1f}ms, expected under {assertion.max_ms}ms"
        
    return AssertionResult(
        assertion_type=assertion.type,
        passed=passed,
        message=message,
        details={
            "assertion_type": assertion.type,
            "expected": assertion.max_ms,
            "actual": trace.duration_ms
        }
    )

def _eval_conversation_flow_followed(assertion: ConversationFlowFollowed, trace: AgentTrace) -> AssertionResult:
    # ConversationFlowFollowed uses the exact same logic as ToolCallsSequence 
    # but mapped to Voice semantic fields instead of standard tools.
    # Convert assertion for the sequence evaluator
    seq_assert = ToolCallsSequence(tools=assertion.states, strict=assertion.strict)
    result = _eval_tool_calls_sequence(seq_assert, trace)
    result.assertion_type = assertion.type
    if not result.passed:
        result.message = f"Conversation did not follow expected flow. Got: {[tc.tool_name for tc in trace.tool_calls]} Expected: {assertion.states}"
    else:
        result.message = f"Conversation flow followed expected states: {assertion.states}"
    return result

def _eval_never_interrupted(assertion: NeverInterrupted, trace: AgentTrace) -> AssertionResult:
    passed = trace.interrupted_user == assertion.expected
    if passed:
        message = f"Agent {'interrupted' if trace.interrupted_user else 'did not interrupt'} user, as expected."
    else:
        if assertion.expected:
            message = "Agent was expected to interrupt user but did not"
        else:
            message = "Agent interrupted user mid-speech unexpectedly"
            
    return AssertionResult(
        assertion_type=assertion.type,
        passed=passed,
        message=message,
        details={
            "assertion_type": assertion.type,
            "expected": assertion.expected,
            "actual": trace.interrupted_user
        }
    )

def _eval_multi_turn(assertion: MultiTurnAssertion, trace: AgentTrace) -> AssertionResult:
    """
    MultiTurnAssertion evaluation is handled orchestrally by the ScenarioRunner over sequence state.
    This evaluator just passively clears it during the standard scalar assert_ pass.
    """
    return AssertionResult(
        assertion_type=assertion.type,
        passed=True,
        message="Multi-turn assertion handled by Runner",
        details={
            "assertion_type": assertion.type,
            "expected": "multi_turn_run",
            "actual": "handled_by_runner"
        }
    )

# O(1) dispatch dictionary mapping assertion type strings to evaluator functions
EVALUATORS: Dict[str, Callable[[Any, AgentTrace], AssertionResult]] = {
    "tool_calls_include": _eval_tool_calls_include,
    "tool_calls_exclude": _eval_tool_calls_exclude,
    "tool_calls_sequence": _eval_tool_calls_sequence,
    "response_contains": _eval_response_contains,
    "response_not_contains": _eval_response_not_contains,
    "response_matches_regex": _eval_response_matches_regex,
    "requires_confirmation": _eval_requires_confirmation,
    "max_tool_calls": _eval_max_tool_calls,
    "tool_call_arg_contains": _eval_tool_call_arg_contains,
    "response_within_ms": _eval_response_within_ms,
    "conversation_flow_followed": _eval_conversation_flow_followed,
    "never_interrupted": _eval_never_interrupted,
    "multi_turn": _eval_multi_turn,
}

def evaluate(assertion: Any, trace: AgentTrace) -> AssertionResult:
    """Evaluate a single assertion against an agent trace."""
    
    if not hasattr(assertion, "type"):
        return AssertionResult(
            assertion_type="unknown",
            passed=False,
            message="Assertion object is missing a 'type' attribute",
            details={"assertion_type": "unknown", "expected": None, "actual": str(type(assertion))}
        )
        
    assertion_type = assertion.type
    eval_func = EVALUATORS.get(assertion_type)
    
    if not eval_func:
        return AssertionResult(
            assertion_type=assertion_type,
            passed=False,
            message=f"Unknown assertion type: {assertion_type}",
            details={"assertion_type": assertion_type, "expected": None, "actual": None}
        )
        
    try:
        return eval_func(assertion, trace)
    except Exception as e:
        return AssertionResult(
            assertion_type=assertion_type,
            passed=False,
            message=f"Assertion logic crashed: {str(e)}",
            details={
                "assertion_type": assertion_type, 
                "expected": None, 
                "actual": f"Exception {type(e).__name__}"
            }
        )

__all__ = ["evaluate"]

