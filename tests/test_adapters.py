"""
Tests for adapters and the adapter resolver.
"""
import pytest
import asyncio
from covenant.adapters import get_adapter
from covenant.adapters.base import BaseAdapter
from covenant.models import AgentTrace, ToolCallTrace
from covenant.exceptions import CovenantRunError, CovenantTimeoutError, AdapterNotFoundError
from tests.fixtures.mock_adapter import MockAdapter

class DummyLangChainAgent:
    """A minimal mock for LangChain."""
    async def ainvoke(self, *args, **kwargs):
        pass

class DummyOpenAIAgent:
    """A minimal mock for OpenAI Agents."""
    __module__ = "openai_agents"
    def run(self, *args, **kwargs):
        pass

def test_get_adapter_resolves_langchain():
    agent = DummyLangChainAgent()
    adapter = get_adapter(agent)
    assert type(adapter).__name__ == "LangChainAdapter"
    
def test_get_adapter_resolves_openai():
    agent = DummyOpenAIAgent()
    adapter = get_adapter(agent)
    assert type(adapter).__name__ == "OpenAIAgentsAdapter"
    
def test_get_adapter_raises_not_found():
    with pytest.raises(AdapterNotFoundError, match="Supported frameworks"):
        get_adapter("Not A Valid Agent")

@pytest.mark.asyncio
async def test_mock_adapter_returns_trace():
    adapter = MockAdapter("mock")
    trace = AgentTrace(
        final_response="Success",
        tool_calls=[ToolCallTrace(tool_name="test", arguments={}, result="ok", timestamp_ms=0.0)]
    )
    adapter.traces.append(trace)
    
    result = await adapter.run("hello", timeout_seconds=5)
    assert result.final_response == "Success"
    assert len(result.tool_calls) == 1
    
@pytest.mark.asyncio
async def test_mock_adapter_raises_timeout():
    adapter = MockAdapter("mock")
    adapter.timeout_on_calls = True
    
    with pytest.raises(CovenantTimeoutError, match="Mock timeout exceeded"):
        # Use a very short timeout so the test runs quickly
        await adapter.run("hello", timeout_seconds=0.01)

@pytest.mark.asyncio
async def test_mock_adapter_raises_run_error():
    adapter = MockAdapter("mock")
    adapter.exceptions_to_raise.append(ValueError("Crash!"))
    
    with pytest.raises(CovenantRunError) as exc_info:
        await adapter.run("hello", timeout_seconds=5)
        
    assert "Crash!" in str(exc_info.value)
    assert isinstance(exc_info.value.original_exception, ValueError)
