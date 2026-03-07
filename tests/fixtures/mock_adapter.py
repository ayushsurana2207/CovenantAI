"""
Mock adapter for CovenantAI testing.
"""
import asyncio
from typing import Any, List
from covenant.adapters.base import BaseAdapter
from covenant.models import AgentTrace
from covenant.exceptions import CovenantRunError, CovenantTimeoutError

class MockAdapter(BaseAdapter):
    """A mock adapter that replays pre-scripted AgentTrace instances."""
    
    def __init__(self, agent: Any = None) -> None:
        self.agent = agent
        self.traces: List[AgentTrace] = []
        self.exceptions_to_raise: List[Exception] = []
        self.timeout_on_calls: bool = False
        
    @classmethod
    def can_handle(cls, agent: Any) -> bool:
        return isinstance(agent, str) and agent == "mock"
        
    async def run(self, user_input: str, timeout_seconds: int) -> AgentTrace:
        if self.timeout_on_calls:
            await asyncio.sleep(timeout_seconds + 0.1)
            raise CovenantTimeoutError("Mock timeout exceeded")
            
        if self.exceptions_to_raise:
            e = self.exceptions_to_raise.pop(0)
            raise CovenantRunError(f"Mock error: {str(e)}", original_exception=e)
            
        if self.traces:
            return self.traces.pop(0)
            
        return AgentTrace(final_response="Mock fallback response")
