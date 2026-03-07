"""
LangChain adapter for CovenantAI.
"""
import asyncio
import time
from typing import Any, Dict, List
from .base import BaseAdapter
from covenant.models import AgentTrace, ToolCallTrace
from covenant.exceptions import CovenantRunError, CovenantTimeoutError

try:
    from langchain_core.callbacks import AsyncCallbackHandler # type: ignore
except ImportError:
    class AsyncCallbackHandler: # type: ignore
        pass

class CovenantAsyncCallbackHandler(AsyncCallbackHandler): # type: ignore[misc]
    """Callback handler used to intercept tool calls during a LangChain run."""
    def __init__(self) -> None:
        super().__init__()
        self.tool_calls: List[ToolCallTrace] = []
        self._current_starts: Dict[str, Any] = {}

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        run_id = str(kwargs.get("run_id"))
        name = str(serialized.get("name", "unknown"))
        
        args = {"raw": input_str}
        try:
            import json
            parsed = json.loads(input_str)
            if isinstance(parsed, dict):
                args = parsed
        except Exception:
            pass
            
        self._current_starts[run_id] = {
            "name": name,
            "args": args,
            "start_time": time.time() * 1000
        }

    async def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        run_id = str(kwargs.get("run_id"))
        if run_id in self._current_starts:
            start_info = self._current_starts.pop(run_id)
            self.tool_calls.append(
                ToolCallTrace(
                    tool_name=start_info["name"],
                    arguments=start_info["args"],
                    result=output,
                    timestamp_ms=start_info["start_time"]
                )
            )

    async def on_tool_error(self, error: Exception, **kwargs: Any) -> Any:
        run_id = str(kwargs.get("run_id"))
        if run_id in self._current_starts:
            start_info = self._current_starts.pop(run_id)
            self.tool_calls.append(
                ToolCallTrace(
                    tool_name=start_info["name"],
                    arguments=start_info["args"],
                    result=f"Error: {str(error)}",
                    timestamp_ms=start_info["start_time"]
                )
            )

class LangChainAdapter(BaseAdapter):
    """Adapter for LangChain runnable objects."""
    
    def __init__(self, agent: Any) -> None:
        """Initialize with a LangChain runnable."""
        self.agent = agent
        
    @classmethod
    def can_handle(cls, agent: Any) -> bool:
        agent_type_str = str(type(agent)).lower()
        return hasattr(agent, "ainvoke") and ("langchain" in agent_type_str or "runnable" in agent_type_str)

    async def run(self, user_input: str, timeout_seconds: int) -> AgentTrace:
        """Invoke the LangChain agent."""
        handler = CovenantAsyncCallbackHandler()
        start_time = time.time()
        
        try:
            raw_response = await asyncio.wait_for(
                self.agent.ainvoke({"input": user_input}, config={"callbacks": [handler]}),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError as e:
            raise CovenantTimeoutError(f"Agent exceeded timeout of {timeout_seconds}s") from e
        except Exception as e:
            raise CovenantRunError(f"Agent failed: {str(e)}", e) from e
            
        duration_ms = (time.time() - start_time) * 1000
        
        final_text = ""
        if isinstance(raw_response, dict):
            final_text = str(raw_response.get("output", raw_response.get("response", "")))
            if not final_text and len(raw_response) >= 1:
                final_text = str(list(raw_response.values())[0])
            
            # Extract from intermediate steps if no tools captured via callback
            if "intermediate_steps" in raw_response and not handler.tool_calls:
                for action, observation in raw_response["intermediate_steps"]:
                    tool_name = str(getattr(action, "tool", "unknown"))
                    tool_input = getattr(action, "tool_input", {})
                    if not isinstance(tool_input, dict):
                        tool_input = {"raw": str(tool_input)}
                    handler.tool_calls.append(
                        ToolCallTrace(
                            tool_name=tool_name,
                            arguments=tool_input,
                            result=str(observation),
                            timestamp_ms=time.time() * 1000
                        )
                    )
        elif isinstance(raw_response, str):
            final_text = raw_response
        else:
            final_text = str(raw_response)
            
        asked_for_confirmation = False
        confirmation_keywords = {"human_input", "ask_user", "confirm", "request_approval"}
        for tc in handler.tool_calls:
            if tc.tool_name.lower() in confirmation_keywords:
                asked_for_confirmation = True
                break
                
        return AgentTrace(
            tool_calls=handler.tool_calls,
            final_response=final_text,
            asked_for_confirmation=asked_for_confirmation,
            duration_ms=duration_ms
        )

__all__ = ["LangChainAdapter"]

