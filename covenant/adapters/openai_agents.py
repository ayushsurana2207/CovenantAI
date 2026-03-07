"""
OpenAI Agents SDK adapter for CovenantAI.
"""
import asyncio
import time
from typing import Any
from .base import BaseAdapter
from covenant.models import AgentTrace, ToolCallTrace
from covenant.exceptions import CovenantRunError, CovenantTimeoutError

class OpenAIAgentsAdapter(BaseAdapter):
    """Adapter for OpenAI Agents SDK actors."""
    
    def __init__(self, agent: Any) -> None:
        """Initialize with an OpenAI SDK agent."""
        self.agent = agent
        
    @classmethod
    def can_handle(cls, agent: Any) -> bool:
        agent_type_str = str(type(agent)).lower()
        agent_module_str = getattr(type(agent), "__module__", "").lower()
        has_run = hasattr(agent, "run") or hasattr(agent, "run_sync")
        return has_run and ("openai" in agent_type_str or "openai" in agent_module_str) and "agent" in agent_type_str

    async def _invoke_runner(self, user_input: str) -> Any:
        try:
            from openai_agents import Runner  # type: ignore
            return await Runner.run(self.agent, user_input)
        except ImportError:
            # Fallback for testing/duck-typing
            if hasattr(self.agent, "run"):
                res = self.agent.run(user_input)
                if asyncio.iscoroutine(res):
                    return await res
                return res
            raise
            
    async def run(self, user_input: str, timeout_seconds: int) -> AgentTrace:
        """Invoke the OpenAI agent."""
        start_time = time.time()
        
        try:
            raw_result = await asyncio.wait_for(
                self._invoke_runner(user_input),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError as e:
            raise CovenantTimeoutError(f"Agent exceeded timeout of {timeout_seconds}s") from e
        except Exception as e:
            raise CovenantRunError(f"Agent failed: {str(e)}", e) from e
            
        duration_ms = (time.time() - start_time) * 1000
        
        tool_calls = []
        final_text = ""
        asked_for_confirmation = False
        
        if hasattr(raw_result, "new_items"):
            pending_tools = {}
            for item in raw_result.new_items:
                item_type = type(item).__name__
                if "ToolCallItem" in item_type:
                    name = getattr(item, "name", "unknown")
                    args = getattr(item, "arguments", {})
                    if not isinstance(args, dict):
                        try:
                            import json
                            parsed = json.loads(args)
                            args = parsed if isinstance(parsed, dict) else {"raw": str(args)}
                        except Exception:
                            args = {"raw": str(args)}
                            
                    call_id = getattr(item, "call_id", getattr(item, "id", None))
                    pending_tools[call_id] = {
                        "name": name,
                        "args": args,
                        "ts": time.time() * 1000
                    }
                elif "ToolCallOutputItem" in item_type:
                    call_id = getattr(item, "call_id", getattr(item, "tool_call_id", None))
                    output = str(getattr(item, "output", getattr(item, "content", "")))
                    if call_id in pending_tools:
                        ti = pending_tools.pop(call_id)
                        tool_calls.append(ToolCallTrace(
                            tool_name=str(ti["name"]),
                            arguments=ti["args"] if isinstance(ti["args"], dict) else {"raw": str(ti["args"])},
                            result=output,
                            timestamp_ms=float(str(ti["ts"]))
                        ))
                        if str(ti["name"]).lower() in {"confirm", "approve"}:
                            asked_for_confirmation = True
                elif "MessageOutputItem" in item_type:
                    final_text = str(getattr(item, "content", getattr(item, "text", "")))
            
            for ti in pending_tools.values():
                tool_calls.append(ToolCallTrace(
                    tool_name=str(ti["name"]),
                    arguments=ti["args"] if isinstance(ti["args"], dict) else {"raw": str(ti["args"])},
                    result="",
                    timestamp_ms=float(str(ti["ts"]))
                ))
                if str(ti["name"]).lower() in {"confirm", "approve"}:
                    asked_for_confirmation = True
        else:
            final_text = str(raw_result)
            
        return AgentTrace(
            tool_calls=tool_calls,
            final_response=final_text,
            asked_for_confirmation=asked_for_confirmation,
            duration_ms=duration_ms
        )

__all__ = ["OpenAIAgentsAdapter"]

