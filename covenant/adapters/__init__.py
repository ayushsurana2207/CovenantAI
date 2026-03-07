"""
Agent adapters for CovenantAI.
"""
from typing import Any
from .base import BaseAdapter
from .langchain import LangChainAdapter
from .openai_agents import OpenAIAgentsAdapter
from .pipecat import PipecatAdapter
from covenant.exceptions import AdapterNotFoundError

def get_adapter(agent: Any) -> BaseAdapter:
    """Resolve the appropriate adapter for a given agent object."""
    if LangChainAdapter.can_handle(agent):
        return LangChainAdapter(agent)
    if OpenAIAgentsAdapter.can_handle(agent):
        return OpenAIAgentsAdapter(agent)
    if PipecatAdapter.can_handle(agent):
        return PipecatAdapter(agent)
        
    raise AdapterNotFoundError(
        "Could not find a suitable adapter for the provided agent. "
        "Supported frameworks: LangChain (Runnable, AgentExecutor), "
        "OpenAI Agents SDK (Runner compatible instances)."
    )

__all__ = ["BaseAdapter", "LangChainAdapter", "OpenAIAgentsAdapter", "get_adapter"]

