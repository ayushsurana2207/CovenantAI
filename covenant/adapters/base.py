"""
Base adapter interface.
"""
from abc import ABC, abstractmethod
from typing import Any
from covenant.models import AgentTrace

class BaseAdapter(ABC):
    """Wraps an agent callable into a testable interface that returns AgentTrace."""
    
    @abstractmethod
    async def run(self, user_input: str, timeout_seconds: int) -> AgentTrace:
        """Run the agent with the given input and return a structured trace."""
        ...

    async def run_multi_turn(self, turns: list[str], timeout_seconds: int) -> list[AgentTrace]:
        """
        Run the agent across multiple turns in a sequential context.
        By default, evaluates independently by calling run(). State-preserving
        adapters should override this to maintain session contexts.
        """
        results = []
        for turn in turns:
            results.append(await self.run(turn, timeout_seconds))
        return results

    @classmethod
    @abstractmethod
    def can_handle(cls, agent: Any) -> bool:
        """Return True if this adapter can handle the given agent object."""
        ...

__all__ = ["BaseAdapter"]

