"""
Exceptions for CovenantAI.
"""

from typing import Optional

class CovenantError(Exception):
    """Base exception for all CovenantAI errors."""
    pass

class CovenantRunError(CovenantError):
    """Raised when an agent run fails due to an exception inside the agent."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original_exception = original_exception

class CovenantTimeoutError(CovenantError):
    """Raised when an agent run exceeds the specified timeout."""
    pass

class AdapterNotFoundError(CovenantError):
    """Raised when no suitable adapter can handle the provided agent."""
    pass

class CovenantImportError(CovenantError):
    """Raised when the agent specified by a dotted path cannot be imported."""
    pass

__all__ = [
    "CovenantError",
    "CovenantRunError",
    "CovenantTimeoutError",
    "AdapterNotFoundError",
    "CovenantImportError",
]
