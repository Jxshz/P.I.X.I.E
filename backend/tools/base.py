from abc import ABC, abstractmethod
from typing import Any, Dict
from backend.tools.permissions import PermissionLevel

class BaseTool(ABC):
    """
    Abstract base class for all P.I.X.I.E. tools.
    Enforces a strict interface to ensure safety and predictability.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The stable identifier for the tool (e.g., 'system_diagnostics')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A detailed description of what the tool does, used by the LLM."""
        pass

    @property
    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """JSON Schema definition of the tool's input parameters."""
        pass

    @property
    @abstractmethod
    def permission(self) -> PermissionLevel:
        """The permission level required to execute this tool."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        The isolated execution logic.
        Must raise exceptions on failure, which will be caught by the registry.
        """
        pass
