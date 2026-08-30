import time
import platform
from typing import Dict, Any
from backend.tools.base import BaseTool
from backend.tools.permissions import PermissionLevel

class SystemDiagnosticsTool(BaseTool):
    """
    A minimal, deterministic tool to validate the Phase 3 tool execution pipeline.
    It reads safe system properties and does not modify state.
    """

    @property
    def name(self) -> str:
        return "system_diagnostics"

    @property
    def description(self) -> str:
        return "Retrieves safe system diagnostics such as the current timestamp and platform information."

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }

    @property
    def permission(self) -> PermissionLevel:
        return PermissionLevel.SAFE

    def execute(self, **kwargs) -> Any:
        # Deliberately returning a static structure with basic safe OS info.
        return {
            "timestamp": time.time(),
            "platform": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "status": "operational"
        }
