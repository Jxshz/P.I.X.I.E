import os
import json
from typing import Dict, Any, List
from backend.tools.base import BaseTool
from backend.tools.permissions import PermissionLevel

class ToolExecutionError(Exception):
    """Raised when a tool fails to execute safely."""
    pass

class ConfirmationRequiredException(Exception):
    """Raised when a tool requires explicit user confirmation before executing."""
    def __init__(self, tool_call_id: str, tool_name: str, arguments_json: str):
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.arguments_json = arguments_json
        super().__init__(f"Confirmation required for tool '{tool_name}'")

class ToolRegistry:
    """
    Manages registration, validation, and safe execution of tools.
    Enforces the permission boundary.
    """
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        # We read from environment variables to determine if confirmation is required
        self.require_confirmation = os.getenv("REQUIRE_CONFIRMATION", "true").lower() == "true"
        
    def register(self, tool: BaseTool) -> None:
        """Register a new tool."""
        if not isinstance(tool, BaseTool):
            raise TypeError("Tool must inherit from BaseTool.")
        if tool.name in self._tools:
            raise ValueError(f"Registration failed: Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool
        
    def get_tool(self, name: str) -> BaseTool:
        """Retrieve a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._tools[name]
        
    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """Returns the OpenAI-compatible tool schemas for the LLM."""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema
                }
            })
        return schemas
        
    def execute_tool(self, name: str, arguments_json: str, tool_call_id: str = "test_call", is_confirmed: bool = False) -> str:
        """
        Safely executes a tool by name.
        Catches exceptions and validates permissions.
        Returns the result as a string suitable for passing back to the LLM.
        """
        try:
            tool = self.get_tool(name)
        except KeyError:
            return json.dumps({"error": f"Tool '{name}' not found or not permitted."})
            
        # 1. Permission Check
        if tool.permission == PermissionLevel.SAFE:
            pass
        elif tool.permission == PermissionLevel.CONFIRM_REQUIRED:
            if self.require_confirmation and not is_confirmed:
                raise ConfirmationRequiredException(tool_call_id, name, arguments_json)
        elif tool.permission == PermissionLevel.PRIVILEGED:
            return json.dumps({"error": "Execution denied: Missing privileged access."})
        elif tool.permission == PermissionLevel.PROHIBITED:
            return json.dumps({"error": "Execution denied: This tool is currently prohibited."})
        else:
            return json.dumps({"error": "Execution denied: Unknown or invalid permission level."})
            
        # 2. Parse arguments
        try:
            kwargs = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON arguments provided."})
            
        # 3. Execution Wrapper
        try:
            result = tool.execute(**kwargs)
            return json.dumps({"result": result})
        except Exception as e:
            # Safe failure: catch everything and return as an error string
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})
