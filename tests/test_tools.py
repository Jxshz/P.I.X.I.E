import json
import pytest
from backend.tools.permissions import PermissionLevel
from backend.tools.base import BaseTool
from backend.tools.registry import ToolRegistry, ToolExecutionError
from backend.tools.system_diagnostics import SystemDiagnosticsTool

class DummySafeTool(BaseTool):
    @property
    def name(self) -> str: return "dummy_safe"
    @property
    def description(self) -> str: return "A safe dummy tool"
    @property
    def schema(self) -> dict: return {}
    @property
    def permission(self) -> PermissionLevel: return PermissionLevel.SAFE
    def execute(self, **kwargs):
        return "success"

class DummyConfirmTool(BaseTool):
    @property
    def name(self) -> str: return "dummy_confirm"
    @property
    def description(self) -> str: return "Requires confirm"
    @property
    def schema(self) -> dict: return {}
    @property
    def permission(self) -> PermissionLevel: return PermissionLevel.CONFIRM_REQUIRED
    def execute(self, **kwargs):
        return "confirmed_success"

class DummyProhibitedTool(BaseTool):
    @property
    def name(self) -> str: return "dummy_prohibited"
    @property
    def description(self) -> str: return "Prohibited"
    @property
    def schema(self) -> dict: return {}
    @property
    def permission(self) -> PermissionLevel: return PermissionLevel.PROHIBITED
    def execute(self, **kwargs):
        return "hacked"

class DummyFailingTool(BaseTool):
    @property
    def name(self) -> str: return "dummy_fail"
    @property
    def description(self) -> str: return "Fails"
    @property
    def schema(self) -> dict: return {}
    @property
    def permission(self) -> PermissionLevel: return PermissionLevel.SAFE
    def execute(self, **kwargs):
        raise ValueError("Intentional crash")

def test_tool_registration():
    registry = ToolRegistry()
    tool = DummySafeTool()
    registry.register(tool)
    
    retrieved = registry.get_tool("dummy_safe")
    assert retrieved == tool
    
    with pytest.raises(KeyError):
        registry.get_tool("unknown_tool")

def test_tool_schemas():
    registry = ToolRegistry()
    registry.register(DummySafeTool())
    schemas = registry.get_all_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "dummy_safe"

def test_safe_execution():
    registry = ToolRegistry()
    registry.register(DummySafeTool())
    
    result = registry.execute_tool("dummy_safe", "{}")
    data = json.loads(result)
    assert "result" in data
    assert data["result"] == "success"

def test_unknown_tool_execution():
    registry = ToolRegistry()
    result = registry.execute_tool("unknown", "{}")
    data = json.loads(result)
    assert "error" in data
    assert "not found" in data["error"].lower()

def test_prohibited_tool_execution():
    registry = ToolRegistry()
    registry.register(DummyProhibitedTool())
    result = registry.execute_tool("dummy_prohibited", "{}")
    data = json.loads(result)
    assert "error" in data
    assert "denied" in data["error"].lower()

def test_confirm_required_tool_execution():
    registry = ToolRegistry()
    registry.require_confirmation = True
    registry.register(DummyConfirmTool())
    
    result = registry.execute_tool("dummy_confirm", "{}")
    data = json.loads(result)
    assert "error" in data
    assert "confirmation is required" in data["error"].lower()
    
    # Test bypass when REQUIRE_CONFIRMATION is false
    registry.require_confirmation = False
    result2 = registry.execute_tool("dummy_confirm", "{}")
    data2 = json.loads(result2)
    assert "result" in data2
    assert data2["result"] == "confirmed_success"

def test_failing_tool_execution():
    registry = ToolRegistry()
    registry.register(DummyFailingTool())
    result = registry.execute_tool("dummy_fail", "{}")
    data = json.loads(result)
    assert "error" in data
    assert "intentional crash" in data["error"].lower()

def test_invalid_json_arguments():
    registry = ToolRegistry()
    registry.register(DummySafeTool())
    result = registry.execute_tool("dummy_safe", "invalid { json")
    data = json.loads(result)
    assert "error" in data
    assert "invalid json" in data["error"].lower()

def test_system_diagnostics_tool():
    tool = SystemDiagnosticsTool()
    assert tool.name == "system_diagnostics"
    assert tool.permission == PermissionLevel.SAFE
    
    result = tool.execute()
    assert isinstance(result, dict)
    assert "platform" in result
    assert result["status"] == "operational"

class DummyPrivilegedTool(BaseTool):
    @property
    def name(self) -> str: return "dummy_privileged"
    @property
    def description(self) -> str: return "Privileged"
    @property
    def schema(self) -> dict: return {}
    @property
    def permission(self) -> PermissionLevel: return PermissionLevel.PRIVILEGED
    def execute(self, **kwargs):
        return "hacked_root"

class DummyUnknownPermissionTool(BaseTool):
    @property
    def name(self) -> str: return "dummy_unknown_perm"
    @property
    def description(self) -> str: return "Unknown permission"
    @property
    def schema(self) -> dict: return {}
    @property
    def permission(self) -> str: return "made_up_level"
    def execute(self, **kwargs):
        return "should_not_execute"

class DummyMissingPermissionTool(BaseTool):
    @property
    def name(self) -> str: return "dummy_missing_perm"
    @property
    def description(self) -> str: return "Missing permission"
    @property
    def schema(self) -> dict: return {}
    @property
    def permission(self): return None
    def execute(self, **kwargs):
        return "should_not_execute"

def test_privileged_tool_execution():
    registry = ToolRegistry()
    registry.register(DummyPrivilegedTool())
    result = registry.execute_tool("dummy_privileged", "{}")
    data = json.loads(result)
    assert "error" in data
    assert "denied" in data["error"].lower()
    assert "privileged" in data["error"].lower()

def test_unknown_permission_execution():
    registry = ToolRegistry()
    registry.register(DummyUnknownPermissionTool())
    result = registry.execute_tool("dummy_unknown_perm", "{}")
    data = json.loads(result)
    assert "error" in data
    assert "unknown or invalid" in data["error"].lower()

def test_missing_permission_execution():
    registry = ToolRegistry()
    registry.register(DummyMissingPermissionTool())
    result = registry.execute_tool("dummy_missing_perm", "{}")
    data = json.loads(result)
    assert "error" in data
    assert "unknown or invalid" in data["error"].lower()
