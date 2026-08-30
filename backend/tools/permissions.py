from enum import Enum

class PermissionLevel(Enum):
    """Defines the permission boundary for tools."""
    SAFE = "safe"                          # Read-only or completely harmless actions
    CONFIRM_REQUIRED = "confirm_required"  # Requires explicit user confirmation
    PRIVILEGED = "privileged"              # Requires escalated privileges (placeholder)
    PROHIBITED = "prohibited"              # Explicitly denied
