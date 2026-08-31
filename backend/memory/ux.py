import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from backend.memory.boundaries import (
    contains_system_override_attempt,
    is_sensitive_content,
)
from backend.memory.commands import MemoryCommandIntent, MemoryCommandResult
from backend.memory.models import MemoryCategory, MemoryRecord, MemorySource


class MemoryUXStatus(str, Enum):
    """Deterministic interaction status codes for all memory UX interactions."""

    SUCCESS = "SUCCESS"
    EMPTY = "EMPTY"
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    FORGOTTEN = "FORGOTTEN"
    UPDATED = "UPDATED"
    REACTIVATED = "REACTIVATED"
    EXPIRED = "EXPIRED"
    SUPERSEDED = "SUPERSEDED"
    AMBIGUOUS = "AMBIGUOUS"
    SECURITY_REJECTED = "SECURITY_REJECTED"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


def format_confidence_level(confidence: Optional[float]) -> str:
    """
    Converts floating-point confidence scores into stable human-readable levels:
    0.00–0.39 -> Low
    0.40–0.69 -> Medium
    0.70–1.00 -> High
    """
    if confidence is None:
        return "Medium"
    if confidence < 0.40:
        return "Low"
    if confidence < 0.70:
        return "Medium"
    return "High"


def format_provenance_source(source: Optional[Union[MemorySource, str]]) -> str:
    """
    Converts internal memory source provenance into natural user explanations.
    """
    if not source:
        return "inferred from previous interactions."
    src_str = source.value if hasattr(source, "value") else str(source)
    if src_str == "explicit_user_input":
        return "explicitly provided by you."
    return "inferred from previous interactions."


@dataclass
class MemoryUXResponse:
    """
    Unified, deterministic UX Response Contract DTO for P.I.X.I.E. memory interactions.
    """

    intent: MemoryCommandIntent
    status: MemoryUXStatus
    response_text: str
    memories: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, str]] = field(default_factory=list)
    confirmation_required: bool = False
    confirmation_token: Optional[str] = None
    explanation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def status_description(self) -> str:
        """Backward-compatible property returning status string representation."""
        return self.status.value if hasattr(self.status, "value") else str(self.status)


class MemoryUXFormatter:
    """
    Central presentation engine that enforces the P.I.X.I.E. Memory UX Contract.
    Converts structured backend results into natural, conversational assistant responses.
    """

    CATEGORY_TITLES = {
        MemoryCategory.USER_PROFILE: "ABOUT ME",
        MemoryCategory.USER_PREFERENCE: "PREFERENCES",
        MemoryCategory.USER_FACT: "FACTS",
        MemoryCategory.CONTEXT_RULE: "RULES",
    }

    def format_command_result(self, result: MemoryCommandResult) -> MemoryUXResponse:
        """
        Transforms a MemoryCommandResult into a compliant MemoryUXResponse object.
        """
        if not result:
            return MemoryUXResponse(
                intent=MemoryCommandIntent.UNKNOWN,
                status=MemoryUXStatus.ERROR,
                response_text="I couldn't process your memory request right now.",
                error="Null command result.",
            )

        # 1. Security Violation / Rejection
        if not result.success and "Security Violation" in (result.message or ""):
            safe_msg = "I can't save that because it appears to contain sensitive credential information." if "sensitive" in result.message.lower() else "I can't process that as a memory instruction."
            return MemoryUXResponse(
                intent=result.intent,
                status=MemoryUXStatus.SECURITY_REJECTED,
                response_text=safe_msg,
                error=safe_msg,
            )

        # 2. Destructive Bulk Deletion Confirmation Required
        if result.confirmation_required and result.intent == MemoryCommandIntent.MEMORY_FORGET_ALL:
            prompt_text = (
                "This will remove all active memories.\n"
                "Please confirm if you'd like me to continue.\n\n"
                "[ Confirm ] [ Cancel ]"
            )
            actions = [
                {"label": "Confirm", "action": f"confirm_{result.confirmation_token}"},
                {"label": "Cancel", "action": "cancel"},
            ]
            return MemoryUXResponse(
                intent=result.intent,
                status=MemoryUXStatus.PENDING_CONFIRMATION,
                response_text=prompt_text,
                confirmation_required=True,
                confirmation_token=result.confirmation_token,
                actions=actions,
            )

        # 3. Ambiguous Target Response
        if not result.success and "more than one" in (result.message or "").lower():
            return MemoryUXResponse(
                intent=result.intent,
                status=MemoryUXStatus.AMBIGUOUS,
                response_text="I found more than one possible memory to update. Please tell me which one you mean.",
            )

        # 4. General Failure / No Target
        if not result.success:
            if result.intent == MemoryCommandIntent.MEMORY_SEARCH:
                msg = "I couldn't find a saved memory matching that."
            elif result.intent == MemoryCommandIntent.MEMORY_UPDATE:
                msg = "I couldn't find a saved memory to update."
            else:
                msg = self._sanitize_error(result.message)
            return MemoryUXResponse(
                intent=result.intent,
                status=MemoryUXStatus.ERROR if result.intent not in (MemoryCommandIntent.MEMORY_SEARCH, MemoryCommandIntent.MEMORY_UPDATE) else MemoryUXStatus.EMPTY,
                response_text=msg,
                error=msg,
            )

        # 5. MEMORY_LIST
        if result.intent == MemoryCommandIntent.MEMORY_LIST:
            recs: List[MemoryRecord] = result.data if isinstance(result.data, list) else []
            if not recs:
                return MemoryUXResponse(
                    intent=result.intent,
                    status=MemoryUXStatus.EMPTY,
                    response_text="I don't have any saved memories about you yet.",
                    memories=[],
                )
            return self._format_list_response(recs, result.intent)

        # 6. MEMORY_SEARCH / MEMORY_LOOKUP
        if result.intent in (MemoryCommandIntent.MEMORY_SEARCH, MemoryCommandIntent.MEMORY_LOOKUP):
            recs = result.data if isinstance(result.data, list) else ([result.data] if isinstance(result.data, MemoryRecord) else [])
            if not recs:
                return MemoryUXResponse(
                    intent=result.intent,
                    status=MemoryUXStatus.EMPTY,
                    response_text="I couldn't find a saved memory matching that.",
                    memories=[],
                )
            return self._format_list_response(recs, result.intent)

        # 7. MEMORY_CREATE
        if result.intent == MemoryCommandIntent.MEMORY_CREATE:
            rec = result.data if isinstance(result.data, MemoryRecord) else None
            val_str = self._sanitize_value(rec.value) if rec else ""
            msg = f"I'll remember that you prefer {val_str}." if val_str else "I'll remember that."
            return MemoryUXResponse(
                intent=result.intent,
                status=MemoryUXStatus.SUCCESS,
                response_text=msg,
                memories=[self._record_to_dto(rec)] if rec else [],
            )

        # 8. MEMORY_UPDATE (Correction)
        if result.intent == MemoryCommandIntent.MEMORY_UPDATE:
            msg = result.message if result.message and "updated" in result.message.lower() else "I've updated your preferred preference."
            rec = result.data if isinstance(result.data, MemoryRecord) else None
            return MemoryUXResponse(
                intent=result.intent,
                status=MemoryUXStatus.UPDATED,
                response_text=msg,
                memories=[self._record_to_dto(rec)] if rec else [],
            )

        # 9. MEMORY_FORGET
        if result.intent == MemoryCommandIntent.MEMORY_FORGET:
            rec = result.data if isinstance(result.data, MemoryRecord) else None
            key_str = rec.key.replace("_", " ") if rec else "that"
            msg = f"I've forgotten your {key_str} preference." if key_str != "that" else "Done. I won't use that preference anymore."
            return MemoryUXResponse(
                intent=result.intent,
                status=MemoryUXStatus.FORGOTTEN,
                response_text=msg,
            )

        # 10. MEMORY_FORGET_ALL (Executed)
        if result.intent == MemoryCommandIntent.MEMORY_FORGET_ALL:
            return MemoryUXResponse(
                intent=result.intent,
                status=MemoryUXStatus.FORGOTTEN,
                response_text="All stored active memories have been removed.",
            )

        # 11. MEMORY_REACTIVATE
        if result.intent == MemoryCommandIntent.MEMORY_REACTIVATE:
            rec = result.data if isinstance(result.data, MemoryRecord) else None
            key_str = rec.key.replace("_", " ") if rec else "memory"
            msg = f"Your {key_str} preference is active again."
            return MemoryUXResponse(
                intent=result.intent,
                status=MemoryUXStatus.REACTIVATED,
                response_text=msg,
                memories=[self._record_to_dto(rec)] if rec else [],
            )

        # 12. MEMORY_EXPLAIN / MEMORY_CONFIDENCE / MEMORY_EXPIRATION
        if result.intent in (MemoryCommandIntent.MEMORY_EXPLAIN, MemoryCommandIntent.MEMORY_CONFIDENCE, MemoryCommandIntent.MEMORY_EXPIRATION):
            details = result.data if isinstance(result.data, list) else []
            return self._format_explanation_response(details, result.intent)

        # 13. PRIVACY STATUS
        if result.intent == MemoryCommandIntent.MEMORY_PRIVACY_STATUS:
            is_enabled = result.data.get("memory_enabled", True) if isinstance(result.data, dict) else True
            msg = "Memory is currently enabled." if is_enabled else "Memory is currently disabled. I won't save or use personal memories until you turn it back on."
            return MemoryUXResponse(
                intent=result.intent,
                status=MemoryUXStatus.SUCCESS,
                response_text=msg,
                metadata=result.data if isinstance(result.data, dict) else {},
            )

        # 14. PRIVACY DISABLE
        if result.intent == MemoryCommandIntent.MEMORY_PRIVACY_DISABLE:
            msg = "Memory is now off. I won't save or use personal memories until you turn it back on."
            return MemoryUXResponse(
                intent=result.intent,
                status=MemoryUXStatus.SUCCESS,
                response_text=msg,
                metadata={"memory_enabled": False},
            )

        # 15. PRIVACY ENABLE
        if result.intent == MemoryCommandIntent.MEMORY_PRIVACY_ENABLE:
            msg = "Memory is enabled again. Existing memories have not been changed."
            return MemoryUXResponse(
                intent=result.intent,
                status=MemoryUXStatus.SUCCESS,
                response_text=msg,
                metadata={"memory_enabled": True},
            )

        # 16. RETENTION STATUS
        if result.intent == MemoryCommandIntent.MEMORY_RETENTION_STATUS:
            msg = "Your saved memories are kept separately from your conversation history. Active memories are retained until you explicitly forget them or until their specified expiration duration."
            return MemoryUXResponse(
                intent=result.intent,
                status=MemoryUXStatus.SUCCESS,
                response_text=msg,
                metadata=result.data if isinstance(result.data, dict) else {},
            )

        # Fallback default
        return MemoryUXResponse(
            intent=result.intent,
            status=MemoryUXStatus.SUCCESS,
            response_text=result.message or "Operation completed.",
        )

    def format_candidate_approval_request(self, category: str, key: str, value: str) -> MemoryUXResponse:
        """
        Formats candidate approval requests for Phase 7.1/7.3 consent candidates.
        """
        clean_val = self._sanitize_value(value)
        prompt_text = (
            f"I can remember that you prefer {clean_val} for future discussions.\n"
            "Would you like me to remember that?\n\n"
            "[ Remember ] [ Don't remember ]"
        )
        actions = [
            {"label": "Remember", "action": f"approve_{key}"},
            {"label": "Don't remember", "action": f"reject_{key}"},
        ]
        return MemoryUXResponse(
            intent=MemoryCommandIntent.MEMORY_CREATE,
            status=MemoryUXStatus.PENDING_APPROVAL,
            response_text=prompt_text,
            actions=actions,
        )

    def _format_list_response(self, recs: List[MemoryRecord], intent: MemoryCommandIntent) -> MemoryUXResponse:
        """Formats records grouped by human category sections."""
        grouped: Dict[MemoryCategory, List[MemoryRecord]] = {}
        dtos: List[Dict[str, Any]] = []

        for r in recs:
            dtos.append(self._record_to_dto(r))
            grouped.setdefault(r.category, []).append(r)

        lines: List[str] = [f"You currently have {len(recs)} active memories:\n"]
        cat_order = [
            MemoryCategory.USER_PROFILE,
            MemoryCategory.USER_PREFERENCE,
            MemoryCategory.USER_FACT,
            MemoryCategory.CONTEXT_RULE,
        ]

        for cat in cat_order:
            if cat in grouped and grouped[cat]:
                title = self.CATEGORY_TITLES.get(cat, cat.value.upper())
                lines.append(f"{title}")
                for r in grouped[cat]:
                    safe_v = self._sanitize_value(r.value)
                    lines.append(f"- {r.key.replace('_', ' ').capitalize()}: {safe_v}")
                lines.append("")

        return MemoryUXResponse(
            intent=intent,
            status=MemoryUXStatus.SUCCESS,
            response_text="\n".join(lines).strip(),
            memories=dtos,
        )

    def _format_explanation_response(self, details: List[Dict[str, Any]], intent: MemoryCommandIntent) -> MemoryUXResponse:
        """Formats provenance, confidence, and expiration metadata into human assistant text."""
        if not details:
            return MemoryUXResponse(
                intent=intent,
                status=MemoryUXStatus.EMPTY,
                response_text="No detailed memory metadata available to display.",
            )

        lines: List[str] = ["Memory Explanation:\n"]
        for d in details[:5]:
            key_name = (d.get("key") or d.get("id") or "Record").replace("_", " ").capitalize()
            conf_level = format_confidence_level(d.get("confidence"))
            src_text = format_provenance_source(d.get("source"))
            exp_text = "Status: Expired" if d.get("is_expired") else "Status: Active"

            lines.append(f"I remember this because you {src_text}\nConfidence: {conf_level}.\n{exp_text}")

        return MemoryUXResponse(
            intent=intent,
            status=MemoryUXStatus.SUCCESS,
            response_text="\n".join(lines).strip(),
            explanation={"details": details},
        )

    def _record_to_dto(self, record: Optional[MemoryRecord]) -> Dict[str, Any]:
        if not record:
            return {}
        return {
            "id": record.id,
            "category": record.category.value if hasattr(record.category, "value") else str(record.category),
            "key": record.key,
            "value": self._sanitize_value(record.value),
            "source": record.source.value if hasattr(record.source, "value") else str(record.source),
            "confidence": record.confidence,
            "confidence_level": format_confidence_level(record.confidence),
            "status": MemoryUXStatus.SUCCESS.value if record.is_active else MemoryUXStatus.FORGOTTEN.value,
        }

    def _sanitize_value(self, val: str) -> str:
        if not val or not isinstance(val, str):
            return ""
        if is_sensitive_content(val) or contains_system_override_attempt(val):
            return "[REDACTED_SECRET]"
        return val

    def _sanitize_error(self, err_msg: str) -> str:
        if not err_msg or not isinstance(err_msg, str):
            return "An unexpected memory error occurred."
        if any(term in err_msg.lower() for term in ["sqlite", ".db", "traceback", "exception", "/users/", "c:\\"]):
            return "A storage or memory processing error occurred."
        return err_msg
