import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from backend.memory.boundaries import (
    contains_system_override_attempt,
    is_sensitive_content,
)
from backend.memory.management import MemoryManagementAPI
from backend.memory.models import (
    MemoryCategory,
    MemoryRecord,
    MemorySource,
    MemoryValidationError,
)


class MemoryCommandIntent(str, Enum):
    """Supported intent types for user-facing memory management commands."""

    MEMORY_LIST = "memory_list"
    MEMORY_SEARCH = "memory_search"
    MEMORY_LOOKUP = "memory_lookup"
    MEMORY_CREATE = "memory_create"
    MEMORY_UPDATE = "memory_update"
    MEMORY_FORGET = "memory_forget"
    MEMORY_FORGET_ALL = "memory_forget_all"
    MEMORY_REACTIVATE = "memory_reactivate"
    MEMORY_EXPLAIN = "memory_explain"
    MEMORY_CONFIDENCE = "memory_confidence"
    MEMORY_EXPIRATION = "memory_expiration"
    MEMORY_PRIVACY_STATUS = "memory_privacy_status"
    MEMORY_PRIVACY_ENABLE = "memory_privacy_enable"
    MEMORY_PRIVACY_DISABLE = "memory_privacy_disable"
    MEMORY_RETENTION_STATUS = "memory_retention_status"
    UNKNOWN = "unknown"


@dataclass
class MemoryCommand:
    """Structured representation of a parsed user memory command."""

    intent: MemoryCommandIntent
    category: Optional[MemoryCategory] = None
    key: Optional[str] = None
    value: Optional[str] = None
    query: Optional[str] = None
    target_memory_id: Optional[str] = None
    confirmation_required: bool = False
    original_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryCommandResult:
    """Execution result object returned by MemoryCommandExecutor."""

    success: bool
    intent: MemoryCommandIntent
    message: str
    data: Any = None
    confirmation_required: bool = False
    confirmation_token: Optional[str] = None


class MemoryCommandParser:
    """
    Fast, local, deterministic natural language parser for memory user commands.
    Maps text variations into structured MemoryCommand objects without external LLM calls.
    """

    def parse(self, text: str) -> MemoryCommand:
        if not text or not isinstance(text, str) or not text.strip():
            return MemoryCommand(
                intent=MemoryCommandIntent.UNKNOWN, original_text=str(text or "")
            )

        clean_text = text.strip()
        lower_text = clean_text.lower().rstrip(".?!")

        # False-positive protection: Technical questions containing "memory" or "remember"
        # e.g., "explain java memory management", "how should i remember this concept", "tell me about virtual memory", "how to allocate memory"
        if re.search(
            r"\b(explain|how\s+does|how\s+should|understand|describe|teach|tell\s+me\s+about|allocate|deallocate|leak|stack|heap|garbage)\s+.*?\b(memory\s+management|virtual\s+memory|computer\s+memory|ram|heap|stack|concept|algorithm|code|programming|leak|c|java|cpp|rust)\b",
            lower_text,
        ) or re.search(r"\b(how\s+do\s+i\s+allocate|how\s+to\s+free|memory\s+leak|heap\s+memory|virtual\s+memory)\b", lower_text) or re.search(r"\bhow\s+should\s+i\s+remember\b", lower_text):
            return MemoryCommand(
                intent=MemoryCommandIntent.UNKNOWN, original_text=clean_text
            )

        # Privacy Commands
        if re.search(
            r"^\s*(is\s+memory\s+(enabled|on|active|disabled|off)\??|what\s+are\s+my\s+privacy\s+settings\??|what\s+privacy\s+settings\s+do\s+you\s+have(\s+for\s+my\s+memory)?\??|can\s+you\s+use\s+what\s+you\s+remember\s+about\s+me\??)\s*$",
            lower_text,
        ):
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_PRIVACY_STATUS,
                original_text=clean_text,
            )

        if re.search(
            r"^\s*(turn\s+memory\s+(off|disabled)|disable\s+memory|stop\s+remembering(\s+things)?)\s*$",
            lower_text,
        ):
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_PRIVACY_DISABLE,
                original_text=clean_text,
            )

        if re.search(
            r"^\s*(turn\s+memory\s+(on|enabled)|enable\s+memory|start\s+remembering(\s+things)?)\s*$",
            lower_text,
        ):
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_PRIVACY_ENABLE,
                original_text=clean_text,
            )

        if re.search(
            r"^\s*(how\s+long\s+do\s+you\s+keep\s+my\s+memories\??|what\s+is\s+the\s+memory\s+retention\s+policy\??|show\s+(retention|expiration)\s+status\??)\s*$",
            lower_text,
        ):
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_RETENTION_STATUS,
                original_text=clean_text,
            )

        # 0. Check explicit memory correction patterns first
        from backend.memory.correction import CorrectionDetector
        corr_cand = CorrectionDetector().parse_correction(clean_text)
        if corr_cand:
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_UPDATE,
                category=corr_cand.category,
                key=corr_cand.key,
                value=corr_cand.new_value,
                original_text=clean_text,
                metadata={"correction_candidate": corr_cand},
            )

        # 1. MEMORY_FORGET_ALL (destructive bulk action)
        if re.search(
            r"\b(forget|delete|clear|remove)\s+(everything|all)(\s+(you\s+)?remember(\s+about\s+me)?)?\b",
            lower_text,
        ) or re.search(r"\bclear\s+all\s+memories\b", lower_text):
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_FORGET_ALL,
                confirmation_required=True,
                original_text=clean_text,
            )

        # 2. MEMORY_LIST / SHOW ALL
        if lower_text in [
            "what do you remember about me?",
            "what do you remember about me",
            "what do you remember?",
            "what do you remember",
            "show my memories",
            "show my stored preferences",
            "list my memories",
            "show memories",
        ] or re.search(r"\bshow\s+my\s+(stored\s+)?(memories|preferences)\b", lower_text):
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_LIST,
                original_text=clean_text,
            )

        # 3. MEMORY_EXPIRATION
        if re.search(r"\b(what\s+memories\s+expire|expiring\s+memories|memories\s+expire\s+soon)\b", lower_text):
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_EXPIRATION,
                original_text=clean_text,
            )

        # 4. MEMORY_CONFIDENCE
        if re.search(r"\b(how\s+confident|memory\s+confidence|show\s+confidence)\b", lower_text):
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_CONFIDENCE,
                original_text=clean_text,
            )

        # 5. MEMORY_EXPLAIN
        if re.search(r"\b(why\s+do\s+you\s+remember|when\tag\s+did\s+you\s+learn|explain\s+memory)\b", lower_text):
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_EXPLAIN,
                original_text=clean_text,
            )

        # 6. MEMORY_REACTIVATE / RESTORE
        m_restore = re.search(r"\b(restore|reactivate|unforget)\s+(my\s+)?([a-z0-9_\s]+?)(\s+preference|\s+memory)?$", lower_text)
        if m_restore:
            target_key = m_restore.group(3).strip()
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_REACTIVATE,
                query=target_key,
                original_text=clean_text,
            )

        # 7. MEMORY_FORGET (specific)
        m_forget_pref = re.search(r"\bforget\s+(that\s+i\s+prefer|my\s+preference\s+for|my\s+preference\s+of)\s+([a-z0-9_\s]+)$", lower_text)
        if m_forget_pref:
            val = m_forget_pref.group(2).strip()
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_FORGET,
                category=MemoryCategory.USER_PREFERENCE,
                key="primary_language" if val in ["java", "python", "javascript", "c++", "rust", "go"] else "preference",
                query=val,
                original_text=clean_text,
            )

        m_forget_about = re.search(r"\bforget\s+(everything\s+about\s+my|my)\s+([a-z0-9_\s]+)$", lower_text)
        if m_forget_about:
            target = m_forget_about.group(2).strip()
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_FORGET,
                query=target,
                original_text=clean_text,
            )

        # 8. MEMORY_CREATE / REMEMBER
        m_rem_name = re.search(r"\bremember\s+that\s+my\s+name\s+is\s+([a-z0-9_\s]+)$", lower_text)
        if m_rem_name:
            val = m_rem_name.group(1).strip().capitalize()
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_CREATE,
                category=MemoryCategory.USER_PROFILE,
                key="name",
                value=val,
                original_text=clean_text,
            )

        m_rem_pref = re.search(r"\bremember\s+(that\s+i\s+prefer|my\s+preference\s+for)\s+([a-z0-9_\s]+)$", lower_text)
        if m_rem_pref:
            val = m_rem_pref.group(2).strip().capitalize()
            key_name = "primary_language" if val.lower() in ["java", "python", "javascript", "c++", "rust", "go"] else "preference"
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_CREATE,
                category=MemoryCategory.USER_PREFERENCE,
                key=key_name,
                value=val,
                original_text=clean_text,
            )

        m_rem_like = re.search(r"\bremember\s+i\s+like\s+([a-z0-9_\s]+)\s+(answers|responses)$", lower_text)
        if m_rem_like:
            style_val = m_rem_like.group(1).strip()
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_CREATE,
                category=MemoryCategory.USER_PREFERENCE,
                key="response_style",
                value=style_val,
                original_text=clean_text,
            )

        m_rem_fact = re.search(r"\bremember\s+(that\s+)?(my\s+)?([a-z0-9_\s\-]+?)\s+is\s+([a-z0-9_\s\-]+)$", lower_text)
        if m_rem_fact:
            k = m_rem_fact.group(3).strip()
            v = m_rem_fact.group(4).strip()
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_CREATE,
                category=MemoryCategory.USER_FACT,
                key=k,
                value=v,
                original_text=clean_text,
            )

        # 9. MEMORY_SEARCH / LOOKUP
        m_search = re.search(r"\bwhat\s+do\s+you\s+remember\s+about\s+([a-z0-9_\s]+)\??$", lower_text)
        if m_search:
            topic = m_search.group(1).strip()
            return MemoryCommand(
                intent=MemoryCommandIntent.MEMORY_SEARCH,
                query=topic,
                original_text=clean_text,
            )

        return MemoryCommand(
            intent=MemoryCommandIntent.UNKNOWN,
            original_text=clean_text,
        )


class MemoryCommandExecutor:
    """
    Executes structured MemoryCommands through MemoryManagementAPI.
    Enforces confirmation tokens for destructive bulk operations, conflict resolution,
    and security boundaries.
    """

    def __init__(self, management_api: MemoryManagementAPI):
        self.api = management_api
        self._pending_tokens: Dict[str, MemoryCommand] = {}

    def execute(
        self,
        command: MemoryCommand,
        confirmation_token: Optional[str] = None,
    ) -> MemoryCommandResult:
        if not command or command.intent == MemoryCommandIntent.UNKNOWN:
            return MemoryCommandResult(
                success=False,
                intent=MemoryCommandIntent.UNKNOWN,
                message="Unknown or ambiguous memory command. Please clarify your request.",
            )

        # Security check on values/queries
        check_text = f"{command.value or ''} {command.query or ''}".strip()
        if check_text and (is_sensitive_content(check_text) or contains_system_override_attempt(check_text)):
            return MemoryCommandResult(
                success=False,
                intent=command.intent,
                message="Security Violation: Memory command contains sensitive credentials or system override attempt.",
            )

        # Privacy Commands Execution
        if command.intent == MemoryCommandIntent.MEMORY_PRIVACY_STATUS:
            summary = self.api.get_privacy_settings()
            return MemoryCommandResult(
                success=True,
                intent=command.intent,
                message="Privacy status retrieved.",
                data=summary,
            )

        if command.intent == MemoryCommandIntent.MEMORY_PRIVACY_DISABLE:
            success = self.api.set_memory_enabled(False)
            return MemoryCommandResult(
                success=success,
                intent=command.intent,
                message="Memory disabled successfully." if success else "Failed to update privacy setting.",
                data={"memory_enabled": False},
            )

        if command.intent == MemoryCommandIntent.MEMORY_PRIVACY_ENABLE:
            success = self.api.set_memory_enabled(True)
            return MemoryCommandResult(
                success=success,
                intent=command.intent,
                message="Memory enabled successfully." if success else "Failed to update privacy setting.",
                data={"memory_enabled": True},
            )

        if command.intent == MemoryCommandIntent.MEMORY_RETENTION_STATUS:
            summary = self.api.get_privacy_settings()
            return MemoryCommandResult(
                success=True,
                intent=command.intent,
                message="Retention status retrieved.",
                data=summary,
            )

        # 1. MEMORY_FORGET_ALL (confirmation-gated bulk deletion)
        if command.intent == MemoryCommandIntent.MEMORY_FORGET_ALL:
            if confirmation_token and confirmation_token in self._pending_tokens:
                # Confirmed! Perform soft-deactivation of all memories
                del self._pending_tokens[confirmation_token]
                memories = self.api.list_memories(active_only=True, limit=1000)
                count = 0
                for rec in memories:
                    if self.api.forget_memory(rec.id):
                        count += 1
                return MemoryCommandResult(
                    success=True,
                    intent=MemoryCommandIntent.MEMORY_FORGET_ALL,
                    message=f"Successfully forgot all {count} active memory records.",
                    data={"count": count},
                )
            else:
                # Return confirmation-required result with new token
                token = f"forget_all_{uuid.uuid4().hex[:8]}"
                self._pending_tokens[token] = command
                return MemoryCommandResult(
                    success=False,
                    intent=MemoryCommandIntent.MEMORY_FORGET_ALL,
                    message="Destructive Action Warning: 'Forget everything' requires explicit user confirmation.",
                    confirmation_required=True,
                    confirmation_token=token,
                )

        # 2. MEMORY_LIST
        if command.intent == MemoryCommandIntent.MEMORY_LIST:
            recs = self.api.list_memories(category=command.category, active_only=True)
            return MemoryCommandResult(
                success=True,
                intent=MemoryCommandIntent.MEMORY_LIST,
                message=f"Found {len(recs)} active memory record(s).",
                data=recs,
            )

        # 3. MEMORY_SEARCH
        if command.intent == MemoryCommandIntent.MEMORY_SEARCH:
            q = command.query or ""
            recs = self.api.search_memories(query=q, category=command.category)
            return MemoryCommandResult(
                success=True,
                intent=MemoryCommandIntent.MEMORY_SEARCH,
                message=f"Found {len(recs)} memory record(s) matching '{q}'.",
                data=recs,
            )

        # 3b. MEMORY_LOOKUP
        if command.intent == MemoryCommandIntent.MEMORY_LOOKUP:
            cat = command.category or MemoryCategory.USER_PREFERENCE
            k = command.key or ""
            rec = self.api.get_memory_by_key(cat, k, active_only=True)
            return MemoryCommandResult(
                success=rec is not None,
                intent=MemoryCommandIntent.MEMORY_LOOKUP,
                message=f"Lookup completed for '{k}'.",
                data=rec,
            )

        # 4. MEMORY_CREATE / MEMORY_UPDATE (routes through correction/supersede workflow)
        if command.intent in (MemoryCommandIntent.MEMORY_CREATE, MemoryCommandIntent.MEMORY_UPDATE):
            from backend.memory.correction import MemoryCorrectionWorkflow, CorrectionDecisionOutcome, CorrectionCandidate
            workflow = MemoryCorrectionWorkflow(memory_service=self.api.memory_service)
            if "correction_candidate" in command.metadata:
                cand = command.metadata["correction_candidate"]
            else:
                cand = CorrectionCandidate(
                    category=command.category or MemoryCategory.USER_FACT,
                    key=command.key or "user_fact",
                    new_value=command.value or "",
                    original_text=command.original_text,
                )

            dec = workflow.execute_correction(cand, confirmation_token=confirmation_token)

            if dec.outcome == CorrectionDecisionOutcome.SUCCESS:
                rec = self.api.get_memory_by_id(dec.created_memory_id) if dec.created_memory_id else None
                return MemoryCommandResult(
                    success=True,
                    intent=command.intent,
                    message=dec.message,
                    data=rec,
                )
            else:
                return MemoryCommandResult(
                    success=False,
                    intent=command.intent,
                    message=dec.message or "Unable to update memory.",
                )

        # 5. MEMORY_FORGET
        if command.intent == MemoryCommandIntent.MEMORY_FORGET:
            target_q = (command.query or "").lower()
            recs = self.api.list_memories(active_only=True)
            forgotten_count = 0

            for r in recs:
                if target_q in r.key.lower() or target_q in r.value.lower():
                    if self.api.forget_memory(r.id):
                        forgotten_count += 1

            if forgotten_count > 0:
                return MemoryCommandResult(
                    success=True,
                    intent=MemoryCommandIntent.MEMORY_FORGET,
                    message=f"Successfully forgot {forgotten_count} memory record(s) matching '{command.query}'.",
                    data={"count": forgotten_count},
                )
            else:
                return MemoryCommandResult(
                    success=False,
                    intent=MemoryCommandIntent.MEMORY_FORGET,
                    message=f"No active memory records found matching '{command.query}'.",
                )

        # 6. MEMORY_REACTIVATE
        if command.intent == MemoryCommandIntent.MEMORY_REACTIVATE:
            target_q = (command.query or "").lower()
            all_recs = self.api.list_memories(active_only=False)
            reactivated_rec = None

            for r in all_recs:
                if not r.is_active and (target_q in r.key.lower() or target_q in r.value.lower()):
                    try:
                        reactivated_rec = self.api.reactivate_memory(r.id)
                        if reactivated_rec:
                            break
                    except MemoryValidationError:
                        continue

            if reactivated_rec:
                return MemoryCommandResult(
                    success=True,
                    intent=MemoryCommandIntent.MEMORY_REACTIVATE,
                    message=f"Successfully restored memory '{reactivated_rec.key} = {reactivated_rec.value}'.",
                    data=reactivated_rec,
                )
            else:
                return MemoryCommandResult(
                    success=False,
                    intent=MemoryCommandIntent.MEMORY_REACTIVATE,
                    message=f"No inactive memory records found to restore for '{command.query}'.",
                )

        # 7. MEMORY_EXPLAIN / CONFIDENCE / EXPIRATION
        if command.intent in (MemoryCommandIntent.MEMORY_EXPLAIN, MemoryCommandIntent.MEMORY_CONFIDENCE, MemoryCommandIntent.MEMORY_EXPIRATION):
            recs = self.api.list_memories(active_only=True)
            details = []
            for r in recs:
                if command.intent == MemoryCommandIntent.MEMORY_EXPLAIN:
                    details.append(self.api.inspect_memory_confidence_source(r.id))
                elif command.intent == MemoryCommandIntent.MEMORY_CONFIDENCE:
                    details.append({"key": r.key, "confidence": r.confidence, "source": r.source.value})
                elif command.intent == MemoryCommandIntent.MEMORY_EXPIRATION:
                    details.append(self.api.inspect_expiration(r.id))

            return MemoryCommandResult(
                success=True,
                intent=command.intent,
                message=f"Inspection completed for {len(details)} memory record(s).",
                data=details,
            )

        return MemoryCommandResult(
            success=False,
            intent=MemoryCommandIntent.UNKNOWN,
            message="Unhandled memory command.",
        )
