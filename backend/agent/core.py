import os
import time
import json
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from groq import AsyncGroq
from dotenv import load_dotenv
from backend.agent.personality import SYSTEM_PROMPT, generate_spoken_response, format_display_response
from backend.agent.token_governor import TokenGovernor
from backend.memory import (
    MemoryCommandExecutor,
    MemoryCommandIntent,
    MemoryCommandParser,
    MemoryContextBuilder,
    MemoryManagementAPI,
    MemoryRetriever,
    MemoryService,
    MemoryUXFormatter,
)
from backend.storage.session_store import SessionStore
from backend.storage.usage_store import UsageStore
from backend.tools import SystemDiagnosticsTool, ToolRegistry
from backend.tools.registry import ConfirmationRequiredException

# Load .env from the project root
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class RateLimitException(Exception):
    """Raised when the Token Governor blocks a request."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

@dataclass
class PendingConfirmation:
    confirmation_id: str
    tool_call_id: str
    tool_name: str
    arguments_json: str
    created_at: float
    expires_at: float
    session_id: Optional[str] = None

class AgentCore:
    def __init__(
        self,
        db_path: Optional[str] = None,
        session_store: Optional[SessionStore] = None,
        session_id: Optional[str] = None,
        memory_service: Optional[MemoryService] = None,
        memory_retriever: Optional[MemoryRetriever] = None,
        enable_memory: bool = True,
        memory_db_path: Optional[str] = None,
    ):
        # AsyncGroq automatically picks up GROQ_API_KEY from env
        self.client = AsyncGroq()
        # Use model from environment, fallback to the confirmed working model
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        self.governor = TokenGovernor()
        self.usage_store = UsageStore(db_path=db_path)
        self.tool_registry = ToolRegistry()

        # SessionStore integration (Phase 5.1.2)
        self.session_store = session_store
        self.session_id = session_id

        if self.session_store:
            if self.session_id:
                # Ensure session exists in SessionStore
                if not self.session_store.get_session(self.session_id):
                    self.session_store.create_session(session_id=self.session_id)
            else:
                # Create a default session
                session = self.session_store.create_session()
                self.session_id = session["id"]

        # Persistent Memory Integration (Phase 6.5 & Phase 8.3)
        self.memory_retriever: Optional[MemoryRetriever] = None
        if enable_memory:
            if memory_retriever:
                self.memory_retriever = memory_retriever
            elif memory_service:
                self.memory_retriever = MemoryRetriever(memory_service=memory_service)
            else:
                try:
                    self.memory_retriever = MemoryRetriever(db_path=memory_db_path)
                except Exception:
                    self.memory_retriever = None

        self.memory_context_builder = MemoryContextBuilder(retriever=self.memory_retriever)

        # Phase 8.3/8.4 User Commands, UX & Management API Integration
        self.memory_management_api: Optional[MemoryManagementAPI] = None
        self.memory_command_parser: Optional[MemoryCommandParser] = None
        self.memory_command_executor: Optional[MemoryCommandExecutor] = None
        self.memory_ux_formatter = MemoryUXFormatter()
        if self.memory_retriever and self.memory_retriever.memory_service:
            self.memory_management_api = MemoryManagementAPI(memory_service=self.memory_retriever.memory_service)
            self.memory_command_parser = MemoryCommandParser()
            self.memory_command_executor = MemoryCommandExecutor(management_api=self.memory_management_api)

        # State for confirmation flow
        self.require_confirmation = True
        self.pending_confirmations: Dict[str, PendingConfirmation] = {}
        self.confirmation_lock = threading.Lock()

        # Register core tools
        self.tool_registry.register(SystemDiagnosticsTool())

        # Load context from SessionStore or initialize default
        self._load_history_from_session_store()

    @property
    def last_memory_retrieval_stats(self) -> Dict[str, Any]:
        """Exposes observability statistics for the most recent memory retrieval."""
        if hasattr(self, "memory_context_builder") and self.memory_context_builder:
            return self.memory_context_builder.last_retrieval_stats
        return {
            "retrieved": False,
            "count": 0,
            "relevance_scores": [],
            "categories": [],
            "memory_ids": [],
            "retrieval_failed": False,
        }

    def _load_history_from_session_store(self):
        """Loads persisted messages from session_store for session_id."""
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        if not self.session_store or not self.session_id:
            return

        try:
            stored_msgs = self.session_store.get_messages(self.session_id)
            for msg in stored_msgs:
                try:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    tool_calls_json = msg.get("tool_calls_json")

                    if role == "user":
                        self.conversation_history.append({"role": "user", "content": content})
                    elif role == "assistant":
                        asst_msg = {"role": "assistant"}
                        if content:
                            asst_msg["content"] = content
                        if tool_calls_json:
                            try:
                                parsed = json.loads(tool_calls_json)
                                if isinstance(parsed, list):
                                    asst_msg["tool_calls"] = parsed
                            except Exception:
                                pass
                        self.conversation_history.append(asst_msg)
                    elif role == "tool":
                        tool_msg = {"role": "tool", "content": content}
                        if tool_calls_json:
                            try:
                                meta = json.loads(tool_calls_json)
                                if isinstance(meta, dict):
                                    if "tool_call_id" in meta:
                                        tool_msg["tool_call_id"] = meta["tool_call_id"]
                                    if "name" in meta:
                                        tool_msg["name"] = meta["name"]
                            except Exception:
                                pass
                        self.conversation_history.append(tool_msg)
                except Exception:
                    continue
        except Exception:
            # Fallback cleanly to system prompt if session loading encounters errors
            self.conversation_history = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

    def _persist_message(self, role: str, content: str, tool_calls_json: Optional[str] = None):
        """Helper to persist a message to session_store safely."""
        if not self.session_store or not self.session_id:
            return
        try:
            self.session_store.add_message(
                session_id=self.session_id,
                role=role,
                content=content,
                tool_calls_json=tool_calls_json
            )
        except Exception:
            # Persistence failures must not corrupt in-memory history or crash flow
            pass

    @property
    def context_lock(self):
        if not hasattr(self, '_context_lock'):
            import asyncio
            self._context_lock = asyncio.Lock()
        return self._context_lock

    def _trim_context(self):
        """
        Trims the conversation history to stay within a safe token budget.
        Preserves the system message (index 0) and the most recent turns.
        Safely handles tool call/response pairs.
        """
        # Sweep expired pending confirmations
        now = time.time()
        with self.confirmation_lock:
            expired_keys = [k for k, v in self.pending_confirmations.items() if now > v.expires_at]
            for k in expired_keys:
                self.pending_confirmations.pop(k, None)
            active_conf_ids = set(self.pending_confirmations.keys())

        max_context_tokens = 3000
        while len(self.conversation_history) > 2 and self.governor.estimate_tokens(self.conversation_history) > max_context_tokens:
            # Check if the message at index 1 is an active pending confirmation tool placeholder
            msg = self.conversation_history[1]
            if msg.get("role") == "tool" and any(f"[PENDING_CONFIRMATION_{cid}]" in str(msg.get("content", "")) for cid in active_conf_ids):
                break

            popped = self.conversation_history.pop(1)
            # If we pop an assistant message with tool calls, we must also pop the corresponding tool responses
            if popped.get("role") == "assistant" and "tool_calls" in popped:
                num_calls = len(popped["tool_calls"])
                for _ in range(num_calls):
                    if len(self.conversation_history) > 1 and self.conversation_history[1].get("role") == "tool":
                        tool_msg = self.conversation_history[1]
                        if not any(f"[PENDING_CONFIRMATION_{cid}]" in str(tool_msg.get("content", "")) for cid in active_conf_ids):
                            self.conversation_history.pop(1)
            # If we hit a dangling non-pending tool message, pop it too
            while len(self.conversation_history) > 1 and self.conversation_history[1].get("role") == "tool":
                tool_msg = self.conversation_history[1]
                if any(f"[PENDING_CONFIRMATION_{cid}]" in str(tool_msg.get("content", "")) for cid in active_conf_ids):
                    break
                self.conversation_history.pop(1)

    async def handle_confirmation(self, confirmation_id: str, approved: bool) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Handles the resolution of a pending confirmation from the user.
        """
        async with self.context_lock:
            with self.confirmation_lock:
                pending = self.pending_confirmations.get(confirmation_id)
                if pending and pending.session_id and self.session_id and pending.session_id != self.session_id:
                    # Cross-session execution rejection
                    pending = None
                else:
                    if pending:
                        self.pending_confirmations.pop(confirmation_id, None)

            if not pending:
                # Replay, expired, cross-session, or forged
                msg = "Confirmation failed: Unknown, expired, or already used confirmation ID."
                self.conversation_history.append({"role": "assistant", "content": msg})
                self._persist_message("assistant", msg)
                display_msg = format_display_response(msg)
                spoken_msg = generate_spoken_response(msg)
                return display_msg, spoken_msg, None

            placeholder = f"[PENDING_CONFIRMATION_{confirmation_id}]"

            if time.time() > pending.expires_at:
                tool_result_str = json.dumps({"error": "Execution aborted: The confirmation request has expired."})
            elif approved:
                # Re-check and execute tool (lock is released so we can execute safely)
                tool_result_str = self.tool_registry.execute_tool(
                    name=pending.tool_name,
                    arguments_json=pending.arguments_json,
                    tool_call_id=pending.tool_call_id,
                    is_confirmed=True
                )
            else:
                tool_result_str = json.dumps({"error": "Execution aborted: User rejected the confirmation."})

            # Find the placeholder and replace it
            replaced = False
            for msg in reversed(self.conversation_history):
                if msg.get("role") == "tool" and msg.get("content") == placeholder:
                    msg["content"] = tool_result_str
                    replaced = True
                    break

            if not replaced:
                # If the placeholder was trimmed due to context length, append the result directly
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": pending.tool_call_id,
                    "name": pending.tool_name,
                    "content": tool_result_str
                })

            tool_meta = json.dumps({"tool_call_id": pending.tool_call_id, "name": pending.tool_name})
            self._persist_message("tool", tool_result_str, tool_calls_json=tool_meta)

            return await self._resume_loop()

    async def process_intent(self, user_input: str) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Process the user's intent via memory commands or Groq LLM.
        """
        async with self.context_lock:
            # Phase 8.3/8.4: Memory Command Detection & UX Formatting (Application Control Path)
            if self.memory_command_parser and self.memory_command_executor:
                cmd = self.memory_command_parser.parse(user_input)
                if cmd and cmd.intent != MemoryCommandIntent.UNKNOWN:
                    res = self.memory_command_executor.execute(cmd)
                    ux_res = self.memory_ux_formatter.format_command_result(res)
                    asst_text = ux_res.response_text

                    # Store user turn in history & SessionStore
                    self.conversation_history.append({"role": "user", "content": user_input})
                    self._persist_message("user", user_input)

                    self.conversation_history.append({"role": "assistant", "content": asst_text})
                    self._persist_message("assistant", asst_text)

                    display_msg = format_display_response(asst_text)
                    spoken_msg = generate_spoken_response(asst_text)
                    return display_msg, spoken_msg, None

            self.conversation_history.append({"role": "user", "content": user_input})
            self._persist_message("user", user_input)
            return await self._resume_loop()

    def _get_llm_messages(self) -> List[Dict[str, Any]]:
        """
        Constructs the message payload for LLM inference.
        Injects retrieved untrusted memory context if available,
        without mutating self.conversation_history or SessionStore.
        """
        if not hasattr(self, "memory_context_builder") or not self.memory_context_builder:
            return list(self.conversation_history)

        user_query = ""
        for msg in reversed(self.conversation_history):
            if msg.get("role") == "user" and msg.get("content"):
                user_query = msg["content"]
                break

        if not user_query:
            return list(self.conversation_history)

        memory_context = self.memory_context_builder.build_memory_context(query=user_query)
        if not memory_context:
            return list(self.conversation_history)

        llm_msgs = list(self.conversation_history)
        if len(llm_msgs) >= 2 and llm_msgs[-1].get("role") == "user":
            memory_msg = {"role": "system", "content": memory_context}
            llm_msgs.insert(len(llm_msgs) - 1, memory_msg)
        else:
            llm_msgs.append({"role": "system", "content": memory_context})

        return llm_msgs

    async def _resume_loop(self) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Internal loop to run inference.
        """
        max_iterations = 3

        for iteration in range(max_iterations):
            try:
                self._trim_context()
                llm_messages = self._get_llm_messages()
                is_allowed, error_msg, reservation = self.governor.preflight(llm_messages)
            except Exception as e:
                if iteration == 0 and len(self.conversation_history) > 1:
                    self.conversation_history.pop()
                msg = "Sir, my token governor is currently experiencing issues. Please try again in a moment."
                self.conversation_history.append({"role": "assistant", "content": msg})
                self._persist_message("assistant", msg)
                display_msg = format_display_response(msg)
                spoken_msg = generate_spoken_response(msg)
                return display_msg, spoken_msg, None

            if not is_allowed:
                if iteration == 0 and len(self.conversation_history) > 1:
                    self.conversation_history.pop()
                self.usage_store.record_rate_limit(self.model)
                raise RateLimitException(error_msg)

            try:
                tools = self.tool_registry.get_all_tool_schemas()
                kwargs = {
                    "messages": llm_messages,
                    "model": self.model,
                    "temperature": 0.7,
                    "max_tokens": self.governor.max_completion_tokens,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                chat_completion = await self.client.chat.completions.create(**kwargs)

                usage = getattr(chat_completion, 'usage', None)
                self.governor.record_usage(reservation, usage, failed=False)

                if usage:
                    req_tokens = getattr(usage, 'prompt_tokens', None)
                    tot_tokens = getattr(usage, 'total_tokens', 0)
                    self.usage_store.record_success(self.model, total_tokens=tot_tokens, request_tokens=req_tokens)
                else:
                    self.usage_store.record_success(self.model, total_tokens=0, request_tokens=None)

                message = chat_completion.choices[0].message

                # Construct dictionary for history safely
                assistant_msg = {"role": "assistant"}
                if message.content is not None:
                    assistant_msg["content"] = message.content

                if message.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in message.tool_calls
                    ]

                self.conversation_history.append(assistant_msg)
                tc_json = json.dumps(assistant_msg["tool_calls"]) if "tool_calls" in assistant_msg else None
                self._persist_message("assistant", assistant_msg.get("content", ""), tool_calls_json=tc_json)

                # If there are no tool calls, the model has given its final response
                if not message.tool_calls:
                    content = message.content or ""
                    display_response = format_display_response(content)
                    spoken_response = generate_spoken_response(content)
                    return display_response, spoken_response, None

                # We have tool calls, process them and continue the loop
                skip_remaining = False
                pending_conf = None

                for tc in message.tool_calls:
                    name = tc.function.name
                    arguments = tc.function.arguments

                    if skip_remaining:
                        skipped_str = json.dumps({"error": "Execution skipped: waiting on prior confirmation."})
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": skipped_str
                        })
                        tool_meta = json.dumps({"tool_call_id": tc.id, "name": name})
                        self._persist_message("tool", skipped_str, tool_calls_json=tool_meta)
                        continue

                    try:
                        tool_result_str = self.tool_registry.execute_tool(name, arguments, tool_call_id=tc.id)
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": tool_result_str
                        })
                        tool_meta = json.dumps({"tool_call_id": tc.id, "name": name})
                        self._persist_message("tool", tool_result_str, tool_calls_json=tool_meta)
                    except ConfirmationRequiredException as e:
                        conf_id = secrets.token_hex(16)
                        pending = PendingConfirmation(
                            confirmation_id=conf_id,
                            tool_call_id=e.tool_call_id,
                            tool_name=e.tool_name,
                            arguments_json=e.arguments_json,
                            created_at=time.time(),
                            expires_at=time.time() + 300,
                            session_id=self.session_id
                        )
                        with self.confirmation_lock:
                            self.pending_confirmations[conf_id] = pending

                        placeholder_str = f"[PENDING_CONFIRMATION_{conf_id}]"
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": placeholder_str
                        })
                        tool_meta = json.dumps({"tool_call_id": tc.id, "name": name})
                        self._persist_message("tool", placeholder_str, tool_calls_json=tool_meta)
                        skip_remaining = True
                        pending_conf = pending

                if skip_remaining and pending_conf:
                    try:
                        parsed_args = json.loads(pending_conf.arguments_json) if pending_conf.arguments_json else {}
                    except json.JSONDecodeError:
                        parsed_args = {"error": "unparseable_json", "raw_arguments": pending_conf.arguments_json}

                    action_required = {
                        "confirmation_id": pending_conf.confirmation_id,
                        "tool_name": pending_conf.tool_name,
                        "arguments": parsed_args
                    }
                    msg = f"I need your confirmation to execute the tool '{pending_conf.tool_name}'."
                    display_response = format_display_response(msg)
                    spoken_response = generate_spoken_response(msg)
                    return display_response, spoken_response, action_required

            except Exception as e:
                self.governor.record_usage(reservation, failed=True)
                error_msg = f"Error connecting to Groq API: {str(e)}"
                self.conversation_history.append({"role": "assistant", "content": error_msg})
                self._persist_message("assistant", error_msg)
                display_msg = format_display_response(error_msg)
                spoken_msg = generate_spoken_response(error_msg)
                return display_msg, spoken_msg, None

        # If we exceed max iterations without returning
        msg = "I encountered an issue processing the tool results, taking too many steps."
        self.conversation_history.append({"role": "assistant", "content": msg})
        self._persist_message("assistant", msg)
        display_msg = format_display_response(msg)
        spoken_msg = generate_spoken_response(msg)
        return display_msg, spoken_msg, None

    def clear_context(self):
        """Reset the conversation context and clear pending confirmations for active session."""
        with self.confirmation_lock:
            if self.session_id:
                keys_to_remove = [k for k, v in self.pending_confirmations.items() if v.session_id == self.session_id]
                for k in keys_to_remove:
                    self.pending_confirmations.pop(k, None)
            else:
                self.pending_confirmations.clear()
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        if self.session_store and self.session_id:
            try:
                self.session_store.delete_session(self.session_id)
                self.session_store.create_session(session_id=self.session_id)
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the Token Governor."""
        return self.governor.get_status()
