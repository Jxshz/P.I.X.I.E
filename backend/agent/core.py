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
from backend.storage.usage_store import UsageStore
from backend.tools import ToolRegistry, SystemDiagnosticsTool
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

class AgentCore:
    def __init__(self, db_path: str = None):
        # AsyncGroq automatically picks up GROQ_API_KEY from env
        self.client = AsyncGroq()
        # Use model from environment, fallback to the confirmed working model
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        self.governor = TokenGovernor()
        self.usage_store = UsageStore(db_path=db_path)
        self.tool_registry = ToolRegistry()

        # State for confirmation flow
        self.require_confirmation = True
        self.pending_confirmations: Dict[str, PendingConfirmation] = {}
        self.confirmation_lock = threading.Lock()

        # Register core tools
        self.tool_registry.register(SystemDiagnosticsTool())

        # The initial context
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

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

        max_context_tokens = 6000
        while len(self.conversation_history) > 2 and self.governor.estimate_tokens(self.conversation_history) > max_context_tokens:
            msg = self.conversation_history.pop(1)
            # If we pop an assistant message with tool calls, we must also pop the corresponding tool responses
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                num_calls = len(msg["tool_calls"])
                for _ in range(num_calls):
                    if len(self.conversation_history) > 1 and self.conversation_history[1].get("role") == "tool":
                        self.conversation_history.pop(1)
            # If we somehow hit a dangling tool message, pop it too
            while len(self.conversation_history) > 1 and self.conversation_history[1].get("role") == "tool":
                self.conversation_history.pop(1)

    async def handle_confirmation(self, confirmation_id: str, approved: bool) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Handles the resolution of a pending confirmation from the user.
        """
        async with self.context_lock:
            with self.confirmation_lock:
                pending = self.pending_confirmations.pop(confirmation_id, None)

            if not pending:
                # Replay, expired, or forged
                msg = "Confirmation failed: Unknown, expired, or already used confirmation ID."
                self.conversation_history.append({"role": "assistant", "content": msg})
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

            return await self._resume_loop()

    async def process_intent(self, user_input: str) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Process the user's intent via Groq, handling potential tool calls.
        """
        async with self.context_lock:
            self.conversation_history.append({"role": "user", "content": user_input})
            return await self._resume_loop()

    async def _resume_loop(self) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Internal loop to run inference.
        """
        max_iterations = 3

        for iteration in range(max_iterations):
            try:
                self._trim_context()
                is_allowed, error_msg, reservation = self.governor.preflight(self.conversation_history)
            except Exception as e:
                if iteration == 0 and len(self.conversation_history) > 1:
                    self.conversation_history.pop()
                msg = "Sir, my token governor is currently experiencing issues. Please try again in a moment."
                self.conversation_history.append({"role": "assistant", "content": msg})
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
                    "messages": self.conversation_history,
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
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": json.dumps({"error": "Execution skipped: waiting on prior confirmation."})
                        })
                        continue

                    try:
                        tool_result_str = self.tool_registry.execute_tool(name, arguments, tool_call_id=tc.id)
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": tool_result_str
                        })
                    except ConfirmationRequiredException as e:
                        conf_id = secrets.token_hex(16)
                        pending = PendingConfirmation(
                            confirmation_id=conf_id,
                            tool_call_id=e.tool_call_id,
                            tool_name=e.tool_name,
                            arguments_json=e.arguments_json,
                            created_at=time.time(),
                            expires_at=time.time() + 300
                        )
                        with self.confirmation_lock:
                            self.pending_confirmations[conf_id] = pending

                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": f"[PENDING_CONFIRMATION_{conf_id}]"
                        })
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
                display_msg = format_display_response(error_msg)
                spoken_msg = generate_spoken_response(error_msg)
                return display_msg, spoken_msg, None

        # If we exceed max iterations without returning
        msg = "I encountered an issue processing the tool results, taking too many steps."
        self.conversation_history.append({"role": "assistant", "content": msg})
        display_msg = format_display_response(msg)
        spoken_msg = generate_spoken_response(msg)
        return display_msg, spoken_msg, None

    def clear_context(self):
        """Reset the conversation context and clear pending confirmations."""
        with self.confirmation_lock:
            self.pending_confirmations.clear()
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the Token Governor."""
        return self.governor.get_status()
