import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from groq import AsyncGroq
from dotenv import load_dotenv
from backend.agent.personality import SYSTEM_PROMPT, generate_spoken_response
from backend.agent.token_governor import TokenGovernor
from backend.storage.usage_store import UsageStore
from backend.tools import ToolRegistry, SystemDiagnosticsTool

# Load .env from the project root
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class RateLimitException(Exception):
    """Raised when the Token Governor blocks a request."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class AgentCore:
    def __init__(self, db_path: str = None):
        # AsyncGroq automatically picks up GROQ_API_KEY from env
        self.client = AsyncGroq()
        # Use model from environment, fallback to the confirmed working model
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.governor = TokenGovernor()
        self.usage_store = UsageStore(db_path=db_path)

        self.tool_registry = ToolRegistry()
        self.tool_registry.register(SystemDiagnosticsTool())

        self.conversation_history: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def _trim_context(self):
        """
        Trims the conversation history to stay within a safe token budget.
        Preserves the system message (index 0) and the most recent turns.
        Safely handles tool call/response pairs.
        """
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

    async def process_intent(self, user_input: str) -> Tuple[str, str]:
        """
        Process the user's intent via Groq, handling potential tool calls.
        Returns:
            Tuple containing (display_response, spoken_response)
        """
        self.conversation_history.append({"role": "user", "content": user_input})

        # Allow a couple of iterations for tool calling (model calls tool -> tool returns -> model replies)
        max_iterations = 3

        for iteration in range(max_iterations):
            try:
                self._trim_context()
                is_allowed, error_msg, reservation = self.governor.preflight(self.conversation_history)
            except Exception as e:
                if iteration == 0:
                    self.conversation_history.pop()
                msg = "Sir, my token governor is currently experiencing issues. Please try again in a moment."
                self.conversation_history.append({"role": "assistant", "content": msg})
                return msg, msg

            if not is_allowed:
                if iteration == 0:
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
                    spoken_response = generate_spoken_response(content)
                    return content, spoken_response

                # We have tool calls, process them and continue the loop
                for tc in message.tool_calls:
                    name = tc.function.name
                    arguments = tc.function.arguments
                    tool_result_str = self.tool_registry.execute_tool(name, arguments)

                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": tool_result_str
                    })

            except Exception as e:
                self.governor.record_usage(reservation, failed=True)
                error_msg = f"Error connecting to Groq API: {str(e)}"
                self.conversation_history.append({"role": "assistant", "content": error_msg})
                return error_msg, error_msg

        # If we exceed max iterations without returning
        msg = "I encountered an issue processing the tool results, taking too many steps."
        self.conversation_history.append({"role": "assistant", "content": msg})
        return msg, msg

    def clear_context(self):
        """Reset the conversation context."""
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the Token Governor."""
        return self.governor.get_status()
