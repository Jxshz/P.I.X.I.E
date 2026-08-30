import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from groq import AsyncGroq
from dotenv import load_dotenv
from backend.agent.personality import SYSTEM_PROMPT, generate_spoken_response
from backend.agent.token_governor import TokenGovernor

# Load .env from the project root
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class RateLimitException(Exception):
    """Raised when the Token Governor blocks a request."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class AgentCore:
    def __init__(self):
        # AsyncGroq automatically picks up GROQ_API_KEY from env
        self.client = AsyncGroq()
        # Use model from environment, fallback to the confirmed working model
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.governor = TokenGovernor()
        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def _trim_context(self):
        """
        Trims the conversation history to stay within a safe token budget.
        Preserves the system message (index 0) and the most recent turns.
        """
        # Let's say max context tokens we want to send is 6000 (out of 8000 TPM limit)
        max_context_tokens = 6000
        while len(self.conversation_history) > 2 and self.governor.estimate_tokens(self.conversation_history) > max_context_tokens:
            # Remove the oldest message after the system prompt
            self.conversation_history.pop(1)

    async def process_intent(self, user_input: str) -> Tuple[str, str]:
        """
        Process the user's intent via Groq and return the response.
        Returns:
            Tuple containing (display_response, spoken_response)
        """
        self.conversation_history.append({"role": "user", "content": user_input})

        try:
            self._trim_context()
            is_allowed, error_msg, reservation = self.governor.preflight(self.conversation_history)
        except Exception as e:
            self.conversation_history.pop()
            msg = "Sir, my token governor is currently experiencing issues. Please try again in a moment."
            return msg, msg

        if not is_allowed:
            # Do not corrupt history, pop the unprocessed message
            self.conversation_history.pop()
            raise RateLimitException(error_msg)

        try:
            chat_completion = await self.client.chat.completions.create(
                messages=self.conversation_history,
                model=self.model,
                temperature=0.7,
                max_tokens=self.governor.max_completion_tokens,
            )

            self.governor.record_usage(reservation, getattr(chat_completion, 'usage', None), failed=False)

            response = chat_completion.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": response})

            spoken_response = generate_spoken_response(response)
            return response, spoken_response

        except Exception as e:
            self.governor.record_usage(reservation, failed=True)
            error_msg = f"Error connecting to Groq API: {str(e)}"
            return error_msg, error_msg

    def clear_context(self):
        """Reset the conversation context."""
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the Token Governor."""
        return self.governor.get_status()
