import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from groq import AsyncGroq
from dotenv import load_dotenv
from backend.agent.personality import SYSTEM_PROMPT, generate_spoken_response

# Load .env from the project root
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class AgentCore:
    def __init__(self):
        # AsyncGroq automatically picks up GROQ_API_KEY from env
        self.client = AsyncGroq()
        # Use model from environment, fallback to the confirmed working model
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    async def process_intent(self, user_input: str) -> Tuple[str, str]:
        """
        Process the user's intent via Groq and return the response.
        Returns:
            Tuple containing (display_response, spoken_response)
        """
        self.conversation_history.append({"role": "user", "content": user_input})
        
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=self.conversation_history,
                model=self.model,
                temperature=0.7,
                max_tokens=1024,
            )
            
            response = chat_completion.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": response})
            
            spoken_response = generate_spoken_response(response)
            return response, spoken_response
            
        except Exception as e:
            error_msg = f"Error connecting to Groq API: {str(e)}"
            return error_msg, error_msg

    def clear_context(self):
        """Reset the conversation context."""
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
