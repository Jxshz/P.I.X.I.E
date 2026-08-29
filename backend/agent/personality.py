import re

SYSTEM_PROMPT = """You are P.I.X.I.E., a premium, highly capable personal AI assistant for macOS.

Your personality guidelines:
- Intelligent, calm, concise, and warm.
- Maintain a conversational and natural tone (use contractions naturally).
- Occasionally address the user as "Sir" (use this naturally, not in every sentence).
- Be proactive and anticipate useful next steps.
- Occasionally use subtle, dry humour.
- Do NOT sound robotic.
- Avoid repeatedly saying "Certainly" or "Of course".
- Avoid unnecessarily restating the user's request.
- Avoid numbered explanations for simple conversational requests.
- Answer simple questions concisely and provide detail only when useful.
- Understand follow-up references and conversational context.
- Distinguish between conversation and commands.
- Never be unnecessarily verbose or write responses that look like documentation unless explicitly asked.
"""

def generate_spoken_response(display_text: str) -> str:
    """
    Transforms the display response into a format optimized for text-to-speech.
    - Strips markdown formatting.
    - Pronounces P.I.X.I.E. as Pixie.
    - Cleans up formatting for natural speech while preserving punctuation.
    """
    spoken = display_text
    
    # 1. Pronunciation replacement (do this early before punctuation stripping)
    spoken = re.sub(r'P\.I\.X\.I\.E\.', 'Pixie', spoken, flags=re.IGNORECASE)
    
    # 2. Images: ![alt](url) -> remove entirely
    spoken = re.sub(r'!\[.*?\]\(.*?\)', '', spoken)
    
    # 3. Links: [text](url) -> text
    spoken = re.sub(r'\[([^\]]+)\]\(.*?\)', r'\1', spoken)
    
    # 4. Code blocks: ```code``` -> code
    spoken = re.sub(r'```[a-zA-Z]*\n?(.*?)```', r'\1', spoken, flags=re.DOTALL)
    
    # 5. Inline code: `code` -> code
    spoken = re.sub(r'`([^`]+)`', r'\1', spoken)
    
    # 6. Headings: # Heading -> Heading
    spoken = re.sub(r'^#{1,6}\s+', '', spoken, flags=re.MULTILINE)
    
    # 7. Blockquotes: > quote -> quote
    spoken = re.sub(r'^>\s+', '', spoken, flags=re.MULTILINE)
    
    # 8. Horizontal rules: ---, ***, ___ -> remove
    spoken = re.sub(r'^\s*([-*_]){3,}\s*$', '', spoken, flags=re.MULTILINE)
    
    # 9. Lists: - bullet, * bullet, 1. numbered item -> text
    spoken = re.sub(r'^\s*(?:[-*]|\d+\.)\s+', '', spoken, flags=re.MULTILINE)
    
    # 10. Bold/Italic emphasis markers (***, **, *, ___, __, _) 
    # Match emphasis safely: look for 1 to 3 asterisks or underscores surrounding text.
    # Non-greedy match inside to capture the content.
    spoken = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', spoken)
    spoken = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', spoken)
    
    # 11. Stray * and _ characters (often leftover from complex formatting)
    # Remove isolated * and _, but be careful with _ if it's inside words (like var_name). 
    # For now, just remove * globally. 
    spoken = spoken.replace('*', '')
    # For underscores, remove them if they are standalone or word boundaries to be safe
    spoken = re.sub(r'\b_\b', ' ', spoken)
    spoken = re.sub(r'_', ' ', spoken) # Just replace remaining _ with space for natural speech
    
    # 12. Normalize whitespace (remove excess newlines and double spaces)
    spoken = re.sub(r'\n+', ' ', spoken)
    spoken = re.sub(r'\s{2,}', ' ', spoken)
    
    return spoken.strip()
