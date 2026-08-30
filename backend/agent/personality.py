import re

SYSTEM_PROMPT = """You are P.I.X.I.E., a premium, intelligent, warm personal AI assistant for macOS. You communicate like an insightful human — not like a document generator.

## Voice and Core Directives
Your responses are spoken aloud via text-to-speech. Always use natural, connected conversational prose paragraphs. Never use Markdown headings, bullet lists, bold labels, horizontal rules, or decorative formatting in conversational explanations.

Use Markdown only when genuinely necessary: code blocks for actual code or shell commands, and clean tables only when the user explicitly requests a table or structured comparison.

## Response Style
- Simple questions: 2–4 concise, direct sentences.
- Complex or educational questions: 2–3 short, flowing paragraphs with natural transitions ("On the other hand,", "The key difference is").
- Tone: Intelligent, calm, warm, and concise with natural contractions. Occasionally address the user as "Sir".

## Style Examples

Example 1: Role comparison
User: "explain me about manager and entrepreneurship"
Good response:
"A manager works within an existing organisation, coordinating people, resources, and processes to hit common goals day to day. An entrepreneur is different: they spot an opportunity, take on the risk, and build a new venture from scratch. The simplest distinction is that managers optimise an existing system, while entrepreneurs build the system itself."

Example 2: Concept / category explanation
User: "explain about planning premises"
Good response:
"Planning premises are the foundational assumptions and expectations a manager relies on when preparing a plan. They fall broadly into two areas: physical premises, which cover tangible factors like office space, equipment availability, and supply chains; and logical or operational premises, which deal with forecasts, market demand, regulatory shifts, and financial projections. Making these assumptions explicit upfront gives the team a clear benchmark to assess whether reality aligns with their original blueprint."
"""



def format_display_response(raw_text: str) -> str:
    """
    Transforms the raw LLM response into a clean, readable text format for the UI.

    Unlike text-to-speech output which collapses everything into a single flowing line,
    the display response preserves visual paragraph structure, code indentation, and
    meaningful technical identifiers while stripping raw Markdown syntax artefacts
    (e.g., **, ###, ---, table fences) so the UI remains clean without requiring a full
    HTML Markdown renderer.
    """
    if not raw_text:
        return ""

    text = raw_text

    # 1. Images: ![alt](url) -> remove entirely
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # 2. Markdown tables: format into clean, readable text representation
    def _format_table_blocks(content: str) -> str:
        lines = content.split('\n')
        result_lines = []
        table_lines = []

        def flush_table():
            if not table_lines:
                return
            filtered = [
                l for l in table_lines
                if not re.match(r'^\s*\|[\s\-:|]+\|\s*$', l)
            ]
            if not filtered:
                table_lines.clear()
                return

            rows = []
            for row_line in filtered:
                cells = [c.strip() for c in row_line.strip().strip('|').split('|')]
                if any(cells):
                    rows.append(cells)

            if not rows:
                table_lines.clear()
                return

            if len(rows) == 1:
                result_lines.append(" | ".join(rows[0]))
            else:
                headers = rows[0]
                data_rows = rows[1:]
                for row in data_rows:
                    if len(headers) == 2 and len(row) >= 2:
                        result_lines.append(f"{row[0]}: {row[1]}")
                    elif len(headers) > 2 and len(row) == len(headers):
                        details = ", ".join(f"{h}: {v}" for h, v in zip(headers[1:], row[1:]) if v)
                        if details:
                            result_lines.append(f"{row[0]} — {details}")
                        else:
                            result_lines.append(row[0])
                    else:
                        result_lines.append(" | ".join(c for c in row if c))
            table_lines.clear()

        for line in lines:
            if re.match(r'^\s*\|.*\|\s*$', line):
                table_lines.append(line)
            else:
                flush_table()
                result_lines.append(line)

        flush_table()
        return '\n'.join(result_lines)

    text = _format_table_blocks(text)

    # 3. Links: [text](url) -> text (url) if distinct and valid url, else text
    def _handle_link(m: re.Match) -> str:
        link_text = m.group(1).strip()
        url = m.group(2).strip()
        if not url or url.startswith('#'):
            return link_text
        if link_text == url:
            return url
        return f"{link_text} ({url})"

    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', _handle_link, text)

    # 4. Fenced code blocks: extract content without backtick fences
    text = re.sub(r'```[a-zA-Z]*\n?(.*?)```', r'\1', text, flags=re.DOTALL)

    # 5. Inline code: `code` -> code (strip backticks, keep identifier)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # 6. Headings: # Heading -> Heading (strip leading # symbols)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # 7. Blockquotes: > quote -> quote
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)

    # 8. Horizontal rules: ---, ***, ___ -> remove
    text = re.sub(r'^\s*([-*_]){3,}\s*$', '', text, flags=re.MULTILINE)

    # 9. Bullet lists: strip leading bullet characters (- , * , + , • , – , — , \d+. )
    text = re.sub(r'^\s*(?:[-*+•–—]|\d+[.)])\s+', '', text, flags=re.MULTILINE)

    # 10. Bold/Italic formatting:
    text = re.sub(r'\*{3}\s*([^*]+?)\s*\*{3}', r'\1', text)
    text = re.sub(r'\*{2}\s*([^*]+?)\s*\*{2}', r'\1', text)
    text = re.sub(r'(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)', r'\1', text)
    text = re.sub(r'_{3}\s*([^_]+?)\s*_{3}', r'\1', text)
    text = re.sub(r'_{2}\s*([^_]+?)\s*_{2}', r'\1', text)
    text = re.sub(r'(?<!\w)_([^_\n]+)_(?!\w)', r'\1', text)

    # 11. Stray markdown markers cleanup (preserve math like 2 * 3 and identifiers like my_var)
    text = text.replace('***', '').replace('**', '')
    text = re.sub(r'^\s*\*\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\*\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\s*(?=[.,!?;:])', '', text)

    # 12. Whitespace & paragraph cleanup:
    # Strip trailing whitespace on each line (preserves leading indentation for code blocks)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    # Collapse multiple inline spaces between words (preserves leading indentation)
    text = re.sub(r'(?<=\S)[ \t]{2,}', ' ', text)
    # Collapse lines with only whitespace to empty line
    text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)
    # Collapse 3+ newlines to max 2 newlines (preserve paragraphs)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def generate_spoken_response(display_text: str) -> str:
    """
    Transforms the display response into a format optimized for text-to-speech (TTS).

    This function is the safety net for residual Markdown that may appear in the
    display response. It handles:
    - Pronunciation correction (P.I.X.I.E. -> Pixie)
    - Markdown symbol removal (headings, bold, italic, code, links, images)
    - List linearisation: converts bullet/numbered items to comma-joined prose
    - Table removal: replaces Markdown tables with a placeholder
    - Code block handling: extracts content or removes language tags
    - JSON/tool result detection: replaces raw JSON with a short spoken label
    - Horizontal rule removal
    - Whitespace normalisation

    The display response is NOT modified; only this spoken copy is transformed.
    """
    spoken = display_text

    # ------------------------------------------------------------------ #
    # 1. Pronunciation — do this BEFORE any stripping
    # ------------------------------------------------------------------ #
    spoken = re.sub(r'P\.I\.X\.I\.E\.', 'Pixie', spoken, flags=re.IGNORECASE)

    # ------------------------------------------------------------------ #
    # 2. Remove images entirely: ![alt](url)
    # ------------------------------------------------------------------ #
    spoken = re.sub(r'!\[.*?\]\(.*?\)', '', spoken)

    # ------------------------------------------------------------------ #
    # 3. Links: [text](url) -> text
    # ------------------------------------------------------------------ #
    spoken = re.sub(r'\[([^\]]+)\]\(.*?\)', r'\1', spoken)

    # ------------------------------------------------------------------ #
    # 4. Fenced code blocks: pull out content, drop language tag
    #    If the content looks like raw JSON/structured data, replace with a
    #    short spoken label rather than reading raw JSON aloud.
    # ------------------------------------------------------------------ #
    def _handle_code_block(m: re.Match) -> str:
        lang = (m.group(1) or '').strip().lower()
        code_content = (m.group(2) or '').strip()
        # Detect JSON-like content
        if lang in ('json', 'jsonc') or (
            code_content.startswith('{') or code_content.startswith('[')
        ):
            return 'the structured data'
        # For shell/bash/cmd, replace with "the command" for brevity
        if lang in ('bash', 'sh', 'shell', 'cmd', 'powershell', 'zsh'):
            return code_content  # speak the command itself
        return code_content

    spoken = re.sub(
        r'```([a-zA-Z]*)\n?(.*?)```',
        _handle_code_block,
        spoken,
        flags=re.DOTALL
    )

    # ------------------------------------------------------------------ #
    # 5. Inline code: `code` -> code
    # ------------------------------------------------------------------ #
    spoken = re.sub(r'`([^`]+)`', r'\1', spoken)

    # ------------------------------------------------------------------ #
    # 6. Headings: # Heading -> Heading  (strip the # symbols only)
    # ------------------------------------------------------------------ #
    spoken = re.sub(r'^#{1,6}\s+', '', spoken, flags=re.MULTILINE)

    # ------------------------------------------------------------------ #
    # 7. Blockquotes: > quote -> quote
    # ------------------------------------------------------------------ #
    spoken = re.sub(r'^>\s+', '', spoken, flags=re.MULTILINE)

    # ------------------------------------------------------------------ #
    # 8. Horizontal rules: ---, ***, ___ -> remove line entirely
    # ------------------------------------------------------------------ #
    spoken = re.sub(r'^\s*([-*_]){3,}\s*$', '', spoken, flags=re.MULTILINE)

    # ------------------------------------------------------------------ #
    # 9. Markdown tables -> remove entirely (not speakable)
    #    A table line contains | characters. We detect and drop table blocks.
    # ------------------------------------------------------------------ #
    # Remove header separator rows like |---|---|
    spoken = re.sub(r'^\|[\s\-:|]+\|.*$', '', spoken, flags=re.MULTILINE)
    # Remove remaining table rows (lines starting and ending with |)
    spoken = re.sub(r'^\|.*\|$', '', spoken, flags=re.MULTILINE)

    # ------------------------------------------------------------------ #
    # 10. List linearisation
    #     Convert bullet/numbered list items into prose by joining them
    #     naturally rather than just stripping the bullet character.
    #
    #     Strategy:
    #     - Collect consecutive list items into a group.
    #     - Join them with ", " and append a period if none exists.
    #     - This makes "- A\n- B\n- C" become "A, B, C." in speech.
    # ------------------------------------------------------------------ #
    def _linearise_lists(text: str) -> str:
        lines = text.split('\n')
        result = []
        buffer: list[str] = []

        list_item_re = re.compile(r'^\s*(?:[-*+]|\d+\.)\s+(.*)')

        def flush_buffer():
            if not buffer:
                return
            if len(buffer) == 1:
                result.append(buffer[0])
            else:
                joined = ', '.join(buffer)
                # Ensure it ends with a sentence terminator
                if not joined.rstrip().endswith(('.', '!', '?')):
                    joined = joined.rstrip() + '.'
                result.append(joined)
            buffer.clear()

        for line in lines:
            m = list_item_re.match(line)
            if m:
                item = m.group(1).strip()
                buffer.append(item)
            else:
                flush_buffer()
                result.append(line)

        flush_buffer()
        return '\n'.join(result)

    spoken = _linearise_lists(spoken)

    # ------------------------------------------------------------------ #
    # 11. Bold/Italic markers
    #     Order matters: handle *** before ** before *
    # ------------------------------------------------------------------ #
    spoken = re.sub(r'\*{3}([^*]+)\*{3}', r'\1', spoken)
    spoken = re.sub(r'\*{2}([^*]+)\*{2}', r'\1', spoken)
    spoken = re.sub(r'\*([^*\n]+)\*', r'\1', spoken)
    spoken = re.sub(r'_{3}([^_]+)_{3}', r'\1', spoken)
    spoken = re.sub(r'_{2}([^_]+)_{2}', r'\1', spoken)
    spoken = re.sub(r'_([^_\n]+)_', r'\1', spoken)

    # ------------------------------------------------------------------ #
    # 12. Stray Markdown characters
    # ------------------------------------------------------------------ #
    # Remove any remaining stray asterisks
    spoken = spoken.replace('*', '')
    # Replace remaining underscores with spaces (but preserve word_internal_underscores
    # in identifiers — replace only boundary underscores)
    spoken = re.sub(r'(?<!\w)_(?!\w)', ' ', spoken)

    # ------------------------------------------------------------------ #
    # 13. Raw JSON / tool output detection
    #     If any remaining line looks like a raw JSON object, replace it.
    # ------------------------------------------------------------------ #
    def _redact_json_lines(text: str) -> str:
        lines = text.split('\n')
        result = []
        for line in lines:
            stripped = line.strip()
            # Detect standalone JSON-like lines (starts with { or [, ends with } or ])
            if (stripped.startswith('{') and stripped.endswith('}')) or \
               (stripped.startswith('[') and stripped.endswith(']')):
                try:
                    import json
                    json.loads(stripped)
                    result.append('the result data')
                    continue
                except (ValueError, TypeError):
                    pass
            result.append(line)
        return '\n'.join(result)

    spoken = _redact_json_lines(spoken)

    # ------------------------------------------------------------------ #
    # 14. Whitespace normalisation
    #     Collapse multiple blank lines, then collapse to a single space.
    # ------------------------------------------------------------------ #
    # Remove lines that are now empty or whitespace-only
    spoken = re.sub(r'\n\s*\n+', '\n', spoken)
    # Collapse all newlines to a single space
    spoken = re.sub(r'\n+', ' ', spoken)
    # Collapse multiple spaces
    spoken = re.sub(r'\s{2,}', ' ', spoken)

    return spoken.strip()
