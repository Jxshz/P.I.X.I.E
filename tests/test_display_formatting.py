"""
test_display_formatting.py
==========================
Comprehensive deterministic tests for format_display_response() in backend/agent/personality.py.

Covers:
  - Bold, italic, bold-italic markers
  - Headings (# through ######)
  - Horizontal rules (---, ***, ___)
  - Unordered and ordered lists
  - Markdown tables (retained as clean, readable text representation)
  - Code blocks and inline code
  - Technical identifiers with underscores (e.g. my_variable, token_count)
  - Links and images
  - Blockquotes
  - Whitespace and paragraph preservation
  - Edge cases (empty string, whitespace, nested formatting)
"""

import pytest
from backend.agent.personality import format_display_response


class TestDisplayFormatting:

    def test_bold_double_asterisk(self):
        text = "Hello **world**."
        assert format_display_response(text) == "Hello world."

    def test_bold_double_underscore(self):
        text = "Hello __world__."
        assert format_display_response(text) == "Hello world."

    def test_italic_single_asterisk(self):
        text = "That is *important*."
        assert format_display_response(text) == "That is important."

    def test_italic_single_underscore(self):
        text = "That is _important_."
        assert format_display_response(text) == "That is important."

    def test_bold_italic_triple_asterisk(self):
        text = "This is ***critical***."
        assert format_display_response(text) == "This is critical."

    def test_bold_italic_triple_underscore(self):
        text = "This is ___critical___."
        assert format_display_response(text) == "This is critical."

    def test_headings_stripped(self):
        assert format_display_response("# Heading 1") == "Heading 1"
        assert format_display_response("## Heading 2") == "Heading 2"
        assert format_display_response("### Heading 3") == "Heading 3"
        assert format_display_response("###### Heading 6") == "Heading 6"

    def test_horizontal_rules_removed(self):
        text = "Section A\n\n---\n\nSection B"
        result = format_display_response(text)
        assert "---" not in result
        assert "Section A" in result
        assert "Section B" in result

    def test_horizontal_rules_asterisks(self):
        text = "Top\n\n***\n\nBottom"
        result = format_display_response(text)
        assert "***" not in result
        assert "Top" in result
        assert "Bottom" in result

    def test_bullet_lists_stripped_cleanly(self):
        text = "- Primary winding\n- Secondary winding\n- Core"
        result = format_display_response(text)
        assert "- Primary" not in result
        assert "Primary winding\nSecondary winding\nCore" in result

    def test_bullet_stars_and_plus(self):
        text = "* Item 1\n* Item 2\n+ Item 3"
        result = format_display_response(text)
        assert "* Item" not in result
        assert "+ Item" not in result
        assert "Item 1\nItem 2\nItem 3" in result

    def test_markdown_tables_preserved_as_readable_text(self):
        text = "| Item | Value |\n|---|---|\n| Voltage | 230 V |\n| Current | 5 A |"
        result = format_display_response(text)
        assert "|" not in result
        assert "---" not in result
        assert "Voltage: 230 V" in result
        assert "Current: 5 A" in result

    def test_multi_column_tables_preserved(self):
        text = "| Language | Type | Speed |\n|---|---|---|\n| Python | Dynamic | High |\n| Rust | Static | Maximum |"
        result = format_display_response(text)
        assert "|" not in result
        assert "Python" in result
        assert "Type: Dynamic" in result
        assert "Speed: High" in result
        assert "Rust" in result
        assert "Type: Static" in result
        assert "Speed: Maximum" in result

    def test_code_blocks_content_preserved(self):
        text = "Here is the code:\n```python\ndef greet():\n    return 'hello'\n```"
        result = format_display_response(text)
        assert "```" not in result
        assert "def greet():\n    return 'hello'" in result

    def test_inline_code_backticks_removed_identifier_preserved(self):
        text = "Set `total_tokens` and call `calculate_usage(arg)`."
        result = format_display_response(text)
        assert "`" not in result
        assert "Set total_tokens and call calculate_usage(arg)." == result

    def test_identifiers_with_underscores_not_mangled(self):
        text = "The variable my_user_id and file_path_name should stay intact."
        result = format_display_response(text)
        assert "my_user_id" in result
        assert "file_path_name" in result

    def test_math_asterisks_not_removed(self):
        text = "Calculate 2 * 3 = 6."
        result = format_display_response(text)
        assert "2 * 3 = 6" in result

    def test_links_converted_cleanly(self):
        text = "Check the [documentation](https://docs.example.com) for details."
        result = format_display_response(text)
        assert "[documentation]" not in result
        assert "documentation (https://docs.example.com)" in result

    def test_images_removed(self):
        text = "Look at this: ![diagram](https://example.com/diag.png) Nice chart."
        result = format_display_response(text)
        assert "![" not in result
        assert "diag.png" not in result
        assert "Look at this: Nice chart." in result

    def test_blockquotes_preserved_without_marker(self):
        text = "> Note: This is an important quote."
        result = format_display_response(text)
        assert ">" not in result
        assert "Note: This is an important quote." == result

    def test_paragraphs_preserved(self):
        text = "Paragraph 1 is here.\n\nParagraph 2 is here.\n\nParagraph 3 is here."
        result = format_display_response(text)
        assert result == text

    def test_excess_newlines_collapsed_to_paragraphs(self):
        text = "Paragraph 1\n\n\n\n\nParagraph 2"
        result = format_display_response(text)
        assert result == "Paragraph 1\n\nParagraph 2"

    def test_empty_and_whitespace(self):
        assert format_display_response("") == ""
        assert format_display_response("   ") == ""
        assert format_display_response("\n\n\n") == ""

    def test_complex_mixed_markdown(self):
        text = (
            "## Transformer Architecture\n\n"
            "A **transformer** uses *self-attention* mechanisms.\n\n"
            "---\n\n"
            "### Components\n\n"
            "- Encoder layer\n"
            "- Decoder layer\n\n"
            "```python\nmodel = Transformer()\n```\n\n"
            "Refer to `model_config.json` for parameters."
        )
        result = format_display_response(text)
        assert "##" not in result
        assert "###" not in result
        assert "**" not in result
        assert "---" not in result
        assert "```" not in result
        assert "`model_config.json`" not in result
        assert "model_config.json" in result
        assert "Transformer Architecture" in result
        assert "A transformer uses self-attention mechanisms." in result
        assert "Encoder layer" in result
        assert "Decoder layer" in result
        assert "model = Transformer()" in result

    def test_screenshot_defect_planning_premises(self):
        """
        Step 7 Mandatory Test: Reproduces the exact screenshot defect pattern.
        """
        text = (
            "- **Physical premises** — the building or space you need to plan.\n"
            "- **Logical premises** — the assumptions or statements that form the basis of a plan."
        )
        result = format_display_response(text)
        assert "**" not in result
        assert "***" not in result
        assert "##" not in result
        assert "---" not in result
        assert not any(line.strip().startswith("- ") for line in result.split("\n"))
        assert "Physical premises — the building or space you need to plan." in result
        assert "Logical premises — the assumptions or statements that form the basis of a plan." in result

    def test_combined_heading_and_bold_bullets(self):
        text = (
            "### Manager\n"
            "- **People**\n"
            "- **Process**\n"
            "- **Execution**"
        )
        result = format_display_response(text)
        assert "###" not in result
        assert "**" not in result
        assert "- " not in result
        assert "Manager" in result
        assert "People" in result
        assert "Process" in result
        assert "Execution" in result

    def test_bold_definition_lines(self):
        text = (
            "**Manager** — an existing organisation.\n"
            "**Entrepreneur** — a new venture."
        )
        result = format_display_response(text)
        assert "**" not in result
        assert "Manager — an existing organisation." in result
        assert "Entrepreneur — a new venture." in result

    def test_preservation_of_technical_terms(self):
        text = "Use `my_variable` with [https://example.com](https://example.com) and calculate 2 * 3 = 6 using Java HashMap over HTTP/HTTPS."
        result = format_display_response(text)
        assert "my_variable" in result
        assert "https://example.com" in result
        assert "2 * 3 = 6" in result
        assert "Java HashMap" in result
        assert "HTTP/HTTPS" in result

    def test_unicode_and_numbered_bullets(self):
        text = (
            "• **First point** — detail 1\n"
            "– **Second point** — detail 2\n"
            "1. **Third point** — detail 3"
        )
        result = format_display_response(text)
        assert "**" not in result
        assert "•" not in result
        assert "First point — detail 1" in result
        assert "Second point — detail 2" in result
        assert "Third point — detail 3" in result
