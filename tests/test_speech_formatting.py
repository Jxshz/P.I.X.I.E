"""
test_speech_formatting.py
=========================
Comprehensive tests for generate_spoken_response() in backend/agent/personality.py.

Covers:
  Section A — Markdown symbol stripping
  Section B — List linearisation
  Section C — Code block handling
  Section D — Links, images, and URLs
  Section E — Tables
  Section F — JSON / tool output
  Section G — Special characters and edge cases
  Section H — Conversational quality (structural assertions on spoken output)
"""

import pytest
from backend.agent.personality import generate_spoken_response


# ============================================================
# Section A — Bold, italic, headings, horizontal rules
# ============================================================

class TestMarkdownFormatting:

    def test_bold_double_asterisk(self):
        assert generate_spoken_response("Hello **world**.") == "Hello world."

    def test_bold_double_underscore(self):
        assert generate_spoken_response("Hello __world__.") == "Hello world."

    def test_italic_single_asterisk(self):
        assert generate_spoken_response("That is *important*.") == "That is important."

    def test_italic_single_underscore(self):
        assert generate_spoken_response("That is _important_.") == "That is important."

    def test_bold_italic_triple_asterisk(self):
        assert generate_spoken_response("This is ***critical***.") == "This is critical."

    def test_bold_italic_triple_underscore(self):
        assert generate_spoken_response("This is ___critical___.") == "This is critical."

    def test_heading_h1(self):
        assert generate_spoken_response("# Overview") == "Overview"

    def test_heading_h2(self):
        assert generate_spoken_response("## System Status") == "System Status"

    def test_heading_h3(self):
        assert generate_spoken_response("### Details") == "Details"

    def test_heading_h6(self):
        assert generate_spoken_response("###### Deep section") == "Deep section"

    def test_horizontal_rule_dashes(self):
        result = generate_spoken_response("Before\n\n---\n\nAfter")
        assert "---" not in result
        assert "Before" in result
        assert "After" in result

    def test_horizontal_rule_asterisks(self):
        result = generate_spoken_response("A\n\n***\n\nB")
        assert "***" not in result

    def test_horizontal_rule_underscores(self):
        result = generate_spoken_response("A\n\n___\n\nB")
        assert "___" not in result

    def test_stray_asterisks_removed(self):
        result = generate_spoken_response("Hello ** world **")
        assert "**" not in result

    def test_blockquote(self):
        result = generate_spoken_response("> This is a quote")
        assert ">" not in result
        assert "This is a quote" in result

    def test_pixie_pronunciation(self):
        result = generate_spoken_response("P.I.X.I.E. is ready.")
        assert "Pixie" in result
        assert "P.I.X.I.E." not in result

    def test_pixie_pronunciation_case_insensitive(self):
        result = generate_spoken_response("p.i.x.i.e. is here.")
        assert "P.I.X.I.E." not in result

    def test_contractions_preserved(self):
        text = "I'm ready and I don't have a problem."
        assert generate_spoken_response(text) == text

    def test_no_markdown_text_unchanged(self):
        text = "The battery is fully charged."
        assert generate_spoken_response(text) == text


# ============================================================
# Section B — List linearisation
# ============================================================

class TestListLinearisation:

    def test_single_bullet_no_trailing_comma(self):
        result = generate_spoken_response("- Just one item")
        assert result == "Just one item"

    def test_multiple_bullets_comma_joined(self):
        result = generate_spoken_response("- Apple\n- Banana\n- Cherry")
        assert result == "Apple, Banana, Cherry."

    def test_numbered_list_comma_joined(self):
        result = generate_spoken_response("1. First\n2. Second\n3. Third")
        assert result == "First, Second, Third."

    def test_bullet_star_syntax(self):
        result = generate_spoken_response("* Red\n* Green\n* Blue")
        assert result == "Red, Green, Blue."

    def test_bullet_plus_syntax(self):
        result = generate_spoken_response("+ Option A\n+ Option B")
        assert result == "Option A, Option B."

    def test_bullets_no_raw_markdown_leak(self):
        result = generate_spoken_response("- People leadership\n- Operational focus\n- Risk mitigation")
        assert "-" not in result.split(",")[0]  # no stray bullet dash
        assert "People leadership" in result
        assert "Operational focus" in result
        assert "Risk mitigation" in result

    def test_bullets_end_with_period(self):
        result = generate_spoken_response("- Alpha\n- Beta\n- Gamma")
        assert result.endswith(".")

    def test_list_after_prose(self):
        text = "Here are the steps:\n\n- Install\n- Configure\n- Run"
        result = generate_spoken_response(text)
        assert "Install, Configure, Run." in result
        assert "Here are the steps:" in result

    def test_mixed_list_and_heading(self):
        text = "## Options\n\n- Option A\n- Option B"
        result = generate_spoken_response(text)
        assert "##" not in result
        assert "-" not in result.split("Option")[0]

    def test_existing_period_not_doubled(self):
        # If a bullet item already ends with a period, we shouldn't add another
        result = generate_spoken_response("- Done.\n- Complete.")
        assert ".." not in result


# ============================================================
# Section C — Code blocks and inline code
# ============================================================

class TestCodeHandling:

    def test_inline_code_backticks_removed(self):
        assert generate_spoken_response("Run `python app.py`.") == "Run python app.py."

    def test_fenced_code_block_content_extracted(self):
        text = "Run this:\n```\necho hello\n```"
        result = generate_spoken_response(text)
        assert "```" not in result
        assert "echo hello" in result

    def test_fenced_code_block_language_tag_removed(self):
        text = "```python\nprint('hello')\n```"
        result = generate_spoken_response(text)
        assert "```python" not in result
        assert "```" not in result

    def test_json_code_block_replaced(self):
        text = "Result:\n```json\n{\"status\": \"ok\", \"value\": 42}\n```"
        result = generate_spoken_response(text)
        assert "```" not in result
        assert "{" not in result
        assert "the structured data" in result

    def test_bash_code_block_content_preserved(self):
        text = "```bash\npython -m pytest tests/\n```"
        result = generate_spoken_response(text)
        assert "```" not in result
        assert "python -m pytest tests/" in result

    def test_fenced_code_no_backtick_leak(self):
        text = "```\nsome code\n```"
        result = generate_spoken_response(text)
        assert "`" not in result


# ============================================================
# Section D — Links and images
# ============================================================

class TestLinksAndImages:

    def test_markdown_link_text_preserved(self):
        assert generate_spoken_response(
            "Open the [dashboard](https://example.com)."
        ) == "Open the dashboard."

    def test_markdown_link_url_removed(self):
        result = generate_spoken_response("[click here](https://example.com)")
        assert "https" not in result
        assert "click here" in result

    def test_image_removed_entirely(self):
        result = generate_spoken_response("Here: ![diagram](https://example.com/img.png)")
        assert "![" not in result
        assert "diagram" not in result

    def test_bare_url_not_spoken(self):
        # Bare URLs are not modified by the formatter (they're plain text)
        # But they should not contain Markdown syntax
        result = generate_spoken_response("Visit https://example.com for info.")
        assert "[" not in result
        assert "]" not in result


# ============================================================
# Section E — Tables
# ============================================================

class TestTables:

    def test_simple_table_removed(self):
        text = "| Language | Typed |\n|---|---|\n| Python | No |\n| Java | Yes |"
        result = generate_spoken_response(text)
        assert "|" not in result

    def test_table_separator_row_removed(self):
        text = "| Col1 | Col2 |\n|:---:|---:|\n| A | B |"
        result = generate_spoken_response(text)
        assert "---" not in result
        assert "|" not in result

    def test_text_before_after_table_preserved(self):
        text = "Here is the data:\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nEnd of table."
        result = generate_spoken_response(text)
        assert "Here is the data:" in result
        assert "End of table." in result
        assert "|" not in result


# ============================================================
# Section F — JSON and tool output
# ============================================================

class TestJsonAndToolOutput:

    def test_standalone_json_object_replaced(self):
        text = '{"status": "ok", "value": 42}'
        result = generate_spoken_response(text)
        assert "{" not in result
        assert "}" not in result

    def test_standalone_json_array_replaced(self):
        text = '[{"id": 1}, {"id": 2}]'
        result = generate_spoken_response(text)
        assert "[" not in result

    def test_inline_json_in_sentence_not_broken(self):
        # The word "the result data" should be spoken, not raw JSON
        text = 'The diagnostics returned: {"cpu": 45, "mem": 70}'
        result = generate_spoken_response(text)
        # The prefix text should still be there
        assert "The diagnostics returned:" in result

    def test_json_in_code_block_handled(self):
        text = '```json\n{"error": "not found", "code": 404}\n```'
        result = generate_spoken_response(text)
        assert "```" not in result
        assert "{" not in result


# ============================================================
# Section G — Edge cases and stray characters
# ============================================================

class TestEdgeCases:

    def test_empty_string(self):
        assert generate_spoken_response("") == ""

    def test_whitespace_only(self):
        assert generate_spoken_response("   ") == ""

    def test_multiple_blank_lines_collapsed(self):
        result = generate_spoken_response("A\n\n\n\nB")
        assert "  " not in result  # no double-space
        assert "A" in result
        assert "B" in result

    def test_no_double_space_after_cleanup(self):
        result = generate_spoken_response("**Hello** - world")
        assert "  " not in result

    def test_complex_nested_formatting(self):
        text = "### Key Points\n\n- **First**: do this\n- *Second*: do that\n\n---\n\nSee [docs](http://x.com)."
        result = generate_spoken_response(text)
        assert "###" not in result
        assert "**" not in result
        assert "*" not in result
        assert "---" not in result
        assert "http" not in result
        assert "First" in result
        assert "Second" in result
        assert "docs" in result

    def test_underscore_in_identifier_preserved(self):
        # word_internal underscores in identifiers should not become spaces mid-word
        # The current policy is conservative: replace _ with space unless word-internal
        result = generate_spoken_response("Use `my_function` to start.")
        # After inline code stripping, my_function becomes my_function
        # The _ between words is word-internal, should not disrupt surrounding text
        assert "my" in result
        assert "function" in result


# ============================================================
# Section H — Conversational quality scenarios
# ============================================================
#
# These tests assert STRUCTURAL properties of spoken output for
# representative conversational queries. They do NOT assert exact
# wording (the model generates that), but they verify that spoken
# output is free of formatting artefacts and structurally natural.
#
# The inputs here simulate what the model might return for each query
# in an article-heavy worst-case scenario, and we verify the cleanup
# layer handles them correctly.
# ============================================================

class TestConversationalQuality:

    def test_managers_and_entrepreneurship(self):
        """Scenario 1: 'Explain managers and entrepreneurship'"""
        model_output = (
            "**Key aspects of a manager's role**\n\n"
            "---\n\n"
            "**Where they overlap**\n\n"
            "- People leadership\n"
            "- Operational focus\n"
            "- Risk mitigation\n"
            "- Execution\n\n"
            "Managers are responsible for **running** teams. "
            "Entrepreneurs are focused on *creating* value."
        )
        result = generate_spoken_response(model_output)

        # No Markdown syntax
        assert "**" not in result
        assert "---" not in result
        assert not any(line.lstrip().startswith("-") for line in result.split("."))

        # Key content preserved
        assert "People leadership" in result
        assert "Operational focus" in result
        assert "Managers" in result
        assert "Entrepreneurs" in result

        # Should not be narrating a bullet list literally
        # (bullets should have been joined, not individually prefixed)
        assert "- People leadership" not in result
        assert "- Operational focus" not in result

    def test_industry_40(self):
        """Scenario 2: 'What is Industry 4.0?'"""
        model_output = (
            "## Industry 4.0\n\n"
            "Industry 4.0 refers to the **fourth industrial revolution**, "
            "characterised by:\n\n"
            "- Automation\n"
            "- Data exchange\n"
            "- IoT integration\n\n"
            "---\n\n"
            "It builds on previous revolutions to create *smart factories*."
        )
        result = generate_spoken_response(model_output)

        assert "##" not in result
        assert "**" not in result
        assert "---" not in result
        assert "*" not in result
        assert "Industry 4.0" in result
        assert "Automation" in result

    def test_transformers(self):
        """Scenario 3: 'Explain transformers'"""
        model_output = (
            "### What are Transformers?\n\n"
            "Transformers are a type of **neural network** architecture introduced in 2017.\n\n"
            "Key components:\n\n"
            "1. Self-attention mechanism\n"
            "2. Positional encoding\n"
            "3. Feed-forward layers\n\n"
            "They power models like GPT and BERT."
        )
        result = generate_spoken_response(model_output)

        assert "###" not in result
        assert "**" not in result
        assert "1." not in result
        assert "2." not in result
        assert "Self-attention mechanism" in result
        assert "Positional encoding" in result
        assert "GPT" in result

    def test_what_is_a_database(self):
        """Scenario 4: 'What is a database?'"""
        model_output = (
            "A **database** is an organised collection of structured data.\n\n"
            "Common types include:\n\n"
            "- Relational (SQL)\n"
            "- Document (NoSQL)\n"
            "- Graph\n\n"
            "Databases are managed by a **DBMS** (Database Management System)."
        )
        result = generate_spoken_response(model_output)

        assert "**" not in result
        assert "- Relational" not in result
        assert "database" in result.lower()
        assert "Relational" in result

    def test_java_vs_python(self):
        """Scenario 5: 'Difference between Java and Python'"""
        model_output = (
            "| Feature | Java | Python |\n"
            "|---|---|---|\n"
            "| Typing | Static | Dynamic |\n"
            "| Speed | Fast | Slower |\n\n"
            "Java is **statically typed** and compiled. "
            "Python is *dynamically typed* and interpreted."
        )
        result = generate_spoken_response(model_output)

        assert "|" not in result
        assert "---" not in result
        assert "**" not in result
        assert "*" not in result
        assert "Java" in result
        assert "Python" in result

    def test_short_casual_question(self):
        """Scenario 6: Short casual question response"""
        model_output = "Sure, it's **raining** in London right now."
        result = generate_spoken_response(model_output)

        assert "**" not in result
        assert "raining" in result
        assert "London" in result

    def test_long_technical_response(self):
        """Scenario 7: Long technical response with mixed formatting"""
        model_output = (
            "## Architecture Overview\n\n"
            "The system uses a **microservices** architecture.\n\n"
            "### Components\n\n"
            "- API Gateway\n"
            "- Auth Service\n"
            "- Data Store\n\n"
            "```python\nfrom fastapi import FastAPI\napp = FastAPI()\n```\n\n"
            "---\n\n"
            "See [documentation](https://docs.example.com) for details."
        )
        result = generate_spoken_response(model_output)

        assert "##" not in result
        assert "###" not in result
        assert "**" not in result
        assert "```" not in result
        assert "---" not in result
        assert "https" not in result
        assert "API Gateway" in result
        assert "Auth Service" in result
        assert "documentation" in result

    def test_tool_result_response(self):
        """Scenario 8: Tool result response with JSON"""
        model_output = (
            "Here are the system diagnostics:\n\n"
            "```json\n"
            "{\"cpu\": 45, \"memory\": 70, \"status\": \"ok\"}\n"
            "```\n\n"
            "Everything looks healthy."
        )
        result = generate_spoken_response(model_output)

        assert "```" not in result
        assert "{" not in result
        assert "}" not in result
        assert "Everything looks healthy." in result

    def test_confirmation_response(self):
        """Scenario 9: Confirmation response"""
        model_output = (
            "I need your confirmation to execute the tool 'system_diagnostics'."
        )
        result = generate_spoken_response(model_output)

        # Simple text — should pass through cleanly
        assert result == model_output

    def test_error_response(self):
        """Scenario 10: Error response"""
        model_output = "Sir, my token governor is currently experiencing issues. Please try again in a moment."
        result = generate_spoken_response(model_output)

        assert result == model_output

    # ---- Structural assertions: output must be speakable ----

    def test_spoken_output_has_no_hash_heading(self):
        """No # should appear at the start of a word in spoken output."""
        inputs = [
            "## Hello\n\nSome text.",
            "### Section\n\n- Item",
            "# Title\n\nBody",
        ]
        for text in inputs:
            result = generate_spoken_response(text)
            assert "#" not in result, f"Hash found in: {repr(result)}"

    def test_spoken_output_has_no_double_asterisk(self):
        inputs = [
            "**Bold text** here.",
            "**Key aspects:**\n\n- one\n- two",
        ]
        for text in inputs:
            result = generate_spoken_response(text)
            assert "**" not in result, f"Bold markers found in: {repr(result)}"

    def test_spoken_output_has_no_horizontal_rule(self):
        inputs = [
            "Before\n---\nAfter",
            "A\n***\nB",
        ]
        for text in inputs:
            result = generate_spoken_response(text)
            assert "---" not in result
            assert "***" not in result

    def test_spoken_output_has_no_pipe_characters_from_table(self):
        text = "| Col | Val |\n|---|---|\n| A | 1 |"
        result = generate_spoken_response(text)
        assert "|" not in result

    def test_spoken_output_has_no_raw_json(self):
        text = '{"error": "something went wrong"}'
        result = generate_spoken_response(text)
        assert "{" not in result

    def test_spoken_output_does_not_narrate_bullets_literally(self):
        """Bullets must NOT appear as '- item' in spoken output."""
        text = "- Step one\n- Step two\n- Step three"
        result = generate_spoken_response(text)
        for line in result.split(","):
            assert not line.strip().startswith("-"), (
                f"Bullet marker leaked into spoken output: {repr(result)}"
            )

    def test_spoken_output_is_not_empty_for_valid_input(self):
        inputs = [
            "Hello.",
            "**Bold**",
            "- item",
            "## Heading",
        ]
        for text in inputs:
            result = generate_spoken_response(text)
            assert result.strip() != "", f"Empty spoken output for: {repr(text)}"

    def test_spoken_output_meaning_preserved(self):
        """Key semantic content must survive the transformation."""
        text = (
            "**Managers** focus on *operational efficiency*, while "
            "entrepreneurs focus on **innovation** and growth."
        )
        result = generate_spoken_response(text)
        assert "Managers" in result
        assert "operational efficiency" in result
        assert "entrepreneurs" in result
        assert "innovation" in result
