"""
Tests for think token handling.
"""

import warnings
import pytest

from rollouts.think_handler import (
    detect_model_type,
    format_messages_with_thinking,
    format_think_messages,
    create_test_prompts,
    debug_messages,
)


class TestDetectModelType:
    """Test model type detection."""

    def test_gpt_oss_detection(self):
        """Test GPT-OSS model detection."""
        assert detect_model_type("openai/gpt-oss-20b") == "gpt-oss"
        assert detect_model_type("openai/gpt-oss-120b") == "gpt-oss"
        assert detect_model_type("GPT-OSS-20B") == "gpt-oss"

    def test_think_model_detection(self):
        """Test think-enabled model detection."""
        assert detect_model_type("qwen/qwen3-30b") == "think"
        assert detect_model_type("qwq-32b-preview") == "think"
        assert detect_model_type("claude-3-opus") == "think"
        assert detect_model_type("anthropic/claude-3") == "think"
        assert detect_model_type("deepseek/deepseek-r1") == "think"
        assert detect_model_type("deepseek-r1-distill") == "think"

    def test_gemini_thinking_detection(self):
        """Test Gemini thinking model detection."""
        assert detect_model_type("gemini-2.0-flash-thinking") == "gemini-thinking"
        assert detect_model_type("google/gemini-2.5-flash") == "gemini-thinking"
        assert detect_model_type("gemini-2.5-pro") == "gemini-thinking"

    def test_standard_model_detection(self):
        """Test standard model detection (no special handling)."""
        assert detect_model_type("gpt-3.5-turbo") == "think"
        assert detect_model_type("llama-2-70b") == "think"
        assert detect_model_type("mistral-7b") == "think"


class TestFormatMessagesWithThinking:
    """Test message formatting with thinking tokens."""

    def test_gpt_oss_no_prefill(self):
        """Test that GPT-OSS models don't support prefilling."""
        prompt = "Test <think>Thinking here"
        messages = format_messages_with_thinking(prompt, "gpt-oss-20b", verbose=False)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == prompt

    def test_think_model_no_token(self):
        """Test think model without think token."""
        prompt = "What is 2+2?"
        messages = format_messages_with_thinking(prompt, "qwen/qwen3", verbose=False)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == prompt

    def test_think_model_with_token(self):
        """Test think model with think token."""
        prompt = "What is 2+2? <think>Let me calculate"
        messages = format_messages_with_thinking(prompt, "deepseek-r1", verbose=False)

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is 2+2?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "<think>Let me calculate"

    def test_gemini_thinking_warning(self, capsys):
        """Test Gemini thinking models show note about internal handling."""
        prompt = "Test <think>Thinking"
        messages = format_messages_with_thinking(prompt, "gemini-2.5-flash", verbose=True)

        captured = capsys.readouterr()
        assert "Gemini thinking models handle reasoning internally" in captured.out


class TestFormatThinkMessages:
    """Test think message formatting."""

    def test_no_think_token(self):
        """Test formatting without think token."""
        messages = format_think_messages("Simple prompt", has_think=False)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Simple prompt"

    def test_with_think_token(self):
        """Test formatting with think token."""
        prompt = "Question here<think>My thinking process"
        messages = format_think_messages(prompt, has_think=True)

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Question here"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "<think>My thinking process"

    def test_empty_user_part(self):
        """Test formatting when user part is empty."""
        prompt = "<think>Just thinking"
        messages = format_think_messages(prompt, has_think=True)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "<think>Just thinking"

    def test_trailing_space_warning(self):
        """Test warning for trailing space in post-think text."""
        prompt = "Question<think>Thinking "  # Note trailing space

        with pytest.warns(UserWarning, match="post-<think> text has a trailing space"):
            messages = format_think_messages(prompt, has_think=True)

    def test_multiple_think_tags_warning(self):
        """Test warning for multiple think tags."""
        prompt = "First<think>Middle<think>End"

        with pytest.warns(UserWarning, match="Multiple <think> tags detected"):
            messages = format_think_messages(prompt, has_think=True)

    def test_malformed_think(self):
        """Test handling of malformed think token."""
        # This shouldn't happen as has_think would be False, but test the edge case
        prompt = "No think here"
        messages = format_think_messages(prompt, has_think=True)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == prompt


class TestCreateTestPrompts:
    """Test prompt creation for different model types."""

    def test_gpt_oss_prompts(self):
        """Test prompts for GPT-OSS models."""
        prompts = create_test_prompts("gpt-oss-20b")

        assert len(prompts) == 3
        assert all(isinstance(p, tuple) and len(p) == 2 for p in prompts)
        assert "doesn't support thinking injection" in prompts[1][0]

    def test_think_model_prompts(self):
        """Test prompts for think-enabled models."""
        prompts = create_test_prompts("deepseek-r1")

        assert len(prompts) == 3
        assert "<think>" not in prompts[0][1]  # First prompt has no thinking
        assert "<think>" in prompts[1][1]  # Second has thinking
        assert "<think>" in prompts[2][1]  # Third has complex thinking



class TestDebugMessages:
    """Test debug message printing."""

    def test_debug_messages_verbose(self, capsys):
        """Test debug messages when verbose is True."""
        messages = [
            {"role": "user", "content": "Test message"},
            {"role": "assistant", "content": "Response"},
        ]

        debug_messages(messages, verbose=True)

        captured = capsys.readouterr()
        assert "DEBUG: Formatted Messages" in captured.out
        assert "Message 1:" in captured.out
        assert "Role: user" in captured.out
        assert "Content: Test message" in captured.out
        assert "Message 2:" in captured.out
        assert "Role: assistant" in captured.out

    def test_debug_messages_not_verbose(self, capsys):
        """Test debug messages when verbose is False."""
        messages = [{"role": "user", "content": "Test"}]

        debug_messages(messages, verbose=False)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_debug_messages_truncation(self, capsys):
        """Test that long messages are truncated."""
        long_content = "x" * 300
        messages = [{"role": "user", "content": long_content}]

        debug_messages(messages, verbose=True)

        captured = capsys.readouterr()
        assert "x" * 200 + "..." in captured.out
        assert "x" * 201 not in captured.out
