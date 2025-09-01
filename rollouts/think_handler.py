"""
Handle think token insertion for different model types.

This module manages thinking/reasoning injection for various LLM models,
allowing users to provide partial reasoning that the model will continue.

Supported formats:
- Standard <think> tags: Used by Qwen, DeepSeek, Claude, etc.
- GPT-OSS Harmony format: Not currently supported via OpenRouter
- Gemini thinking: Handled internally by the model, not in messages

Key functionality:
- Detect model type from model name
- Format messages appropriately for each model type
- Handle think injection for supported models
- Provide warnings for unsupported features
"""

import warnings
from typing import List, Dict, Any, Tuple

from .types import Message


def detect_model_type(model: str) -> str:
    """Detect the type of model based on model name.

    Args:
        model: The model identifier string (e.g., "qwen/qwen3-30b-a3b")

    Returns:
        Model type string: "think", "gpt-oss", "gemini-thinking", etc.

    Supported model types:
        - "think": Models using standard <think> tags (Qwen, DeepSeek, Claude, etc.)
        - "gpt-oss": GPT-OSS models (limited support on OpenRouter)
        - "gemini-thinking": Gemini models with internal reasoning
    """
    model_lower = model.lower()

    # GPT-OSS models use Harmony format
    if "gpt-oss" in model_lower:
        return "gpt-oss"

    # Models that use standard <think> tags
    think_models = [
        "qwen",  # Qwen models (including regular and QwQ reasoning)
        "qwq",  # QwQ-32B and other QwQ reasoning models
        "claude",  # Claude models
        "anthropic",  # Anthropic models
        "deepseek",  # DeepSeek R1 and distilled variants
        "deepseek-r1",  # Explicit DeepSeek R1 models
    ]

    for m in think_models:
        if m in model_lower:
            return "think"

    # Gemini thinking models (handle reasoning differently)
    # These models have thinking but don't expose it in messages
    gemini_thinking = [
        "gemini-2.0-flash-thinking",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    for m in gemini_thinking:
        if m in model_lower:
            return "gemini-thinking"

    # Default to assuming a model using <think>
    return "think"


def format_messages_with_thinking(prompt: str, model: str, verbose: bool = False) -> List[Message]:
    """
    Format messages with thinking tokens based on model type.

    Args:
        prompt: The user's prompt, potentially containing thinking tokens
        model: The model name
        verbose: Whether to print debug info

    Returns:
        List of messages formatted for the API
    """
    model_type = detect_model_type(model)

    if verbose:
        print(f"Model type detected: {model_type} for model: {model}")

    # Check if prompt contains any thinking tokens
    has_think = "<think>" in prompt
    has_analysis = "<|channel|>analysis" in prompt

    if model_type == "gpt-oss":
        # GPT-OSS on OpenRouter doesn't support message prefilling
        # The reasoning is handled separately by OpenRouter
        if verbose and (has_think or has_analysis):
            print(f"Warning: GPT-OSS models on OpenRouter don't support thinking injection")
            print(f"         Reasoning is handled internally and returned in a separate field")
        return [{"role": "user", "content": prompt}]
    elif model_type == "think":
        return format_think_messages(prompt, has_think, verbose)
    elif model_type == "gemini-thinking":
        # Gemini models handle reasoning internally, not in messages
        if verbose and has_think:
            print(f"Note: Gemini thinking models handle reasoning internally")
            print(f"      Think injection may not work as expected")
        return [{"role": "user", "content": prompt}]
    else:
        # Standard models - no special handling
        if verbose and (has_think or has_analysis):
            print(f"Warning: Model {model} may not support thinking tokens")
        return [{"role": "user", "content": prompt}]


def format_think_messages(prompt: str, has_think: bool, verbose: bool = False) -> List[Message]:
    """
    Format messages for models that use <think> tags.

    If the prompt contains <think>, split it so the model continues from the thinking.

    Args:
        prompt: The user's prompt, potentially containing <think> tag
        has_think: Whether the prompt contains a <think> tag
        verbose: Whether to print debug information

    Returns:
        List of message dictionaries formatted for the API

    Example:
        Input: "What is 2+2? <think>Let me calculate: 2"
        Output: [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>Let me calculate: 2"}
        ]
    """
    if not has_think:
        return [{"role": "user", "content": prompt}]

    # Split on <think> to separate user part from thinking part
    if prompt.count("<think>") > 1:
        warnings.warn("Multiple <think> tags detected.", UserWarning, stacklevel=2)

    parts = prompt.split("<think>", 1)

    if len(parts) != 2:
        # Malformed, just return as is
        return [{"role": "user", "content": prompt}]

    # Check for trailing space before <think> - this can cause tokenization issues
    if parts[0] and parts[1][-1] == " ":
        warnings.warn(
            "You have included a <think> token, but your post-<think> text has a trailing space. "
            "This is likely to create tokenization issues (e.g., increases the likelihood of the next token being a number). You should probably change your prompt.",
            UserWarning,
            stacklevel=2,
        )

    user_part = parts[0].strip()
    think_part = "<think>" + parts[1]  # Keep the <think> tag

    if verbose:
        print(f"Think format detected:")
        print(f"  User part: {user_part[:50]}...")
        print(f"  Think part: {think_part[:50]}...")

    # Create messages where assistant starts with the thinking
    messages = []

    if user_part:
        messages.append({"role": "user", "content": user_part})

    # Assistant continues from the thinking
    messages.append({"role": "assistant", "content": think_part})

    return messages


# Note: GPT-OSS message formatting removed
# GPT-OSS models on OpenRouter don't support assistant message prefilling
# The reasoning is returned in a separate field by OpenRouter
# Original format_gpt_oss_messages function was here but is no longer used


def create_test_prompts(model: str) -> List[Tuple[str, str]]:
    """
    Create test prompts for different model types to verify thinking insertion.

    Returns:
        List of (description, prompt) tuples
    """
    model_type = detect_model_type(model)

    if model_type == "gpt-oss":
        return [
            ("Test 1: Simple prompt", "What is 2+2?"),
            (
                "Test 2: GPT-OSS doesn't support thinking injection",
                "What is 10*5? (Note: thinking injection not supported)",
            ),
            ("Test 3: Reasoning handled internally by OpenRouter", "Calculate 15/3"),
        ]
    elif model_type == "think":
        return [
            ("Test 1: Simple prompt (no thinking)", "What is 2+2?"),
            (
                "Test 2: Inject thinking",
                "What is 10*5? <think>The user wants me to multiply 10 by 5. That equals",
            ),
            (
                "Test 3: Continue complex thinking",
                "Explain gravity. <think>This is a physics question about gravity. Let me think through the key concepts:\n1. Gravity is a fundamental force\n2.",
            ),
        ]
    else:
        return [
            ("Test 1: Simple prompt", "What is 2+2?"),
            ("Test 2: Model doesn't support thinking", "What is 10*5?"),
        ]


def debug_messages(messages: List[Message], verbose: bool = True):
    """Print formatted messages for debugging."""
    if not verbose:
        return

    print("Messages:")
    for i, msg in enumerate(messages):
        print(f"\nMessage {i+1}:")
        print(f"  Role: {msg['role']}")
        content = msg["content"]
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"  Content: {content}")
