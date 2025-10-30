"""
Fireworks provider implementation with logprobs support.

This module provides a drop-in replacement for openrouter.py but uses Fireworks API
with full logprobs support including echo mode (getting logprobs for prompt tokens).

Key features:
- Full logprobs support with echo mode
- Automatic caching of responses (same structure as openrouter.py)
- Concurrent request handling
- Exponential backoff retry logic
- Comprehensive error handling
- Async interface matching openrouter.py

Main differences from openrouter.py:
- Uses Fireworks completions API (not chat) for echo mode support
- Returns logprobs for both prompt and generated tokens
- Maximum 5 alternatives per token (Fireworks limit)
"""

import os
import time
import asyncio
from typing import Optional, Dict, Any, List
import httpx
import json

from .fw_datatypes import FwResponse, Usage, Logprobs
from .types import GenerationConfig, APIResponse
from .think_handler import format_messages_with_thinking, debug_messages


class Fireworks:
    """Fireworks API provider for LLM generation with logprobs support.

    This class handles direct communication with the Fireworks API,
    including request formatting, error handling, response parsing, and logprobs extraction.

    Attributes:
        api_key: The Fireworks API key
        api_url: The Fireworks API endpoint URL
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Fireworks provider.

        Args:
            api_key: Fireworks API key. If not provided, uses FIREWORKS_API_KEY
                environment variable.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Fireworks API key not found. Set FIREWORKS_API_KEY environment variable or pass api_key parameter."
            )
        # self.api_url = "https://api.fireworks.ai/inference/v1/completions"
        self.api_url = "https://api.fireworks.ai/inference/v1/chat/completions"

    def _convert_model_name(self, model: str) -> str:
        """Convert model name to Fireworks format.

        Args:
            model: Model identifier string

        Returns:
            Converted model name for Fireworks API
        """
        # Special case mappings
        model_map = {
            "qwen-15b": "accounts/fireworks/models/deepseek-r1-distill-qwen-1p5b",
            "llama-v3p1-8b": "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "gpt-oss-20b": "accounts/fireworks/models/gpt-oss-20b",
        }

        if model in model_map:
            return model_map[model]

        # Convert from OpenRouter format to Fireworks format
        # e.g., "qwen/qwen3-30b-a3b" -> "accounts/fireworks/models/qwen3-30b-a3b"
        if "/" in model and not model.startswith("accounts/"):
            model_parts = model.split("/")
            if len(model_parts) == 2:
                return f"accounts/fireworks/models/{model_parts[1]}"
        elif not model.startswith("accounts/"):
            # Add default Fireworks prefix
            return f"accounts/fireworks/models/{model}"

        return model

    def _format_prompt_from_messages(self, messages: List[dict]) -> str:
        """Convert chat messages to a single prompt string for completions API.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            Formatted prompt string
        """
        # For Fireworks completions API, we need to convert messages to a single prompt
        # We'll format it as a chat conversation for best results
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        # Join with newlines and add final Assistant: to prompt continuation
        prompt = "\n\n".join(prompt_parts)
        if messages and messages[-1].get("role") != "assistant":
            prompt += "\n\nAssistant:"

        return prompt

    async def generate_single(
        self,
        prompt: str,
        config: GenerationConfig,
        seed: Optional[int] = None,
        api_key: Optional[str] = None,
        rate_limiter=None,
    ) -> FwResponse:
        """Generate a single response from the Fireworks API with logprobs.

        Args:
            prompt: The input prompt to send to the model
            config: Configuration dictionary containing model parameters
                (model, temperature, max_tokens, etc.)
            seed: Optional random seed for reproducible generation (Note: Fireworks doesn't support seeds in completions API)
            api_key: Optional API key to override instance key
            rate_limiter: Optional rate limiter to control request frequency

        Returns:
            FwResponse object containing the generated text, metadata, and logprobs

        Note:
            This method includes automatic retry logic with exponential backoff
            for handling transient errors and rate limits.
        """
        # Use provided API key, fallback to instance key, then error if none
        key = api_key or self.api_key
        if not key:
            raise ValueError(
                "No API key provided. Pass api_key parameter or set FIREWORKS_API_KEY",
            )

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        # Format messages with model-specific thinking support
        messages = format_messages_with_thinking(
            prompt, config["model"], config["verbose"]
        )

        # Show formatted messages if verbose
        if config["verbose"]:
            debug_messages(messages, verbose=True)

        # Convert messages to a single prompt for completions API
        # formatted_prompt = self._format_prompt_from_messages(messages)

        # Convert model name to Fireworks format
        fw_model = self._convert_model_name(config["model"])

        # Extract parameters for Fireworks API
        # Note: Fireworks completions API has different parameter support than chat API
        payload = {
            "model": fw_model,
            "messages": messages,
            # "prompt": formatted_prompt,
            "max_tokens": config.get("max_tokens", 4096),
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.95),
            "logprobs": config.get("logprobs", 5),  # Number of logprobs (max 5)
            "echo": config.get(
                "echo", True
            ),  # Include prompt tokens for their logprobs
            "stream": False,
            "reasoning_effort": "none",
        }
        print(f"{config=}")

        # Add optional parameters if present
        if config.get("top_k") is not None:
            payload["top_k"] = config["top_k"]
        if config.get("presence_penalty") is not None:
            payload["presence_penalty"] = config["presence_penalty"]
        if config.get("frequency_penalty") is not None:
            payload["frequency_penalty"] = config["frequency_penalty"]

        # Note: Fireworks doesn't support seed parameter in completions API
        # Seeds are used for cache organization only

        # Retry logic
        retry_delay = 2
        for attempt in range(config.get("max_retries", 100)):
            try:
                # Apply rate limiting if configured
                if rate_limiter:
                    await rate_limiter.acquire()

                async with httpx.AsyncClient(
                    timeout=config.get("timeout", 300)
                ) as client:
                    response = await client.post(
                        self.api_url, headers=headers, json=payload
                    )

                    if response.status_code in [500, 429]:
                        error_type = (
                            "Server error"
                            if response.status_code == 500
                            else "Rate limit"
                        )
                        if config["verbose"] or attempt == 0:
                            print(
                                f"{error_type} on attempt {attempt+1}/{config.get('max_retries', 100)}"
                            )

                        delay = min(retry_delay * (2**attempt), 60)
                        await asyncio.sleep(delay)
                        continue

                    elif response.status_code != 200:
                        if config["verbose"] or attempt == 0:
                            print(
                                f"API error ({response.status_code}) on attempt {attempt+1}/{config.get('max_retries', 100)}. Returned json:\n{response.json()}"
                            )
                        if attempt == config.get("max_retries", 100) - 1:
                            return self._error_response(
                                f"API error: {response.status_code}",
                                config["model"],
                            )
                        delay = min(retry_delay * (2**attempt), 60)
                        await asyncio.sleep(delay)
                        continue

                    try:
                        result = response.json()
                    except json.decoder.JSONDecodeError:
                        if config["verbose"]:
                            print(
                                f"JSON decode error on attempt {attempt+1}/{config.get('max_retries', 100)}. Returned response:\n{response}"
                            )
                        if attempt == config.get("max_retries", 100) - 1:
                            return self._error_response(
                                f"JSON decode error: {response.status_code}",
                                config["model"],
                            )
                        delay = min(retry_delay * (2**attempt), 60)
                        continue

                    return self._parse_response(
                        result,
                        config["model"],
                        seed,
                        prompt,
                        config.get("echo", True),
                    )

            except (
                httpx.RequestError,
                httpx.TimeoutException,
                httpx.HTTPStatusError,
            ) as e:
                if config["verbose"]:
                    print(f"Request error on attempt {attempt+1}: {e}")

                if attempt == config.get("max_retries", 100) - 1:
                    return self._error_response(str(e), config["model"])

                delay = min(retry_delay * (2**attempt), 60)
                await asyncio.sleep(delay)

        return self._error_response("Max retries exceeded", config["model"])

    def _parse_response(
        self,
        result: Dict[str, Any],
        model: str,
        seed: Optional[int],
        prompt: str,
        echo: bool,
    ) -> FwResponse:
        """Parse API response into FwResponse object with logprobs.

        Args:
            result: Raw JSON response from Fireworks API
            model: Model identifier used for the request
            seed: Random seed used for generation (if any)
            prompt: Original prompt sent to the model
            echo: Whether echo mode was enabled

        Returns:
            FwResponse object with parsed content, reasoning, metadata, and logprobs

        Note:
            Handles both reasoning and non-reasoning model responses,
            properly splitting content and reasoning text when present,
            and extracting comprehensive logprobs data.
        """
        if "choices" not in result or len(result["choices"]) == 0:
            return self._error_response("No choices in response", model)

        choice = result["choices"][0]
        text = choice.get("text", "")
        finish_reason = choice.get("finish_reason", "")

        # Extract logprobs data
        logprobs_data = choice.get("logprobs", None)

        # Parse logprobs into structured format
        structured_logprobs = None
        if logprobs_data:
            tokens = logprobs_data.get("tokens", [])
            token_logprobs = logprobs_data.get("token_logprobs", [])
            top_logprobs = logprobs_data.get("top_logprobs", [])
            text_offset = logprobs_data.get("text_offset", [])

            # Identify where prompt ends (if echo is enabled)
            prompt_len = len(prompt) if echo else 0

            structured_logprobs = Logprobs(
                tokens=tokens,
                token_logprobs=token_logprobs,
                top_logprobs=top_logprobs,
                text_offset=text_offset,
                prompt_length=prompt_len,
                num_prompt_tokens=(
                    sum(1 for offset in text_offset if offset < prompt_len)
                    if echo
                    else 0
                ),
            )

        # Extract the generated text (remove prompt if echo was enabled)
        if echo and text.startswith(prompt):
            generated_text = text[len(prompt) :]
        else:
            generated_text = text

        # Handle reasoning/content split
        # Check if the generated text contains thinking tags
        reasoning_text = ""
        content_text = generated_text
        completed_reasoning = True

        if "<think>" in generated_text or "</think>" in generated_text:
            # Split on </think> if it exists
            if "</think>" in generated_text:
                parts = generated_text.split("</think>", 1)
                reasoning_text = parts[0].replace("<think>", "").strip()
                content_text = parts[1].strip() if len(parts) > 1 else ""
                completed_reasoning = True
            else:
                # Has <think> but no </think> - reasoning not completed
                reasoning_text = generated_text.replace("<think>", "").strip()
                content_text = ""
                completed_reasoning = False

        # Reconstruct full text
        if reasoning_text and content_text:
            full = f"{reasoning_text}</think>{content_text}"
        elif reasoning_text:
            full = reasoning_text
        else:
            full = content_text

        # Extract usage
        usage_data = result.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return FwResponse(
            full=full,
            content=content_text,
            reasoning=reasoning_text,
            text=generated_text,  # Just the generated text
            full_text=(
                text if echo else generated_text
            ),  # Full text including prompt if echo
            finish_reason=finish_reason,
            provider="Fireworks",
            response_id=result.get("id", ""),
            model=result.get("model", model),
            object=result.get("object", "text_completion"),
            created=result.get("created", int(time.time())),
            usage=usage,
            logprobs=structured_logprobs,
            echo=echo,
            seed=seed,
            completed_reasoning=completed_reasoning,
        )

    def _error_response(self, error_msg: str, model: str) -> FwResponse:
        """Create an error response object.

        Args:
            error_msg: Error message describing what went wrong
            model: Model identifier that was requested

        Returns:
            FwResponse object with finish_reason="error" and error message in full field

        Note:
            Error responses are not cached and will be regenerated on retry.
        """
        return FwResponse(
            full=f"Error: {error_msg}",
            content="",
            reasoning="",
            text="",
            full_text="",
            finish_reason="error",
            provider="Fireworks",
            response_id="",
            model=model,
            object="error",
            created=int(time.time()),
            usage=Usage(),
            logprobs=None,
            echo=False,
            seed=None,
            completed_reasoning=None,
        )
