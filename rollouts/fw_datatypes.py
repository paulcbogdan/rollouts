"""
Data models for Fireworks LLM rollouts with logprobs support.

This module extends the standard datatypes with additional support for
logprobs data that Fireworks provides.
"""

try:
    from dataclasses import dataclass, field, asdict
except ImportError:
    # For Python < 3.7, use the backport
    from dataclasses_backport import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class Usage:
    """Token usage statistics for a single API response.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt
        completion_tokens: Number of tokens generated in the response
        total_tokens: Sum of prompt_tokens and completion_tokens
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Logprobs:
    """Log probability data for tokens in the response.

    Attributes:
        tokens: List of token strings
        token_logprobs: List of log probabilities for each token
        top_logprobs: List of dictionaries containing top alternative tokens and their logprobs
        text_offset: List of character offsets for each token in the text
        prompt_length: Length of the original prompt (if echo mode enabled)
        num_prompt_tokens: Number of tokens in the prompt (if echo mode enabled)
    """

    tokens: List[str] = field(default_factory=list)
    token_logprobs: List[float] = field(default_factory=list)
    top_logprobs: List[Dict[str, float]] = field(default_factory=list)
    text_offset: List[int] = field(default_factory=list)
    prompt_length: int = 0
    num_prompt_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def get_prompt_logprobs(self) -> Dict[str, Any]:
        """Get logprobs for just the prompt tokens."""
        if self.num_prompt_tokens == 0:
            return {
                "tokens": [],
                "token_logprobs": [],
                "top_logprobs": [],
                "text_offset": []
            }

        return {
            "tokens": self.tokens[:self.num_prompt_tokens],
            "token_logprobs": self.token_logprobs[:self.num_prompt_tokens],
            "top_logprobs": self.top_logprobs[:self.num_prompt_tokens],
            "text_offset": self.text_offset[:self.num_prompt_tokens]
        }

    def get_generation_logprobs(self) -> Dict[str, Any]:
        """Get logprobs for just the generated tokens."""
        if self.num_prompt_tokens >= len(self.tokens):
            return {
                "tokens": [],
                "token_logprobs": [],
                "top_logprobs": [],
                "text_offset": []
            }

        return {
            "tokens": self.tokens[self.num_prompt_tokens:],
            "token_logprobs": self.token_logprobs[self.num_prompt_tokens:],
            "top_logprobs": self.top_logprobs[self.num_prompt_tokens:],
            "text_offset": self.text_offset[self.num_prompt_tokens:]
        }


@dataclass
class FwResponse:
    """Single response from Fireworks with logprobs support.

    This extends the standard Response with additional fields for Fireworks-specific
    features like comprehensive logprobs data.

    Attributes:
        full: Complete response text (reasoning + content if applicable)
        content: Post-reasoning text (after </think>) or full text if no reasoning
        reasoning: Thinking/reasoning text (before </think>) if model supports it
        text: Just the generated text (without prompt if echo was enabled)
        full_text: Full text including prompt if echo mode was enabled
        finish_reason: Why generation stopped ("stop", "length", "error", etc.)
        provider: Which provider served the request (always "Fireworks")
        response_id: Unique identifier for this response
        model: Actual model used
        object: API response type (typically "text_completion")
        created: Unix timestamp when response was created
        usage: Token usage statistics for this response
        logprobs: Comprehensive log probability data for all tokens
        echo: Whether prompt was echoed in response
        seed: Random seed used for generation (if specified)
        completed_reasoning: Whether reasoning was completed with </think> tag
    """

    full: str
    content: str = ""
    reasoning: str = ""
    text: str = ""  # Just the generated text
    full_text: str = ""  # Full text including prompt if echo
    finish_reason: str = ""
    provider: str = "Fireworks"
    response_id: str = ""
    model: str = ""
    object: str = ""
    created: int = 0
    usage: Usage = field(default_factory=Usage)
    logprobs: Optional[Logprobs] = None
    echo: bool = False
    seed: Optional[int] = None
    completed_reasoning: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.usage:
            data["usage"] = self.usage.to_dict()
        if self.logprobs:
            data["logprobs"] = self.logprobs.to_dict()
        return data

    def get_prompt_logprobs(self) -> Optional[Dict[str, Any]]:
        """Get logprobs for just the prompt tokens."""
        if self.logprobs:
            return self.logprobs.get_prompt_logprobs()
        return None

    def get_generation_logprobs(self) -> Optional[Dict[str, Any]]:
        """Get logprobs for just the generated tokens."""
        if self.logprobs:
            return self.logprobs.get_generation_logprobs()
        return None


@dataclass
class FwRollouts:
    """Collection of Fireworks responses for a single prompt with logprobs support."""

    prompt: str
    num_responses: int
    temperature: float
    top_p: float
    max_tokens: int
    model: str
    responses: List[FwResponse]
    cache_dir: Optional[str] = None
    logprobs_enabled: bool = True
    echo_enabled: bool = True

    def __len__(self) -> int:
        """Get number of responses."""
        return len(self.responses)

    def __iter__(self):
        """Iterate over responses."""
        return iter(self.responses)

    def __getitem__(self, index):
        """Get response by index or slice."""
        return self.responses[index]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "num_responses": self.num_responses,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "model": self.model,
            "responses": [r.to_dict() for r in self.responses],
            "cache_dir": self.cache_dir,
            "logprobs_enabled": self.logprobs_enabled,
            "echo_enabled": self.echo_enabled,
        }

    def get_texts(self) -> List[str]:
        """Get all response texts (full responses)."""
        return [r.full for r in self.responses]

    def get_reasonings(self) -> List[str]:
        """Get all reasoning texts."""
        return [r.reasoning for r in self.responses]

    def get_contents(self) -> List[str]:
        """Get all content texts (post-reasoning)."""
        return [r.content for r in self.responses]

    def get_total_tokens(self) -> int:
        """Get total tokens used across all responses."""
        return sum(r.usage.total_tokens for r in self.responses if r.usage)

    def get_finish_reasons(self) -> Dict[str, int]:
        """Get count of finish reasons."""
        from collections import Counter
        return dict(Counter(r.finish_reason for r in self.responses if r.finish_reason))

    def get_all_logprobs(self) -> List[Optional[Logprobs]]:
        """Get logprobs for all responses."""
        return [r.logprobs for r in self.responses]

    def get_all_prompt_logprobs(self) -> List[Optional[Dict[str, Any]]]:
        """Get prompt logprobs for all responses."""
        return [r.get_prompt_logprobs() for r in self.responses]

    def get_all_generation_logprobs(self) -> List[Optional[Dict[str, Any]]]:
        """Get generation logprobs for all responses."""
        return [r.get_generation_logprobs() for r in self.responses]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FwRollouts(num_responses={self.num_responses}, "
            f"actual={len(self.responses)}, "
            f"model='{self.model}', "
            f"logprobs_enabled={self.logprobs_enabled})"
        )