"""
Data models for LLM rollouts.
"""

from dataclasses import dataclass, field, asdict
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
class Response:
    """Single response from an LLM.
    
    Attributes:
        full: Complete response text (reasoning + content if applicable)
        content: Post-reasoning text (after </think>) or full text if no reasoning
        reasoning: Thinking/reasoning text (before </think>) if model supports it
        finish_reason: Why generation stopped ("stop", "length", "error", etc.)
        provider: Which provider served the request (e.g., "openai", "anthropic")
        response_id: Unique identifier for this response
        model: Actual model used (may differ from requested if fallback occurred)
        object: API response type (typically "chat.completion")
        created: Unix timestamp when response was created
        usage: Token usage statistics for this response
        logprobs: Log probabilities (currently not supported, always None)
        echo: Whether prompt was echoed in response (not supported by OpenRouter)
        seed: Random seed used for generation (if specified)
        completed_reasoning: Whether reasoning was completed with </think> tag
    """

    full: str
    content: str = ""
    reasoning: str = ""
    finish_reason: str = ""
    provider: str = ""
    response_id: str = ""
    model: str = ""
    object: str = ""
    created: int = 0
    usage: Usage = field(default_factory=Usage)
    logprobs: Optional[Dict[str, Any]] = None
    echo: bool = False
    seed: Optional[int] = None
    completed_reasoning: bool = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.usage:
            data["usage"] = asdict(self.usage)
        return data


@dataclass
class Rollouts:
    """Collection of responses for a single prompt."""

    prompt: str
    num_responses: int
    temperature: float
    top_p: float
    max_tokens: int
    model: str
    responses: List[Response]
    cache_dir: Optional[str] = None
    logprobs_enabled: bool = False
    echo_enabled: bool = False

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

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Rollouts(num_responses={self.num_responses}, "
            f"actual={len(self.responses)}, "
            f"model='{self.model}')"
        )
