"""
Type definitions for the rollouts package.
"""

from typing import TypedDict, Optional, List, Any, Dict


class ProviderConfig(TypedDict, total=False):
    """Provider routing configuration."""
    order: List[str]  # Preferred providers in order
    ignore: List[str]  # Providers to exclude
    allow_fallbacks: bool  # Whether to allow fallback providers
    require_parameters: bool  # Whether to require parameter support


class ReasoningConfig(TypedDict, total=False):
    """Reasoning configuration for models that support it."""
    max_tokens: int  # Maximum tokens for reasoning
    effort: str  # Effort level: "low", "medium", "high"


class GenerationConfig(TypedDict, total=False):
    """Configuration for generating LLM responses."""
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    top_k: Optional[int]
    presence_penalty: float
    frequency_penalty: float
    provider: Optional[ProviderConfig]
    reasoning: Optional[ReasoningConfig]
    include_reasoning: Optional[bool]
    max_retries: int
    timeout: int
    verbose: bool
    use_cache: bool
    cache_dir: str
    requests_per_minute: Optional[int]
    # Additional OpenRouter parameters
    min_p: Optional[float]
    top_a: Optional[float]
    repetition_penalty: Optional[float]
    logit_bias: Optional[Dict[int, float]]
    stop: Optional[List[str]]
    response_format: Optional[Dict[str, Any]]
    seed: Optional[int]


class Message(TypedDict):
    """Chat message format."""
    role: str  # "user", "assistant", or "system"
    content: str  # Message content


class APIResponse(TypedDict):
    """OpenRouter API response format."""
    id: str
    model: str
    object: str
    created: int
    provider: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]