"""
Configuration management for rollouts.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


@dataclass
class Config:
    """Configuration for OpenRouter API requests.
    
    All parameters match the OpenRouter API specification.
    Only 'model' is required, everything else has sensible defaults.
    """
    
    # Required
    model: str = field(default=None)
    
    # Common generation parameters
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    seed: Optional[int] = None
    stream: bool = False
    
    # Sampling parameters
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: Optional[float] = None
    min_p: Optional[float] = None
    top_a: Optional[float] = None
    
    # Advanced parameters
    logit_bias: Optional[Dict[str, float]] = None
    top_logprobs: Optional[int] = None  # Not currently supported - will raise NotImplementedError
    
    # Routing parameters
    models: Optional[List[str]] = None  # Alternative models for fallback
    provider: Optional[Dict[str, Any]] = None  # Provider routing preferences
    
    # Feature flags
    reasoning: Optional[Dict[str, Any]] = None  # Reasoning/thinking tokens config
    include_reasoning: Optional[bool] = None  # Whether to include reasoning in response
    usage: Optional[Dict[str, Any]] = None  # Usage info config
    transforms: Optional[List[str]] = None  # OpenRouter transforms
    
    # User tracking (for abuse prevention)
    user: Optional[str] = None
    
    # Client behavior settings (not sent to API)
    max_retries: int = 100
    timeout: int = 300
    verbose: bool = False
    use_cache: bool = True
    cache_dir: str = "response_cache"
    requests_per_minute: Optional[int] = None  # Rate limiting (None = no limit)
    
    def __post_init__(self):
        """Validate parameters."""
        if self.model is None:
            raise ValueError("model parameter is required")
        self.validate()
    
    def validate(self):
        """Validate configuration parameters."""
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
        
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be between (0.0, 1.0], got {self.top_p}")
        
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        
        if self.frequency_penalty is not None and not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(f"frequency_penalty must be between -2.0 and 2.0, got {self.frequency_penalty}")
        
        if self.presence_penalty is not None and not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(f"presence_penalty must be between -2.0 and 2.0, got {self.presence_penalty}")
        
        if self.repetition_penalty is not None and not 0.0 < self.repetition_penalty <= 2.0:
            raise ValueError(f"repetition_penalty must be between (0, 2], got {self.repetition_penalty}")
        
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        
        if self.min_p is not None and not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be between [0, 1], got {self.min_p}")
        
        if self.top_a is not None and not 0.0 <= self.top_a <= 1.0:
            raise ValueError(f"top_a must be between [0, 1], got {self.top_a}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_api_params(self) -> Dict[str, Any]:
        """Convert to OpenRouter API parameters, excluding client-only settings."""
        params = self.to_dict()
        # Remove client-only settings
        client_only = ['max_retries', 'timeout', 'verbose', 'use_cache', 'cache_dir', 'requests_per_minute']
        for key in client_only:
            params.pop(key, None)
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return params
    
    def copy_with(self, **kwargs) -> "Config":
        """Create a copy with overridden parameters."""
        current = self.to_dict()
        current.update(kwargs)
        return Config(**current)