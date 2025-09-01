"""
Response caching utilities.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

from .datatypes import Response, Usage


class ResponseCache:
    """Manages cached LLM responses with filesystem-based storage.

    The cache uses a hierarchical directory structure to efficiently store
    and retrieve responses based on model, parameters, and prompt.

    Cache structure:
        cache_dir/
        └── model-name/
            └── parameter-hash/
                └── prompt-hash-prefix/
                    └── prompt-hash/
                        └── seed_00000.json

    This structure allows for:
    - Easy cleanup of specific models or parameter combinations
    - Efficient filesystem navigation
    - Avoiding filesystem limitations on files per directory
    """

    def __init__(self, cache_dir: str = ".rollouts"):
        """Initialize the response cache.

        Args:
            cache_dir: Base directory for storing cached responses
        """
        self.cache_dir = cache_dir

    def _get_cache_path(
        self,
        prompt: str,
        model: str,
        provider: Optional[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> str:
        """Generate cache file path for a specific request.

        Args:
            prompt: The input prompt (hashed for privacy)
            model: Model identifier (cleaned for filesystem compatibility)
            provider: Provider routing preferences (affects cache key)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            seed: Random seed for generation
            top_k: Top-k sampling parameter
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty

        Returns:
            Full path to the cache file for this specific request

        Note:
            The prompt is SHA256 hashed to ensure privacy and avoid
            filesystem issues with special characters.
        """
        # Clean model name for filesystem
        model_str = model.replace("/", "-").replace(":", "").replace("@", "-at-")

        # Hash prompt only
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:32]

        # Build parameter string
        param_str = f"t{temperature}_p{top_p}_tok{max_tokens}"

        # Add optional parameters if they're non-default
        # Default values: top_k=40, presence_penalty=0.0, frequency_penalty=0.0
        if top_k is not None and top_k != 40:
            param_str += f"_tk{top_k}"
        if presence_penalty is not None and presence_penalty != 0.0:
            param_str += f"_pp{presence_penalty}"
        if frequency_penalty is not None and frequency_penalty != 0.0:
            param_str += f"_fp{frequency_penalty}"

        # Add provider preferences to cache path if specified
        if provider is not None:
            # Hash the provider dict for consistent cache keys
            provider_str = json.dumps(provider, sort_keys=True)
            provider_hash = hashlib.sha256(provider_str.encode()).hexdigest()[:8]
            param_str += f"_provider{provider_hash}"

        # Build cache path
        cache_path = Path(self.cache_dir) / model_str
        cache_path = cache_path / param_str
        prompt_hash_start = prompt_hash[:3]
        cache_path = cache_path / prompt_hash_start / prompt_hash

        # Create directory
        cache_path.mkdir(parents=True, exist_ok=True)

        # Return file path
        return str(cache_path / f"seed_{seed:05d}.json")

    def get(
        self,
        prompt: str,
        model: str,
        provider: Optional[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> Optional[Response]:
        """Get cached response if available."""
        cache_file = self._get_cache_path(
            prompt,
            model,
            provider,
            temperature,
            top_p,
            max_tokens,
            seed,
            top_k,
            presence_penalty,
            frequency_penalty,
        )

        if not os.path.exists(cache_file):
            return None

        with open(cache_file, "r") as f:
            data = json.load(f)

        # Convert to Response object
        if "response" not in data:
            return None

        resp_data = data["response"]

        # Create Usage object
        usage_data = resp_data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        # Create Response
        return Response(
            full=resp_data.get("full", resp_data.get("full_text", resp_data.get("text", ""))),
            content=resp_data.get("content", resp_data.get("post", "")),
            reasoning=resp_data.get("reasoning", ""),
            finish_reason=resp_data.get("finish_reason", ""),
            provider=resp_data.get("provider", provider),
            response_id=resp_data.get("response_id", ""),
            model=resp_data.get("model", model),
            object=resp_data.get("object", ""),
            created=resp_data.get("created", 0),
            usage=usage,
            logprobs=resp_data.get("logprobs"),
            echo=resp_data.get("echo", False),
            seed=seed,
        )

    def set(
        self,
        prompt: str,
        model: str,
        provider: Optional[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        response: Response,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> bool:
        """Cache a response."""
        cache_file = self._get_cache_path(
            prompt,
            model,
            provider,
            temperature,
            top_p,
            max_tokens,
            seed,
            top_k,
            presence_penalty,
            frequency_penalty,
        )

        # Prepare cache data
        cache_data = {
            "seed": seed,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "model": model,
            "provider": provider,
            "response": response.to_dict(),
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        return True

    def get_cache_dir(
        self,
        prompt: str,
        model: str,
        provider: Optional[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> str:
        """Get the cache directory for a given configuration."""
        model_str = model.replace("/", "-").replace(":", "").replace("@", "-at-")

        # Hash prompt only
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:32]

        # Build parameter string
        param_str = f"t{temperature}_p{top_p}_tok{max_tokens}"

        # Add optional parameters if they're non-default
        if top_k is not None and top_k != 40:
            param_str += f"_tk{top_k}"
        if presence_penalty is not None and presence_penalty != 0.0:
            param_str += f"_pp{presence_penalty}"
        if frequency_penalty is not None and frequency_penalty != 0.0:
            param_str += f"_fp{frequency_penalty}"

        # Add provider preferences to cache path if specified
        if provider is not None:
            # Hash the provider dict for consistent cache keys
            provider_str = json.dumps(provider, sort_keys=True)
            provider_hash = hashlib.sha256(provider_str.encode()).hexdigest()[:8]
            param_str += f"_provider{provider_hash}"

        cache_path = Path(self.cache_dir) / model_str
        cache_path = cache_path / param_str
        # Use same two-level structure as _get_cache_path for consistency
        prompt_hash_start = prompt_hash[:3]
        cache_path = cache_path / prompt_hash_start / prompt_hash

        return str(cache_path)
