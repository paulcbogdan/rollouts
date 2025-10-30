"""
Extended response caching utilities for Fireworks with logprobs support.

This module extends the standard cache to include logprobs and echo parameters
in the cache key, ensuring that responses with different logprobs settings
are cached separately.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union

from .cache import ResponseCacheJson
from .fw_datatypes import FwResponse, Usage, Logprobs


class FwResponseCacheJson(ResponseCacheJson):
    """Extended cache for Fireworks responses with logprobs support.

    This cache extends the standard ResponseCacheJson to include
    logprobs and echo parameters in the cache key.
    """

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
        logprobs: Optional[int] = None,
        echo: Optional[bool] = None,
        **kwargs,  # Accept any additional parameters
    ) -> str:
        """Generate cache file path for a specific request including logprobs settings.

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
            logprobs: Number of logprobs to return (1-5)
            echo: Whether echo mode is enabled
            **kwargs: Any additional parameters

        Returns:
            Full path to the cache file for this specific request
        """
        # Clean model name for filesystem
        model_str = (
            model.replace("/", "-").replace(":", "").replace("@", "-at-")
        )

        # Hash prompt only
        if isinstance(prompt, str):
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        elif isinstance(prompt, list):
            prompt_hash = hashlib.sha256(
                json.dumps(prompt).encode("utf-8")
            ).hexdigest()
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        # Build parameter string
        param_str = f"t{temperature}_p{top_p}_tok{max_tokens}"

        # Add optional parameters if they're non-default
        if top_k is not None and top_k != 40:
            param_str += f"_tk{top_k}"
        if presence_penalty is not None and presence_penalty != 0.0:
            param_str += f"_pp{presence_penalty}"
        if frequency_penalty is not None and frequency_penalty != 0.0:
            param_str += f"_fp{frequency_penalty}"

        # Add logprobs and echo parameters
        if logprobs is not None:
            param_str += f"_lp{logprobs}"
        if echo is not None:
            param_str += f"_echo{1 if echo else 0}"

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
        logprobs: Optional[int] = None,
        echo: Optional[bool] = None,
        **kwargs,
    ) -> Optional[Union[FwResponse, dict]]:
        """Get cached response if available.

        Args:
            All parameters from _get_cache_path plus any additional kwargs

        Returns:
            Cached FwResponse object, dict, or None if not cached
        """
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
            logprobs,
            echo,
            **kwargs,
        )

        if not Path(cache_file).exists():
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            # Return the raw dict - the client will convert to FwResponse
            return data
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None

    def set(
        self,
        prompt: str,
        model: str,
        provider: Optional[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        response: FwResponse,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logprobs: Optional[int] = None,
        echo: Optional[bool] = None,
        **kwargs,
    ):
        """Cache a response.

        Args:
            All parameters from _get_cache_path plus:
            response: The FwResponse object to cache
            **kwargs: Any additional parameters
        """
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
            logprobs,
            echo,
            **kwargs,
        )

        # Convert response to dict if it's an FwResponse object
        if hasattr(response, "to_dict"):
            data = response.to_dict()
        else:
            data = response

        try:
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")

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
        logprobs: Optional[int] = None,
        echo: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """Get the cache directory for a specific configuration.

        Args:
            All parameters from _get_cache_path except seed
            **kwargs: Any additional parameters

        Returns:
            Path to the cache directory
        """
        # Use seed 0 to get the path, then extract directory
        cache_file = self._get_cache_path(
            prompt,
            model,
            provider,
            temperature,
            top_p,
            max_tokens,
            0,  # Dummy seed
            top_k,
            presence_penalty,
            frequency_penalty,
            logprobs,
            echo,
            **kwargs,
        )
        return str(Path(cache_file).parent)