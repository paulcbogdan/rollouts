"""
Main RolloutsClient for generating multiple LLM responses.
"""

import asyncio
from typing import Optional, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor

from .config import Config
from .datatypes import Rollouts, Response
from .cache import ResponseCache
from .openrouter import OpenRouter
from .rate_limiter import get_rate_limiter


class RolloutsClient:
    """
    Client for generating multiple LLM responses with built-in resampling.

    Example:
        # Sync usage
        client = RolloutsClient(model="qwen/qwen3-30b-a3b")
        responses = client.generate("What is 2+2?", n_samples=5)

        # Async usage
        async def main():
            client = RolloutsClient(model="qwen/qwen3-30b-a3b", temperature=0.9)
            responses = await client.agenerate("What is 2+2?", n_samples=5)

            # Multiple prompts concurrently
            results = await asyncio.gather(
                client.agenerate("prompt1", n_samples=3),
                client.agenerate("prompt2", n_samples=3, temperature=1.2)
            )
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        top_k: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        provider: Optional[Dict[str, Any]] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        include_reasoning: Optional[bool] = None,
        api_key: Optional[str] = None,
        max_retries: int = 100,
        timeout: int = 300,
        verbose: bool = False,
        use_cache: bool = True,
        cache_dir: str = "response_cache",
        requests_per_minute: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the client with default settings.

        Args:
            model: Model identifier (required)
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            top_k: Top-k sampling parameter (None for no limit)
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            provider: Provider routing preferences (dict)
            reasoning: Reasoning configuration for models that support it
                e.g., {"max_tokens": 2000} or {"effort": "low"}
            include_reasoning: Whether to include reasoning in response
            api_key: API key (uses environment variable if None)
            max_retries: Maximum retry attempts (default: 100)
            timeout: Request timeout in seconds
            verbose: Print debug information
            use_cache: Enable response caching
            cache_dir: Directory for cache files
            requests_per_minute: Rate limit for API requests (None = no limit)
        """
        # Create config
        self.config = Config(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            provider=provider,
            reasoning=reasoning,
            include_reasoning=include_reasoning,
            max_retries=max_retries,
            timeout=timeout,
            verbose=verbose,
            use_cache=use_cache,
            cache_dir=cache_dir,
            requests_per_minute=requests_per_minute,
            **kwargs,
        )

        # Check for unsupported features
        if self.config.top_logprobs is not None and self.config.top_logprobs > 0:
            raise NotImplementedError(
                "logprobs are not currently supported. OpenRouter's implementation "
                "of logprobs appears inconsistent across providers, so this feature "
                "has not been implemented in this package."
            )

        # Initialize provider
        self._init_provider(api_key)

        # Initialize cache
        self.cache = ResponseCache(cache_dir) if use_cache else None

        # Initialize rate limiter if specified
        self.rate_limiter = None
        if requests_per_minute is not None:
            self.rate_limiter = get_rate_limiter(requests_per_minute)

        # For sync wrapper
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _init_provider(self, api_key: Optional[str] = None):
        """Initialize OpenRouter provider."""
        self.provider = OpenRouter(api_key)

    async def agenerate(
        self,
        prompt: str,
        n_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Rollouts:
        """
        Generate multiple responses asynchronously.

        Args:
            prompt: Input prompt
            n_samples: Number of samples to generate (default: 1)
            temperature: Override default temperature
            top_p: Override default top_p
            max_tokens: Override default max_tokens
            top_k: Override default top_k
            presence_penalty: Override default presence_penalty
            frequency_penalty: Override default frequency_penalty
            seed: Starting seed for generation
            **kwargs: Additional parameters to override (including api_key)

        Returns:
            Rollouts object containing all responses
        """
        n_samples = n_samples or 1

        # Extract api_key separately (don't include in config)
        api_key = kwargs.pop("api_key", None)

        # Create config with overrides
        overrides = {
            k: v
            for k, v in {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "top_k": top_k,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "seed": seed,
                **kwargs,
            }.items()
            if v is not None
        }

        config = self.config.copy_with(**overrides) if overrides else self.config

        # Check for unsupported features
        if config.top_logprobs is not None and config.top_logprobs > 0:
            raise NotImplementedError(
                "logprobs are not currently supported. OpenRouter's implementation "
                "of logprobs appears inconsistent across providers, so this feature "
                "has not been implemented in this package."
            )

        # Collect responses
        responses = []
        tasks = []

        # Check cache and prepare tasks
        for i in range(n_samples):
            current_seed = (seed + i) if seed is not None else i

            # Check cache
            if self.cache and config.use_cache:
                cached = self.cache.get(
                    prompt=prompt,
                    model=config.model,
                    provider=config.provider,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_tokens=config.max_tokens,
                    seed=current_seed,
                    top_k=config.top_k,
                    presence_penalty=config.presence_penalty,
                    frequency_penalty=config.frequency_penalty,
                )

                # Only use cached response if it's not an error
                if cached and cached.finish_reason != "error":
                    if config.verbose:
                        print(f"Found cached response for seed {current_seed}")
                    responses.append(cached)
                    continue
                elif cached and cached.finish_reason == "error":
                    if config.verbose:
                        print(f"Found cached error for seed {current_seed}, regenerating...")

            # Add generation task
            tasks.append(
                (
                    current_seed,
                    self.provider.generate_single(
                        prompt, config, current_seed, api_key, self.rate_limiter
                    ),
                )
            )

        # Execute tasks concurrently
        if tasks:
            results = await asyncio.gather(*[task for _, task in tasks])

            for (current_seed, _), response in zip(tasks, results):
                if response.finish_reason != "error":
                    # Cache successful response
                    if self.cache and config.use_cache:
                        self.cache.set(
                            prompt=prompt,
                            model=config.model,
                            provider=config.provider,
                            temperature=config.temperature,
                            top_p=config.top_p,
                            max_tokens=config.max_tokens,
                            seed=current_seed,
                            response=response,
                            top_k=config.top_k,
                            presence_penalty=config.presence_penalty,
                            frequency_penalty=config.frequency_penalty,
                        )
                    responses.append(response)
                elif config.verbose:
                    print(f"Error generating response for seed {current_seed}: {response.full}")

        # Get cache directory
        cache_dir = None
        if self.cache:
            cache_dir = self.cache.get_cache_dir(
                prompt=prompt,
                model=config.model,
                provider=config.provider,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                top_k=config.top_k,
                presence_penalty=config.presence_penalty,
                frequency_penalty=config.frequency_penalty,
            )

        # Create Rollouts
        return Rollouts(
            prompt=prompt,
            num_responses=n_samples,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            model=config.model,
            responses=responses,
            cache_dir=cache_dir,
            logprobs_enabled=False,  # Not supported - will error earlier if requested
            echo_enabled=False,  # OpenRouter doesn't support echo mode
        )

    def generate(self, prompt: str, n_samples: Optional[int] = None, **kwargs) -> Rollouts:
        """
        Generate multiple responses synchronously.

        This is a wrapper around agenerate() for users who don't want to deal with async.

        Args:
            prompt: Input prompt
            n_samples: Number of samples to generate (default: 1)
            **kwargs: Additional parameters (see agenerate for full list)

        Returns:
            Rollouts object containing all responses
        """
        # Run async function in sync context

        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if loop and loop.is_running():
            # We're already in an async context, use thread pool
            future = self._executor.submit(asyncio.run, self.agenerate(prompt, n_samples, **kwargs))
            return future.result()
        else:
            # No async context, run directly
            return asyncio.run(self.agenerate(prompt, n_samples, **kwargs))

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RolloutsClient(model='{self.config.model}', "
            f"temperature={self.config.temperature})"
        )


def create_client(model: str, **kwargs) -> RolloutsClient:
    """
    Factory function to create a client.

    Args:
        model: Model identifier
        **kwargs: Additional parameters

    Returns:
        Configured RolloutsClient

    Example:
        client = create_client("qwen/qwen3-30b-a3b", temperature=0.9)
    """
    return RolloutsClient(model=model, **kwargs)
