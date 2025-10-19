"""
Main RolloutsClient for generating multiple LLM responses.
"""

import asyncio
from typing import Optional, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor

from tqdm.asyncio import tqdm as tqdm_async

from .datatypes import Rollouts, Response
from .cache import ResponseCacheJson, ResponseCacheSQL
from .openrouter import OpenRouter
from .rate_limiter import get_rate_limiter
from .types import ProviderConfig, ReasoningConfig, GenerationConfig


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
        provider: Optional[ProviderConfig] = None,
        reasoning: Optional[ReasoningConfig] = None,
        include_reasoning: Optional[bool] = None,
        api_key: Optional[str] = None,
        max_retries: int = 100,
        timeout: int = 300,
        verbose: bool = False,
        use_cache: Union[bool, str] = "sql",
        cache_dir: str = ".rollouts",
        requests_per_minute: Optional[int] = None,
        progress_bar: bool = True,
        **kwargs,
    ):
        """
        Initialize the client with default settings.

        Args:
            model: Model identifier (required)
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter (0.0-1.0)
            max_tokens: Maximum tokens to generate
            top_k: Top-k sampling parameter (None for no limit)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            provider: Provider routing preferences (dict)
                e.g., {"order": ["anthropic", "openai"]} or {"ignore": ["meta"]}
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
            progress_bar: Show progress bar for multiple samples (default: True)
            **kwargs: Additional OpenRouter-specific parameters such as:
                - min_p (float): Minimum probability threshold (0.0-1.0)
                - top_a (float): Top-a sampling parameter (0.0-1.0)
                - repetition_penalty (float): Penalize repetition (0.0-2.0)
                - logit_bias (dict): Token ID to bias mapping
                - stop (list): Stop sequences
                - response_format (dict): Format constraints

        Raises:
            ValueError: If required parameters are missing or invalid
            NotImplementedError: If logprobs are requested (not supported)
        """
        # Store parameters as attributes
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.provider_config = provider  # Store provider configuration
        self.reasoning = reasoning
        self.include_reasoning = include_reasoning
        self.max_retries = max_retries
        self.timeout = timeout
        self.verbose = verbose
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.requests_per_minute = requests_per_minute
        self.progress_bar = progress_bar

        # Additional parameters from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Validate parameters
        self._validate_params()

        # Check for unsupported features
        if (
            getattr(self, "top_logprobs", None) is not None
            and self.top_logprobs > 0
        ):
            raise NotImplementedError(
                "logprobs are not currently supported. OpenRouter's implementation "
                "of logprobs appears inconsistent across providers, so this feature "
                "has not been implemented in this package."
            )

        # Initialize provider
        self._init_provider(api_key)

        # Initialize cache
        if isinstance(use_cache, str):
            if use_cache.lower() == "sql":
                self.cache = ResponseCacheSQL(cache_dir, model=self.model)
            elif use_cache.lower() == "json":
                self.cache = ResponseCacheJson(cache_dir)
            else:
                raise ValueError(f"Invalid cache type: {use_cache}")
        elif isinstance(use_cache, bool):
            if use_cache:
                self.cache = ResponseCacheJson(cache_dir)
            else:
                self.cache = None
        else:
            self.cache = None

        # Initialize rate limiter if specified
        self.rate_limiter = None
        if requests_per_minute is not None:
            self.rate_limiter = get_rate_limiter(requests_per_minute)

        # For sync wrapper
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _init_provider(self, api_key: Optional[str] = None):
        """Initialize OpenRouter provider."""
        self.provider = OpenRouter(api_key)

    def _validate_params(self):
        """Validate configuration parameters."""
        if self.model is None:
            raise ValueError("model parameter is required")

        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )

        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError(
                f"top_p must be between (0.0, 1.0], got {self.top_p}"
            )

        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be positive, got {self.max_tokens}"
            )

        if (
            self.frequency_penalty is not None
            and not -2.0 <= self.frequency_penalty <= 2.0
        ):
            raise ValueError(
                f"frequency_penalty must be between -2.0 and 2.0, got {self.frequency_penalty}"
            )

        if (
            self.presence_penalty is not None
            and not -2.0 <= self.presence_penalty <= 2.0
        ):
            raise ValueError(
                f"presence_penalty must be between -2.0 and 2.0, got {self.presence_penalty}"
            )

        repetition_penalty = getattr(self, "repetition_penalty", None)
        if (
            repetition_penalty is not None
            and not 0.0 < repetition_penalty <= 2.0
        ):
            raise ValueError(
                f"repetition_penalty must be between (0, 2], got {repetition_penalty}"
            )

        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")

        min_p = getattr(self, "min_p", None)
        if min_p is not None and not 0.0 <= min_p <= 1.0:
            raise ValueError(f"min_p must be between [0, 1], got {min_p}")

        top_a = getattr(self, "top_a", None)
        if top_a is not None and not 0.0 <= top_a <= 1.0:
            raise ValueError(f"top_a must be between [0, 1], got {top_a}")

    async def agenerate(
        self,
        prompt: Union[str, List[dict]],
        n_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        progress_bar: Optional[bool] = None,
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
            progress_bar: Override default progress_bar setting
            **kwargs: Additional parameters to override (including api_key)

        Returns:
            Rollouts object containing all responses
        """
        n_samples = n_samples or 1

        if seed is not None:
            assert n_samples == 1, "Cannot specify seed and n_samples > 1"

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
                "progress_bar": progress_bar,
                **kwargs,
            }.items()
            if v is not None
        }

        # Create merged config from instance attributes and overrides
        config = {}

        # Copy all instance attributes
        for attr in [
            "model",
            "temperature",
            "top_p",
            "max_tokens",
            "top_k",
            "presence_penalty",
            "frequency_penalty",
            "reasoning",
            "include_reasoning",
            "max_retries",
            "timeout",
            "verbose",
            "use_cache",
            "cache_dir",
            "requests_per_minute",
            "progress_bar",
        ]:
            if hasattr(self, attr):
                config[attr] = getattr(self, attr)

        # Add provider_config as 'provider' in the config dict
        if hasattr(self, "provider_config"):
            config["provider"] = getattr(self, "provider_config")

        # Add any additional kwargs that were set during init
        # Exclude client-only objects like provider, cache, rate_limiter, _executor
        excluded_attrs = {"provider", "cache", "rate_limiter", "_executor"}
        for attr_name in dir(self):
            if (
                not attr_name.startswith("_")
                and not callable(getattr(self, attr_name))
                and attr_name not in config
                and attr_name not in excluded_attrs
            ):
                config[attr_name] = getattr(self, attr_name)

        # Apply overrides
        config.update(overrides)

        # Check for unsupported features
        if (
            config.get("top_logprobs") is not None
            and config["top_logprobs"] > 0
        ):
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
            if self.cache and config["use_cache"]:
                cached = self.cache.get(
                    prompt=prompt,
                    model=config["model"],
                    provider=config["provider"],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    max_tokens=config["max_tokens"],
                    seed=current_seed,
                    top_k=config["top_k"],
                    presence_penalty=config["presence_penalty"],
                    frequency_penalty=config["frequency_penalty"],
                )

                # Only use cached response if it's not an error
                if cached and cached.finish_reason != "error":
                    if config["verbose"]:
                        print(f"Found cached response for seed {current_seed}")
                    responses.append(cached)
                    continue
                elif cached and cached.finish_reason == "error":
                    if config["verbose"]:
                        print(
                            f"Found cached error for seed {current_seed}, regenerating..."
                        )

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
            # Determine if we should show a progress bar
            show_progress = config.get("progress_bar", True) and n_samples > 1

            if show_progress:
                # Use tqdm for progress tracking
                results = [None] * len(tasks)

                # Create list of tasks with their indices
                indexed_tasks = [
                    (i, task) for i, (seed, task) in enumerate(tasks)
                ]

                # Create progress bar
                pbar = tqdm_async(
                    total=len(tasks),
                    desc=f"Generating {len(tasks)} response{'s' if len(tasks) > 1 else ''}",
                    leave=False,  # Auto-delete progress bar when done
                    unit="response",
                    colour="green",
                )

                # Create wrapper coroutines that update progress
                async def run_with_progress(index, task):
                    result = await task
                    pbar.update(1)
                    return index, result

                # Run all tasks with progress tracking
                completed = await asyncio.gather(
                    *[run_with_progress(i, task) for i, task in indexed_tasks]
                )

                # Sort results back to original order
                for idx, result in completed:
                    results[idx] = result

                pbar.close()
            else:
                # No progress bar for single sample or if disabled
                results = await asyncio.gather(*[task for _, task in tasks])

            for (current_seed, _), response in zip(tasks, results):
                if response.finish_reason != "error":
                    # Cache successful response
                    if self.cache and config["use_cache"]:
                        self.cache.set(
                            prompt=prompt,
                            model=config["model"],
                            provider=config["provider"],
                            temperature=config["temperature"],
                            top_p=config["top_p"],
                            max_tokens=config["max_tokens"],
                            seed=current_seed,
                            response=response,
                            top_k=config["top_k"],
                            presence_penalty=config["presence_penalty"],
                            frequency_penalty=config["frequency_penalty"],
                        )
                    responses.append(response)
                elif config["verbose"]:
                    print(
                        f"Error generating response for seed {current_seed}: {response.full}"
                    )

        # Get cache directory
        cache_dir = None
        if self.cache:
            cache_dir = self.cache.get_cache_dir(
                prompt=prompt,
                model=config["model"],
                provider=config["provider"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                max_tokens=config["max_tokens"],
                top_k=config["top_k"],
                presence_penalty=config["presence_penalty"],
                frequency_penalty=config["frequency_penalty"],
            )

        # Create Rollouts
        return Rollouts(
            prompt=prompt,
            num_responses=n_samples,
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_tokens=config["max_tokens"],
            model=config["model"],
            responses=responses,
            cache_dir=cache_dir,
            logprobs_enabled=False,  # Not supported - will error earlier if requested
            echo_enabled=False,  # OpenRouter doesn't support echo mode
        )

    def generate(
        self,
        prompt: Union[str, List[dict]],
        n_samples: Optional[int] = None,
        progress_bar: Optional[bool] = None,
        **kwargs,
    ) -> Rollouts:
        """
        Generate multiple responses synchronously.

        This is a wrapper around agenerate() for users who don't want to deal with async.

        Args:
            prompt: Input prompt
            n_samples: Number of samples to generate (default: 1)
            progress_bar: Override default progress_bar setting
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
            future = self._executor.submit(
                asyncio.run,
                self.agenerate(
                    prompt, n_samples, progress_bar=progress_bar, **kwargs
                ),
            )
            return future.result()
        else:
            # No async context, run directly
            return asyncio.run(
                self.agenerate(
                    prompt, n_samples, progress_bar=progress_bar, **kwargs
                )
            )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RolloutsClient(model='{self.model}', "
            f"temperature={self.temperature})"
        )
