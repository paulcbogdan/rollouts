"""
Main FireworksClient for generating multiple LLM responses with logprobs support.

This client provides the same interface as RolloutsClient but uses Fireworks API
with comprehensive logprobs support for both prompt and generated tokens.
"""

import asyncio
from typing import Optional, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor

try:
    from tqdm.asyncio import tqdm as tqdm_async
except ImportError:
    # Fallback for older tqdm versions without asyncio support
    from tqdm import tqdm
    tqdm_async = tqdm  # Use regular tqdm as fallback

from .fw_datatypes import FwRollouts, FwResponse
from .fw_cache import FwResponseCacheJson
from .cache import ResponseCacheSQL
from .fireworks import Fireworks
from .rate_limiter import get_rate_limiter
from .types import ProviderConfig, ReasoningConfig, GenerationConfig


class FireworksClient:
    """
    Client for generating multiple LLM responses with Fireworks and logprobs support.

    This client provides comprehensive logprobs data for both prompt and generated tokens,
    which is the main value proposition of using Fireworks over other providers.

    Example:
        # Sync usage
        client = FireworksClient(model="qwen-15b")
        responses = client.generate("What is 2+2?", n_samples=5)

        # Access logprobs
        for response in responses:
            if response.logprobs:
                prompt_logprobs = response.get_prompt_logprobs()
                generation_logprobs = response.get_generation_logprobs()

        # Async usage
        async def main():
            client = FireworksClient(model="qwen-15b", temperature=0.9)
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
        top_k: Optional[int] = 40,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logprobs: int = 5,
        echo: bool = True,
        provider: Optional[ProviderConfig] = None,
        reasoning: Optional[ReasoningConfig] = None,
        include_reasoning: Optional[bool] = None,
        api_key: Optional[str] = None,
        max_retries: int = 100,
        timeout: int = 300,
        verbose: bool = False,
        use_cache: Union[bool, str] = "json",
        cache_dir: str = ".fw_rollouts",
        requests_per_minute: Optional[int] = None,
        progress_bar: bool = True,
        **kwargs,
    ):
        """
        Initialize the Fireworks client with default settings.

        Args:
            model: Model identifier (required) - will be converted to Fireworks format
                Examples: "qwen-15b", "llama-v3p1-8b", "gpt-oss-20b"
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter (0.0-1.0)
            max_tokens: Maximum tokens to generate
            top_k: Top-k sampling parameter (default: 40)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            logprobs: Number of logprobs to return (1-5, default: 5)
            echo: Whether to include prompt tokens in logprobs (default: True)
            provider: Provider routing preferences (dict)
            reasoning: Reasoning configuration for models that support it
            include_reasoning: Whether to include reasoning in response
            api_key: API key (uses environment variable if None)
            max_retries: Maximum retry attempts (default: 100)
            timeout: Request timeout in seconds
            verbose: Print debug information
            use_cache: Enable response caching (default: "json")
            cache_dir: Directory for cache files (default: ".fw_rollouts")
            requests_per_minute: Rate limit for API requests (None = no limit)
            progress_bar: Show progress bar for multiple samples (default: True)
            **kwargs: Additional Fireworks-specific parameters

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Store parameters as attributes
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logprobs = min(max(logprobs, 1), 5)  # Clamp to 1-5
        self.echo = echo
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

        # Initialize provider
        self._init_provider(api_key)

        # Initialize cache
        if isinstance(use_cache, str):
            if use_cache.lower() == "sql":
                self.cache = ResponseCacheSQL(cache_dir, model=self.model)
            elif use_cache.lower() == "json":
                self.cache = FwResponseCacheJson(cache_dir)
            else:
                raise ValueError(f"Invalid cache type: {use_cache}")
        elif isinstance(use_cache, bool):
            if use_cache:
                self.cache = FwResponseCacheJson(cache_dir)
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
        """Initialize Fireworks provider."""
        self.provider = Fireworks(api_key)

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

        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")

        if self.logprobs is not None and not 1 <= self.logprobs <= 5:
            raise ValueError(
                f"logprobs must be between 1 and 5, got {self.logprobs}"
            )

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
        logprobs: Optional[int] = None,
        echo: Optional[bool] = None,
        seed: Optional[int] = None,
        progress_bar: Optional[bool] = None,
        **kwargs,
    ) -> FwRollouts:
        """
        Generate multiple responses asynchronously with logprobs.

        Args:
            prompt: Input prompt
            n_samples: Number of samples to generate (default: 1)
            temperature: Override default temperature
            top_p: Override default top_p
            max_tokens: Override default max_tokens
            top_k: Override default top_k
            presence_penalty: Override default presence_penalty
            frequency_penalty: Override default frequency_penalty
            logprobs: Override default logprobs (1-5)
            echo: Override default echo mode
            seed: Starting seed for generation (Note: Fireworks doesn't use seeds directly)
            progress_bar: Override default progress_bar setting
            **kwargs: Additional parameters to override (including api_key)

        Returns:
            FwRollouts object containing all responses with logprobs
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
                "logprobs": logprobs,
                "echo": echo,
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
            "logprobs",
            "echo",
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

        # Ensure logprobs is within valid range
        if config.get("logprobs"):
            config["logprobs"] = min(max(config["logprobs"], 1), 5)

        # Collect responses
        responses = []
        tasks = []

        # Check cache and prepare tasks
        for i in range(n_samples):
            current_seed = (seed + i) if seed is not None else i

            # Check cache
            if self.cache and config["use_cache"]:
                # Include logprobs and echo in cache key
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
                    logprobs=config.get("logprobs"),
                    echo=config.get("echo"),
                )

                # Only use cached response if it's not an error and has logprobs
                if cached:
                    # Convert cached dict to FwResponse if needed
                    if isinstance(cached, dict):
                        # Reconstruct FwResponse from dict
                        from .fw_datatypes import Usage, Logprobs

                        usage_data = cached.get("usage", {})
                        usage = Usage(
                            prompt_tokens=usage_data.get("prompt_tokens", 0),
                            completion_tokens=usage_data.get("completion_tokens", 0),
                            total_tokens=usage_data.get("total_tokens", 0),
                        )

                        logprobs_data = cached.get("logprobs")
                        logprobs = None
                        if logprobs_data:
                            logprobs = Logprobs(
                                tokens=logprobs_data.get("tokens", []),
                                token_logprobs=logprobs_data.get("token_logprobs", []),
                                top_logprobs=logprobs_data.get("top_logprobs", []),
                                text_offset=logprobs_data.get("text_offset", []),
                                prompt_length=logprobs_data.get("prompt_length", 0),
                                num_prompt_tokens=logprobs_data.get("num_prompt_tokens", 0),
                            )

                        cached = FwResponse(
                            full=cached.get("full", ""),
                            content=cached.get("content", ""),
                            reasoning=cached.get("reasoning", ""),
                            text=cached.get("text", ""),
                            full_text=cached.get("full_text", ""),
                            finish_reason=cached.get("finish_reason", ""),
                            provider=cached.get("provider", "Fireworks"),
                            response_id=cached.get("response_id", ""),
                            model=cached.get("model", config["model"]),
                            object=cached.get("object", "text_completion"),
                            created=cached.get("created", 0),
                            usage=usage,
                            logprobs=logprobs,
                            echo=cached.get("echo", False),
                            seed=cached.get("seed"),
                            completed_reasoning=cached.get("completed_reasoning"),
                        )

                    if cached.finish_reason != "error":
                        # Check if logprobs are present
                        if config.get("logprobs") and not cached.logprobs:
                            if config["verbose"]:
                                print(
                                    f"Cached response for seed {current_seed} missing logprobs, regenerating..."
                                )
                        else:
                            if config["verbose"]:
                                print(f"Found cached response for seed {current_seed}")
                            responses.append(cached)
                            continue
                    elif cached.finish_reason == "error":
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
                # Check if we need logprobs and didn't get them
                if config.get("logprobs") and response.finish_reason != "error":
                    if not response.logprobs:
                        if config["verbose"] or True:  # Always warn about missing logprobs
                            print(
                                f"WARNING: Response for seed {current_seed} missing expected logprobs"
                            )

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
                            logprobs=config.get("logprobs"),
                            echo=config.get("echo"),
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
                logprobs=config.get("logprobs"),
                echo=config.get("echo"),
            )

        # Create FwRollouts
        return FwRollouts(
            prompt=prompt,
            num_responses=n_samples,
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_tokens=config["max_tokens"],
            model=config["model"],
            responses=responses,
            cache_dir=cache_dir,
            logprobs_enabled=config.get("logprobs", 0) > 0,
            echo_enabled=config.get("echo", False),
        )

    def generate(
        self,
        prompt: Union[str, List[dict]],
        n_samples: Optional[int] = None,
        progress_bar: Optional[bool] = None,
        **kwargs,
    ) -> FwRollouts:
        """
        Generate multiple responses synchronously with logprobs.

        This is a wrapper around agenerate() for users who don't want to deal with async.

        Args:
            prompt: Input prompt
            n_samples: Number of samples to generate (default: 1)
            progress_bar: Override default progress_bar setting
            **kwargs: Additional parameters (see agenerate for full list)

        Returns:
            FwRollouts object containing all responses with logprobs
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
            f"FireworksClient(model='{self.model}', "
            f"temperature={self.temperature}, "
            f"logprobs={self.logprobs})"
        )