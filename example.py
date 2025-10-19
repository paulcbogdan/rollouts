#!/usr/bin/env python3
"""
Rollouts Package Examples - Learn How to Use the Package

This script demonstrates the main features of the rollouts package.
Make sure you have set your API key: export OPENROUTER_API_KEY="your-key-here"

Run this script: python example.py
"""

import asyncio
import os
import time
from rollouts import RolloutsClient


def example_0_reasoning_model():
    """Example 0: Basic usage with a reasoning model."""
    print("\n" + "=" * 50)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 50)

    # Create a client for a basic model
    client = RolloutsClient(
        model="qwen/qwen3-30b-a3b",
        temperature=0.7,
        max_tokens=200,  # A good, fast reasoning model
    )

    # Generate a single response
    prompt = "Explain what a neural network is in simple terms."
    rollouts = client.generate(prompt, n_samples=1)

    print(f"Prompt: {prompt}")
    print(f"Response: {rollouts[0].full=}")
    print(f"Reasoning: {rollouts[0].reasoning=}")
    print(f"Content: {rollouts[0].content=}")
    print(f"Tokens used: {rollouts[0].usage.total_tokens}")


def example_1_non_reasoning():
    """Example 1: Basic usage with a simple model."""
    print("\n" + "=" * 50)
    print("EXAMPLE 1: Basic Usage with a non-reasoning model")
    print("=" * 50)

    # Create a client for a basic model
    client = RolloutsClient(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0.7,
        max_tokens=200,  # A good, fast model
    )

    # Generate a single response
    prompt = "Explain what a neural network is in simple terms."
    rollouts = client.generate(prompt, n_samples=1)

    print(f"Prompt: {prompt}")
    print(f"Response: {rollouts[0].full=}")
    print(f"Reasoning: {rollouts[0].reasoning=}")
    print(f"Content: {rollouts[0].content=}")
    print(f"Tokens used: {rollouts[0].usage.total_tokens}")


def example_2_multiple_samples():
    """Example 2: Generate multiple responses to see variety."""
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Multiple Samples")
    print("=" * 50)

    client = RolloutsClient(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0.9,  # Higher temperature for more variety
        max_tokens=100,
    )

    prompt = "Write a creative opening line for a story."
    rollouts = client.generate(prompt, n_samples=3)

    print(f"Prompt: {prompt}")
    print(f"Generated {len(rollouts)} responses:")

    for i, response in enumerate(rollouts, 1):
        print(f"\n{i}. {response.full=}")


def example_3_think_injection():
    """Example 3: Using think injection for reasoning models."""
    print("\n" + "=" * 50)
    print("EXAMPLE 3: Think Injection (Reasoning Models)")
    print("=" * 50)

    # Use a reasoning model that supports think injection
    client = RolloutsClient(
        model="qwen/qwen3-30b-a3b",
        temperature=0.7,
        max_tokens=300,  # Reasoning model
    )

    # Inject partial thinking to guide the model
    prompt = "What is 127 * 43? <think>I need to multiply 127"
    rollouts = client.generate(prompt, n_samples=1)

    response = rollouts[0]

    print(f"Prompt: {prompt}")
    print(f"\nFull Response: {response.full=}")
    print(
        f'\nReasoning Only (should continue "I need to multiply 127..."): {response.reasoning=}'
    )
    print(f"\nContent Only: {response.content=}")


def example_4_parameter_overrides():
    """Example 4: Override parameters at generation time."""
    print("\n" + "=" * 50)
    print("EXAMPLE 4: Parameter Overrides")
    print("=" * 50)

    # Create client with default settings
    client = RolloutsClient(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0.7,  # Default temperature
        max_tokens=100,  # Default max tokens
    )

    prompt = "Write a haiku about programming."

    # First generation with defaults
    rollouts1 = client.generate(prompt, n_samples=1)

    # Second generation with overridden parameters
    rollouts2 = client.generate(
        prompt,
        n_samples=1,
        temperature=1.2,  # More creative
        max_tokens=200,  # Longer response
        seed=42,  # Reproducible
    )

    print(f"Prompt: {prompt}")
    print(f"\nDefault settings: {rollouts1[0].full=}")
    print(f"\nOverridden settings: {rollouts2[0].full=}")


async def example_5_async_usage():
    """Example 5: Asynchronous usage for better performance."""
    print("\n" + "=" * 50)
    print("EXAMPLE 5: Async Usage")
    print("=" * 50)

    client = RolloutsClient(
        model="qwen/qwen-2.5-7b-instruct", temperature=0.8, max_tokens=150
    )

    # Generate responses for multiple prompts concurrently
    prompts = [
        "What is machine learning?",
        "Explain quantum computing.",
        "What is the future of AI?",
    ]

    # Run all generations concurrently
    results = await asyncio.gather(
        *[client.agenerate(prompt, n_samples=1) for prompt in prompts]
    )

    print("Generated responses for 3 prompts concurrently:")
    for i, (prompt, rollouts) in enumerate(zip(prompts, results), 1):
        print(f"\n{i}. {prompt}")
        print(f"   → {rollouts[0].full=}")


def example_6_different_models():
    """Example 6: Trying different model types."""
    print("\n" + "=" * 50)
    print("EXAMPLE 6: Different Model Types")
    print("=" * 50)

    # Different models for different use cases
    models = [
        ("qwen/qwen-2.5-7b-instruct", "Fast, general purpose"),
        ("deepseek/deepseek-r1", "Reasoning and math"),
        ("anthropic/claude-3-haiku", "Fast, good quality"),
    ]

    prompt = "What is 15 * 24?"

    for model, description in models:
        print(f"\n--- {model} ({description}) ---")

        try:
            client = RolloutsClient(model=model, max_tokens=200)
            rollouts = client.generate(prompt, n_samples=1)
            print(f"Response: {rollouts[0].full=}")
        except Exception as e:
            print(f"Error: {e}")


def example_7_caching():
    """Example 7: Demonstrate response caching."""
    print("\n" + "=" * 50)
    print("EXAMPLE 7: Response Caching")
    print("=" * 50)

    client = RolloutsClient(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0.7,
        max_tokens=100,
        use_cache=True,  # Default, but shown explicitly
        verbose=True,  # Show cache hits
        progress_bar=False,  # Disable progress bar for this example
    )

    prompt = "What is the capital of France?"

    print("First call (will make API request):")
    t_start = time.time()
    rollouts1 = client.generate(prompt, n_samples=1, seed=42)
    t_end = time.time()
    print(
        f"Time taken (should take a while if not cached): {t_end - t_start:.5f} seconds"
    )

    t_start = time.time()
    print("\nSecond call (should use cache):")
    rollouts2 = client.generate(prompt, n_samples=1, seed=42)
    t_end = time.time()
    print(f"Time taken (should be very fast): {t_end - t_start:.5f} seconds")

    # Responses should be identical due to caching
    print(f"Same response? {rollouts1[0].full == rollouts2[0].full=}")


def example_8_provider_routing():
    """Example 8: Using provider routing to select specific AI providers."""
    print("\n" + "=" * 50)
    print("EXAMPLE 8: Provider Routing")
    print("=" * 50)

    # Prefer specific providers or exclude others
    client = RolloutsClient(
        model="meta-llama/llama-3.1-8b-instruct",  # Available from multiple providers
        provider={
            "order": [
                "groq",
                "together",
                "deepinfra",
            ],  # Prefer these providers in order
            # You can also exclude providers:
            # "ignore": ["openai", "anthropic"]
        },
        temperature=0.7,
        max_tokens=100,
    )

    prompt = "What is Python?"
    rollouts = client.generate(prompt, n_samples=1)

    print(f"Prompt: {prompt}")
    print(
        f"Response from provider '{rollouts[0].provider}': {rollouts[0].full=}"
    )
    print("\nNote: Provider routing allows you to:")
    print("- Prefer specific providers with 'order'")
    print("- Exclude providers with 'ignore'")
    print("- Control costs and latency by choosing optimal providers")


def example_9_rate_limiting():
    """Example 9: Using rate limiting to control API request speed."""
    print("\n" + "=" * 50)
    print("EXAMPLE 9: Rate Limiting")
    print("=" * 50)

    # Limit to 60 requests per minute (1 per second)
    client = RolloutsClient(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0.7,
        max_tokens=50,
        requests_per_minute=60,  # Rate limit
        verbose=True,  # Show when rate limiting occurs
    )

    prompts = [
        "Count to 3",
        "Name a color",
        "Say hello",
    ]

    print("Generating 3 responses with rate limiting (60 RPM)...")
    print("This ensures we don't exceed API rate limits")

    for i, prompt in enumerate(prompts, 1):
        t_start = time.time()
        rollouts = client.generate(prompt, n_samples=1)
        t_end = time.time()
        print(f"\n{i}. Prompt: '{prompt}' (took {t_end - t_start:.2f}s)")
        print(f"   Response: {rollouts[0].full}")


def example_10_advanced_parameters():
    """Example 10: Using advanced sampling parameters."""
    print("\n" + "=" * 50)
    print("EXAMPLE 10: Advanced Parameters")
    print("=" * 50)

    # Use additional OpenRouter parameters for fine control
    client = RolloutsClient(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0.9,
        max_tokens=100,
        # Advanced sampling parameters
        repetition_penalty=1.1,  # Reduce repetition
        min_p=0.05,  # Minimum probability threshold
        top_a=0.8,  # Top-a sampling
    )

    prompt = (
        "Write a short story about a robot. Be creative and avoid repetition."
    )
    rollouts = client.generate(prompt, n_samples=1)

    print(f"Prompt: {prompt}")
    print(f"\nResponse with advanced parameters:")
    print(f"{rollouts[0].full}")
    print("\nAdvanced parameters used:")
    print("- repetition_penalty=1.1 (reduces repetition)")
    print("- min_p=0.05 (filters low probability tokens)")
    print("- top_a=0.8 (alternative to top_p sampling)")


def example_11_progress_bar():
    """Example 11: Demonstrate progress bar functionality."""
    print("\n" + "=" * 50)
    print("EXAMPLE 11: Progress Bar")
    print("=" * 50)

    client = RolloutsClient(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0.7,
        max_tokens=50,
        progress_bar=True,  # Enabled by default
    )

    print("Generating multiple responses with progress bar:")
    print("(Progress bar appears for n_samples > 1)")

    # This will show a progress bar
    rollouts = client.generate(
        "Write a one-line joke", n_samples=5
    )  # Progress bar will appear

    print(f"\nGenerated {len(rollouts)} responses")
    for i, response in enumerate(rollouts, 1):
        print(f"{i}. {response.full[:50]}...")

    print("\nSingle response (no progress bar):")
    single = client.generate("Quick fact", n_samples=1)
    print(f"Response: {single[0].full[:100]}...")

    print("\nDisabling progress bar for a specific request:")
    quiet_rollouts = client.generate(
        "Another joke", n_samples=3, progress_bar=False  # Override to disable
    )
    print(f"Generated {len(quiet_rollouts)} responses silently")


def example_12_reasoning_config():
    """Example 12: Configure reasoning for models that support it."""
    print("\n" + "=" * 50)
    print("EXAMPLE 12: Reasoning Configuration")
    print("=" * 50)

    # Configure reasoning behavior
    client = RolloutsClient(
        model="deepseek/deepseek-r1",  # A reasoning model
        temperature=0.7,
        max_tokens=500,
        reasoning={
            "max_tokens": 1000,  # Limit reasoning tokens
            # Some models support "effort" parameter:
            # "effort": "low"  # or "medium", "high"
        },
        include_reasoning=True,  # Explicitly include reasoning in response
    )

    prompt = (
        "Solve: If a train travels 60 mph for 2.5 hours, how far does it go?"
    )

    try:
        rollouts = client.generate(prompt, n_samples=1)
        response = rollouts[0]

        print(f"Prompt: {prompt}")
        if response.reasoning:
            print(f"\nReasoning process:")
            print(f"{response.reasoning}")
        print(f"\nFinal answer:")
        print(f"{response.content}")
    except Exception as e:
        print(f"Note: This example requires a reasoning model. Error: {e}")


def example_13_conversation():
    client = RolloutsClient(
        model="qwen/qwen3-30b-a3b",
        temperature=0.7,
        max_tokens=501,  # High enough number to finish reasoning
        verbose=True,
        provider={"ignore": ["SiliconFlow"]},
    )

    messages = []
    messages.append({"role": "user", "content": "Pick a number from 1 to 100?"})
    messages.append({"role": "assistant", "content": "I pick the number 42."})
    messages.append(
        {"role": "user", "content": "Say again, what number did you pick?"}
    )

    rollouts = client.generate(messages, n_samples=1)

    print(f"Prompt: {messages}")
    print(f"{rollouts[0]=}")


def main():
    """Run all examples."""
    print("ROLLOUTS PACKAGE EXAMPLES")
    print(
        "Learn how to use the rollouts package for generating multiple LLM responses"
    )

    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n❌ ERROR: OPENROUTER_API_KEY not set!")
        print("Please set your API key:")
        print("   export OPENROUTER_API_KEY='your-key-here'")
        return

    print(
        f"\n✓ API key found: {'*' * 20}{os.getenv('OPENROUTER_API_KEY')[-4:]}"
    )

    example_0_reasoning_model()
    example_1_non_reasoning()
    example_2_multiple_samples()
    example_3_think_injection()
    example_4_parameter_overrides()
    asyncio.run(example_5_async_usage())
    example_6_different_models()
    example_7_caching()
    example_8_provider_routing()
    example_9_rate_limiting()
    example_10_advanced_parameters()
    example_11_progress_bar()
    example_12_reasoning_config()
    example_13_conversation()

    print("\n" + "=" * 50)
    print("✅ ALL EXAMPLES COMPLETED!")
    print("=" * 50)
    print("\nKey takeaways:")
    print("1. Use RolloutsClient(model='...') for basic usage")
    print("2. Generate multiple samples with n_samples parameter")
    print("3. Use <think> injection for reasoning models")
    print("4. Override parameters at generation time")
    print("5. Use async methods for better performance")
    print("6. Different models have different strengths")
    print("7. Caching saves API costs for repeated requests")
    print("8. Provider routing gives control over which AI provider to use")
    print("9. Rate limiting helps avoid hitting API limits")
    print("10. Advanced parameters provide fine control over generation")
    print("11. Progress bar shows generation status for multiple samples")
    print("12. Reasoning can be configured for supported models")

    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
