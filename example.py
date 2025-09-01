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
        model="qwen/qwen3-30b-a3b", temperature=0.7, max_tokens=200  # A good, fast reasoning model
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
        model="qwen/qwen-2.5-7b-instruct", temperature=0.7, max_tokens=200  # A good, fast model
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
        model="qwen/qwen3-30b-a3b", temperature=0.7, max_tokens=300  # Reasoning model
    )

    # Inject partial thinking to guide the model
    prompt = "What is 127 * 43? <think>I need to multiply 127"
    rollouts = client.generate(prompt, n_samples=1)

    response = rollouts[0]

    print(f"Prompt: {prompt}")
    print(f"\nFull Response: {response.full=}")
    print(f'\nReasoning Only (should continue "I need to multiply 127..."): {response.reasoning=}')
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

    client = RolloutsClient(model="qwen/qwen-2.5-7b-instruct", temperature=0.8, max_tokens=150)

    # Generate responses for multiple prompts concurrently
    prompts = [
        "What is machine learning?",
        "Explain quantum computing.",
        "What is the future of AI?",
    ]

    # Run all generations concurrently
    results = await asyncio.gather(*[client.agenerate(prompt, n_samples=1) for prompt in prompts])

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
    )

    prompt = "What is the capital of France?"

    print("First call (will make API request):")
    t_start = time.time()
    rollouts1 = client.generate(prompt, n_samples=1, seed=42)
    t_end = time.time()
    print(f"Time taken (should take a while if not cached): {t_end - t_start:.5f} seconds")

    t_start = time.time()
    print("\nSecond call (should use cache):")
    rollouts2 = client.generate(prompt, n_samples=1, seed=42)
    t_end = time.time()
    print(f"Time taken (should be very fast): {t_end - t_start:.5f} seconds")

    # Responses should be identical due to caching
    print(f"Same response? {rollouts1[0].full == rollouts2[0].full=}")


def main():
    """Run all examples."""
    print("ROLLOUTS PACKAGE EXAMPLES")
    print("Learn how to use the rollouts package for generating multiple LLM responses")

    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n❌ ERROR: OPENROUTER_API_KEY not set!")
        print("Please set your API key:")
        print("   export OPENROUTER_API_KEY='your-key-here'")
        return

    print(f"\n✓ API key found: {'*' * 20}{os.getenv('OPENROUTER_API_KEY')[-4:]}")

    # Run synchronous examples
    try:
        example_0_reasoning_model()
        example_1_non_reasoning()
        example_2_multiple_samples()
        example_3_think_injection()
        example_4_parameter_overrides()
        example_6_different_models()
        example_7_caching()

        # Run async example
        print("\nRunning async example...")
        asyncio.run(example_5_async_usage())

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("This might be due to:")
        print("- Invalid API key")
        print("- Network issues")
        print("- Model not available")
        return

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

    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
