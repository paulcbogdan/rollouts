#!/usr/bin/env python
"""
Example usage of the FireworksClient for generating LLM responses with logprobs support.

This example demonstrates how to use the Fireworks API integration to get
comprehensive logprobs data for both prompt and generated tokens.

Note: You'll need to set the FIREWORKS_API_KEY environment variable before running.
"""

import asyncio
import os
from rollouts.fw_client import FireworksClient


async def main():
    # Set your Fireworks API key (or use environment variable)
    # os.environ['FIREWORKS_API_KEY'] = 'your-key-here'

    # Create a client with Fireworks
    client = FireworksClient(
        model="glm-4p5",  # Using Qwen 1.5B model accounts/fireworks/models/
        temperature=0.7,
        max_tokens=100,
        logprobs=5,  # Get top 5 logprobs for each token
        echo=False,  # Include prompt tokens in logprobs
        reasoning_effort="none",
    )

    # Generate multiple responses
    prompt = "What is the capital of France?"

    print(f"Prompt: {prompt}\n")
    print("Generating responses...")
    prompt = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "321 * 13 = "},
    ]

    # Generate 3 responses with logprobs
    rollouts = await client.agenerate(
        prompt,
        n_samples=1,
    )

    # Display results
    for i, response in enumerate(rollouts.responses, 1):
        print(f"\n--- Response {i} ---")
        print(f"Full text: {response.full}")

        # If the response has reasoning (for reasoning models)
        if response.reasoning:
            print(f"Reasoning: {response.reasoning}")
            print(f"Content: {response.content}")

        # Show token usage
        print(
            f"Tokens used: {response.usage.total_tokens} "
            f"(prompt: {response.usage.prompt_tokens}, "
            f"completion: {response.usage.completion_tokens})"
        )

        # Show logprobs information if available
        if response.logprobs:
            print("\nLogprobs data available:")
            print(f"  - Total tokens: {len(response.logprobs.tokens)}")
            print(f"  - Prompt tokens: {response.logprobs.num_prompt_tokens}")

            # Get logprobs for just the generated tokens
            gen_logprobs = response.get_generation_logprobs()
            if gen_logprobs and gen_logprobs["tokens"]:
                print("\n  Generated tokens (first 5):")
                for j, (token, logprob) in enumerate(
                    zip(
                        gen_logprobs["tokens"][:5],
                        gen_logprobs["token_logprobs"][:5],
                    )
                ):
                    print(f"    {j+1}. '{token}': {logprob:.4f}")

            # Show top alternatives for first generated token
            if gen_logprobs and gen_logprobs["top_logprobs"]:
                print("\n  Top alternatives for first generated token:")
                first_alternatives = gen_logprobs["top_logprobs"][0]
                for token, logprob in list(first_alternatives.items())[:5]:
                    print(f"    '{token}': {logprob:.4f}")

    # Show cache information
    print(f"\n\nResponses cached in: {rollouts.cache_dir}")


def sync_example():
    """Synchronous example using the generate method."""
    client = FireworksClient(
        model="accounts/fireworks/models/gpt-oss-20b",
        temperature=0.7,
        logprobs=5,
        echo=True,
    )

    # Sync version
    rollouts = client.generate(
        "Explain quantum computing in simple terms", n_samples=2
    )

    for response in rollouts:
        print(f"Response: {response.full[:100]}...")
        if response.logprobs:
            print(f"Logprobs available: {len(response.logprobs.tokens)} tokens")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("FIREWORKS_API_KEY"):
        print("Warning: FIREWORKS_API_KEY not set in environment")
        print("Please set it before running:")
        print("  export FIREWORKS_API_KEY='your-key-here'")
        print()

    # Run async example
    print("=== Async Example ===")
    try:
        asyncio.run(main())
    except AttributeError:
        # For Python 3.6
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())

    print("\n" + "=" * 50)
    print("\n=== Sync Example ===")
    # Uncomment to run sync example
    # sync_example()
