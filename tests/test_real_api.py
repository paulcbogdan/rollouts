"""
Real API integration tests (requires OPENROUTER_API_KEY).
Run with: pytest -m integration
Skip with: pytest -m "not integration"
"""

import os
import pytest
from rollouts import RolloutsClient


@pytest.mark.integration
class TestRealOpenRouterAPI:
    """Integration tests that make real API calls to OpenRouter."""

    def test_real_api_call_tiny_prompt(self):
        """Test a real API call with minimal tokens and no caching."""
        # Skip if no API key
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set - skipping real API test")

        # Use a fast, cheap model with minimal settings
        client = RolloutsClient(
            model="qwen/qwen3-30b-a3b",  # Small, fast, cheap model
            temperature=0.0,  # Deterministic
            max_tokens=5,  # Tiny response
            use_cache=False,  # Force real API call
            verbose=True,  # Show what's happening
        )

        # Tiny prompt
        prompt = "Hi"

        # Make the call
        rollouts = client.generate(prompt, n_samples=1)

        # Verify we got a real response
        assert len(rollouts) == 1
        response = rollouts[0]

        # Basic response validation
        assert response.full is not None
        assert len(response.full) > 0
        assert response.usage.total_tokens > 0
        assert response.finish_reason in ["stop", "length"]
        assert response.model.startswith("qwen")

        print(f"✅ Real API call succeeded!")
        print(f"   Model: {response.model}")
        print(f"   Response: {response.full!r}")
        print(f"   Tokens: {response.usage.total_tokens}")
        print(f"   Finish reason: {response.finish_reason}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_async_api_call(self):
        """Test async API call with real OpenRouter."""
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set - skipping real API test")

        client = RolloutsClient(
            model="qwen/qwen3-30b-a3b", temperature=0.0, max_tokens=3, use_cache=False, verbose=True
        )

        # Make async call
        rollouts = await client.agenerate("Count: 1", n_samples=1)

        assert len(rollouts) == 1
        response = rollouts[0]
        assert response.full is not None
        assert response.usage.total_tokens > 0

        print(f"✅ Async API call succeeded!")
        print(f"   Response: {response.full!r}")
