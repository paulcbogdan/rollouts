"""
Pytest configuration and shared fixtures for the rollouts test suite.

NOTE: This file is not meant to be run directly. It's automatically loaded by pytest.
To run tests, use: pytest (from the project root after pip install -e .)
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import pytest
import json

# Add parent directory to path for imports if running tests without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from rollouts import Response, Usage, Rollouts


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp(prefix="rollouts_test_cache_")
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_api_key():
    """Mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def sample_response():
    """Create a sample Response object for testing."""
    return Response(
        full="This is a test response.",
        content="This is a test response.",
        reasoning="",
        finish_reason="stop",
        provider=None,
        response_id="test-response-id",
        model="qwen/qwen3-30b-a3b",
        object="chat.completion",
        created=1234567890,
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        ),
        logprobs=None,
        echo=False,
        seed=42
    )


@pytest.fixture
def sample_response_with_reasoning():
    """Create a sample Response with reasoning for testing."""
    reasoning_text = "Let me think about this step by step."
    content_text = "The answer is 42."
    return Response(
        full=f"{reasoning_text}\n</think>\n{content_text}",
        content=content_text,
        reasoning=reasoning_text,
        finish_reason="stop",
        provider=None,
        response_id="test-response-id-2",
        model="deepseek/deepseek-r1",
        object="chat.completion",
        created=1234567890,
        usage=Usage(
            prompt_tokens=15,
            completion_tokens=20,
            total_tokens=35
        ),
        logprobs=None,
        echo=False,
        seed=42
    )


@pytest.fixture
def mock_httpx_client():
    """Mock httpx AsyncClient for API testing."""
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "test-id",
        "model": "qwen/qwen3-30b-a3b",
        "object": "chat.completion",
        "created": 1234567890,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Test response"
            },
            "finish_reason": "stop",
            "index": 0
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }
    mock_client.post.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_env_api_key(monkeypatch, mock_api_key):
    """Mock environment variable for API key."""
    monkeypatch.setenv("OPENROUTER_API_KEY", mock_api_key)
    return mock_api_key


@pytest.fixture
def sample_rollouts(sample_response):
    """Create a sample Rollouts object for testing."""
    return Rollouts(
        prompt="Test prompt",
        num_responses=3,
        temperature=0.7,
        top_p=0.95,
        max_tokens=100,
        model="qwen/qwen3-30b-a3b",
        responses=[sample_response, sample_response, sample_response],
        cache_dir=None,
        logprobs_enabled=False,
        echo_enabled=False
    )


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Reset rate limiter singleton
    from rollouts.rate_limiter import _rate_limiters
    _rate_limiters.clear()
    yield
    _rate_limiters.clear()