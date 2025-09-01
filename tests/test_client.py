"""
Tests for the main RolloutsClient class.
"""

import os
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pytest

from rollouts import RolloutsClient, create_client, Config, Response, Usage, Rollouts
from rollouts.cache import ResponseCache


class TestRolloutsClientInit:
    """Test RolloutsClient initialization."""
    
    def test_client_creation_minimal(self, mock_env_api_key):
        """Test creating client with minimal parameters."""
        client = RolloutsClient(model="test/model")
        
        assert client.config.model == "test/model"
        assert client.config.temperature == 0.7
        assert client.config.top_p == 0.95
        assert client.config.max_tokens == 4096
        assert client.config.use_cache is True
        
    def test_client_creation_full(self, mock_env_api_key):
        """Test creating client with all parameters."""
        client = RolloutsClient(
            model="test/model",
            temperature=1.2,
            top_p=0.9,
            max_tokens=2000,
            top_k=50,
            presence_penalty=0.5,
            frequency_penalty=0.3,
            provider={"order": ["openai"]},
            reasoning={"max_tokens": 1000},
            include_reasoning=True,
            api_key="custom-key",
            max_retries=50,
            timeout=600,
            verbose=True,
            use_cache=False,
            cache_dir="custom_cache",
            requests_per_minute=100
        )
        
        assert client.config.model == "test/model"
        assert client.config.temperature == 1.2
        assert client.config.top_p == 0.9
        assert client.config.max_tokens == 2000
        assert client.config.top_k == 50
        assert client.config.presence_penalty == 0.5
        assert client.config.frequency_penalty == 0.3
        assert client.config.provider == {"order": ["openai"]}
        assert client.config.reasoning == {"max_tokens": 1000}
        assert client.config.include_reasoning is True
        assert client.config.max_retries == 50
        assert client.config.timeout == 600
        assert client.config.verbose is True
        assert client.config.use_cache is False
        assert client.config.cache_dir == "custom_cache"
        assert client.config.requests_per_minute == 100
        assert client.cache is None  # use_cache=False
        assert client.rate_limiter is not None
        
    def test_client_no_model_raises_error(self):
        """Test that creating client without model raises error."""
        with pytest.raises(ValueError, match="model parameter is required"):
            RolloutsClient(model=None)
            
    def test_client_logprobs_not_supported(self, mock_env_api_key):
        """Test that logprobs raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="logprobs are not currently supported"):
            RolloutsClient(model="test", top_logprobs=5)
            
    def test_client_cache_initialization(self, mock_env_api_key, temp_cache_dir):
        """Test cache initialization."""
        client = RolloutsClient(
            model="test",
            use_cache=True,
            cache_dir=temp_cache_dir
        )
        
        assert client.cache is not None
        assert client.cache.cache_dir == temp_cache_dir
        
    def test_client_no_cache(self, mock_env_api_key):
        """Test client without cache."""
        client = RolloutsClient(
            model="test",
            use_cache=False
        )
        
        assert client.cache is None
        
    def test_client_rate_limiter(self, mock_env_api_key):
        """Test rate limiter initialization."""
        client = RolloutsClient(
            model="test",
            requests_per_minute=120
        )
        
        assert client.rate_limiter is not None
        assert client.rate_limiter.max_tokens == 120


class TestRolloutsClientGenerate:
    """Test RolloutsClient generate methods."""
    
    @pytest.mark.asyncio
    async def test_agenerate_single(self, mock_env_api_key):
        """Test async generation of single response."""
        client = RolloutsClient(model="test", use_cache=False)
        
        # Mock the provider
        mock_response = Response(
            full="Test response",
            content="Test response",
            reasoning="",
            finish_reason="stop",
            model="test",
            usage=Usage(10, 5, 15)
        )
        
        with patch.object(client.provider, 'generate_single', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_response
            
            rollouts = await client.agenerate("Test prompt", n_samples=1)
            
            assert len(rollouts) == 1
            assert rollouts[0].full == "Test response"
            assert rollouts.prompt == "Test prompt"
            assert rollouts.num_responses == 1
            mock_gen.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_agenerate_multiple(self, mock_env_api_key):
        """Test async generation of multiple responses."""
        client = RolloutsClient(model="test", use_cache=False)
        
        # Mock different responses
        responses = [
            Response(full=f"Response {i}", finish_reason="stop", usage=Usage(10, 5, 15))
            for i in range(3)
        ]
        
        with patch.object(client.provider, 'generate_single', new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = responses
            
            rollouts = await client.agenerate("Test prompt", n_samples=3)
            
            assert len(rollouts) == 3
            assert rollouts.num_responses == 3
            for i, resp in enumerate(rollouts):
                assert resp.full == f"Response {i}"
            assert mock_gen.call_count == 3
            
    @pytest.mark.asyncio
    async def test_agenerate_with_overrides(self, mock_env_api_key):
        """Test generation with parameter overrides."""
        client = RolloutsClient(
            model="test",
            temperature=0.7,
            max_tokens=100,
            use_cache=False
        )
        
        mock_response = Response(full="Test", finish_reason="stop")
        
        with patch.object(client.provider, 'generate_single', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_response
            
            rollouts = await client.agenerate(
                "Test",
                n_samples=1,
                temperature=1.5,  # Override
                max_tokens=200,   # Override
                seed=42
            )
            
            # Check that overrides were passed to provider
            call_args = mock_gen.call_args[0]
            config_used = call_args[1]
            assert config_used.temperature == 1.5
            assert config_used.max_tokens == 200
            assert config_used.seed == 42
            
    @pytest.mark.asyncio
    async def test_agenerate_with_cache_hit(self, mock_env_api_key, temp_cache_dir):
        """Test generation with cache hit."""
        client = RolloutsClient(
            model="test",
            use_cache=True,
            cache_dir=temp_cache_dir
        )
        
        # Pre-populate cache
        cached_response = Response(
            full="Cached response",
            finish_reason="stop",
            seed=0
        )
        
        client.cache.set(
            prompt="Test",
            model="test",
            provider=None,
            temperature=0.7,
            top_p=0.95,
            max_tokens=4096,
            seed=0,
            response=cached_response
        )
        
        # Generate - should use cache
        with patch.object(client.provider, 'generate_single', new_callable=AsyncMock) as mock_gen:
            rollouts = await client.agenerate("Test", n_samples=1, seed=0)
            
            assert len(rollouts) == 1
            assert rollouts[0].full == "Cached response"
            mock_gen.assert_not_called()  # Should not call API
            
    @pytest.mark.asyncio
    async def test_agenerate_skip_cached_errors(self, mock_env_api_key, temp_cache_dir):
        """Test that cached errors are regenerated."""
        client = RolloutsClient(
            model="test",
            use_cache=True,
            cache_dir=temp_cache_dir,
            verbose=True
        )
        
        # Cache an error response
        error_response = Response(
            full="Error message",
            finish_reason="error",
            seed=0
        )
        
        client.cache.set(
            prompt="Test",
            model="test",
            provider=None,
            temperature=0.7,
            top_p=0.95,
            max_tokens=4096,
            seed=0,
            response=error_response
        )
        
        # Mock successful response for regeneration
        success_response = Response(
            full="Success",
            finish_reason="stop"
        )
        
        with patch.object(client.provider, 'generate_single', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = success_response
            
            rollouts = await client.agenerate("Test", n_samples=1, seed=0)
            
            assert len(rollouts) == 1
            assert rollouts[0].full == "Success"
            mock_gen.assert_called_once()  # Should regenerate
            
    @pytest.mark.asyncio
    async def test_agenerate_logprobs_error(self, mock_env_api_key):
        """Test that logprobs in generation raises error."""
        client = RolloutsClient(model="test", use_cache=False)
        
        with pytest.raises(NotImplementedError, match="logprobs are not currently supported"):
            await client.agenerate("Test", n_samples=1, top_logprobs=5)
            
    def test_generate_sync_wrapper(self, mock_env_api_key):
        """Test synchronous generate method."""
        client = RolloutsClient(model="test", use_cache=False)
        
        mock_response = Response(full="Test", finish_reason="stop")
        
        # Mock the async method
        async def mock_agenerate(*args, **kwargs):
            return Rollouts(
                prompt="Test",
                num_responses=1,
                temperature=0.7,
                top_p=0.95,
                max_tokens=100,
                model="test",
                responses=[mock_response]
            )
        
        with patch.object(client, 'agenerate', side_effect=mock_agenerate):
            rollouts = client.generate("Test", n_samples=1)
            
            assert len(rollouts) == 1
            assert rollouts[0].full == "Test"
            
    def test_client_repr(self, mock_env_api_key):
        """Test string representation of client."""
        client = RolloutsClient(model="test/model", temperature=0.9)
        
        repr_str = repr(client)
        assert "RolloutsClient" in repr_str
        assert "model='test/model'" in repr_str
        assert "temperature=0.9" in repr_str


class TestCreateClient:
    """Test create_client factory function."""
    
    def test_create_client(self, mock_env_api_key):
        """Test factory function creates client."""
        client = create_client("test/model", temperature=0.8)
        
        assert isinstance(client, RolloutsClient)
        assert client.config.model == "test/model"
        assert client.config.temperature == 0.8
        
    def test_create_client_with_kwargs(self, mock_env_api_key):
        """Test factory with various kwargs."""
        client = create_client(
            "test",
            temperature=1.2,
            max_tokens=2000,
            use_cache=False
        )
        
        assert client.config.temperature == 1.2
        assert client.config.max_tokens == 2000
        assert client.config.use_cache is False