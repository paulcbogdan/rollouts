"""
Tests for rate limiting functionality.
"""

import asyncio
import time
import pytest

from rollouts.rate_limiter import TokenBucketRateLimiter, get_rate_limiter


class TestTokenBucketRateLimiter:
    """Test suite for TokenBucketRateLimiter."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_creation(self):
        """Test creating a rate limiter."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        
        assert limiter.max_tokens == 60
        assert limiter.tokens == 60
        assert limiter.refill_rate == 1.0  # 60/60 = 1 per second
        
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_immediate(self):
        """Test acquiring tokens when available."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        
        # Should be able to acquire immediately
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start
        
        # Should be nearly instant
        assert elapsed < 0.1
        assert limiter.tokens == 59
        
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_wait(self):
        """Test acquiring tokens when bucket is empty."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter.tokens = 0  # Empty the bucket
        
        # Should wait for refill
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start
        
        # Should wait approximately 1 second (1 token per second)
        assert 0.9 < elapsed < 1.2
        
    @pytest.mark.asyncio
    async def test_rate_limiter_refill(self):
        """Test token refill over time."""
        limiter = TokenBucketRateLimiter(requests_per_minute=120)  # 2 per second
        limiter.tokens = 0
        
        # Wait for refill
        await asyncio.sleep(0.5)
        limiter._refill()
        
        # Should have approximately 1 token after 0.5 seconds
        assert 0.8 < limiter.tokens < 1.2
        
    @pytest.mark.asyncio
    async def test_rate_limiter_max_tokens(self):
        """Test that tokens don't exceed maximum."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter.tokens = 60
        
        # Wait and refill
        await asyncio.sleep(1)
        limiter._refill()
        
        # Should still be at max
        assert limiter.tokens == 60
        
    @pytest.mark.asyncio
    async def test_rate_limiter_multiple_requests(self):
        """Test rate limiting with multiple rapid requests."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        
        # Use up all tokens quickly
        start = time.time()
        tasks = []
        for _ in range(3):
            tasks.append(limiter.acquire())
        
        await asyncio.gather(*tasks)
        elapsed = time.time() - start
        
        # First 3 should be nearly instant (we start with 60 tokens)
        assert elapsed < 0.2
        assert limiter.tokens < 60
        
    @pytest.mark.asyncio
    async def test_high_rate_limiter(self):
        """Test with high rate limit."""
        limiter = TokenBucketRateLimiter(requests_per_minute=600)
        
        assert limiter.max_tokens == 600
        assert limiter.refill_rate == 10.0  # 10 per second
        
        # Should be able to make many requests quickly
        for _ in range(10):
            await limiter.acquire()
        
        # Use approximate equality due to floating point precision
        assert abs(limiter.tokens - 590) < 0.01


class TestGetRateLimiter:
    """Test suite for get_rate_limiter singleton function."""
    
    def test_get_rate_limiter_singleton(self):
        """Test that get_rate_limiter returns the same instance."""
        limiter1 = get_rate_limiter(60)
        limiter2 = get_rate_limiter(60)
        
        assert limiter1 is limiter2
        
    def test_get_rate_limiter_different_rates(self):
        """Test that different rates create different limiters."""
        limiter1 = get_rate_limiter(60)
        limiter2 = get_rate_limiter(120)
        
        assert limiter1 is not limiter2
        assert limiter1.max_tokens == 60
        assert limiter2.max_tokens == 120
        
    def test_get_rate_limiter_reuse(self):
        """Test that same rate reuses existing limiter."""
        limiter1 = get_rate_limiter(100)
        # Modify the limiter
        limiter1.tokens = 50
        
        # Get again with same rate
        limiter2 = get_rate_limiter(100)
        
        # Should be the same instance with modified state
        assert limiter2.tokens == 50