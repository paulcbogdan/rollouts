"""
Global rate limiter for API calls.

This module provides a singleton rate limiter that can be shared across
all async operations to ensure we stay within API rate limits.
"""

import asyncio
import time
from typing import Optional


class TokenBucketRateLimiter:
    """A token bucket rate limiter for controlling API request rates."""
    
    def __init__(self, requests_per_minute: int = 550):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum number of requests per minute
        """
        self.rpm = requests_per_minute
        self.requests_per_second = requests_per_minute / 60.0
        self.min_interval = 1.0 / self.requests_per_second
        
        # Token bucket parameters
        self.max_tokens = requests_per_minute  # Allow full burst
        self.tokens = self.max_tokens
        self.refill_rate = requests_per_minute / 60.0  # Tokens per second
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()
        
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        time_passed = now - self.last_update
        
        # Add tokens based on time passed
        self.tokens = min(
            self.max_tokens,
            self.tokens + time_passed * self.refill_rate
        )
        self.last_update = now
    
    async def acquire(self):
        """
        Acquire permission to make a request. This will wait if necessary
        to maintain the rate limit.
        """
        async with self.lock:
            while True:
                self._refill()
                
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                
                # Calculate wait time until we have a token
                wait_time = (1 - self.tokens) / self.refill_rate
                await asyncio.sleep(wait_time)
    
    def reset(self, requests_per_minute: int):
        """Reset the rate limiter with a new RPM limit."""
        self.__init__(requests_per_minute)


# Global singleton instance
_global_rate_limiter: Optional[TokenBucketRateLimiter] = None

# Singleton storage for multiple rate limiters (used by get_rate_limiter)
_rate_limiters = {}


def get_rate_limiter(requests_per_minute: int = 550) -> TokenBucketRateLimiter:
    """
    Get or create a rate limiter singleton for the given RPM.
    
    Args:
        requests_per_minute: RPM limit
        
    Returns:
        A TokenBucketRateLimiter instance for the given RPM
    """
    global _rate_limiters
    if requests_per_minute not in _rate_limiters:
        _rate_limiters[requests_per_minute] = TokenBucketRateLimiter(requests_per_minute)
    return _rate_limiters[requests_per_minute]


def set_rate_limit(requests_per_minute: int):
    """
    Set or update the global rate limit.
    
    Args:
        requests_per_minute: New RPM limit
    """
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = TokenBucketRateLimiter(requests_per_minute)
    else:
        _global_rate_limiter.reset(requests_per_minute)