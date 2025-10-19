"""
Rollouts - A high-quality Python package for generating multiple LLM responses.
"""

from .client import RolloutsClient
from .datatypes import Rollouts, Response, Usage
from .openrouter import OpenRouter
from .migrate import migrate_cache, auto_migrate

__version__ = "0.1.4"

__all__ = [
    "RolloutsClient",
    "Rollouts",
    "Response",
    "Usage",
    "OpenRouter",
    "migrate_cache",
    "auto_migrate",
]
