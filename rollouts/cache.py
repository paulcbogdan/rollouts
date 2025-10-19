"""
Response caching utilities.
SQLite-based response caching replaces the file-based cache with a more scalable SQLite database.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

from .datatypes import Response, Usage

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
import time


class ResponseCacheJson:
    """Manages cached LLM responses with filesystem-based storage.

    The cache uses a hierarchical directory structure to efficiently store
    and retrieve responses based on model, parameters, and prompt.

    Cache structure:
        cache_dir/
        └── model-name/
            └── parameter-hash/
                └── prompt-hash-prefix/
                    └── prompt-hash/
                        └── seed_00000.json

    This structure allows for:
    - Easy cleanup of specific models or parameter combinations
    - Efficient filesystem navigation
    - Avoiding filesystem limitations on files per directory
    """

    def __init__(self, cache_dir: str = ".rollouts"):
        """Initialize the response cache.

        Args:
            cache_dir: Base directory for storing cached responses
        """
        self.cache_dir = cache_dir

    def _get_cache_path(
        self,
        prompt: str,
        model: str,
        provider: Optional[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> str:
        """Generate cache file path for a specific request.

        Args:
            prompt: The input prompt (hashed for privacy)
            model: Model identifier (cleaned for filesystem compatibility)
            provider: Provider routing preferences (affects cache key)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            seed: Random seed for generation
            top_k: Top-k sampling parameter
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty

        Returns:
            Full path to the cache file for this specific request

        Note:
            The prompt is SHA256 hashed to ensure privacy and avoid
            filesystem issues with special characters.
        """
        # Clean model name for filesystem
        model_str = (
            model.replace("/", "-").replace(":", "").replace("@", "-at-")
        )

        # Hash prompt only
        if isinstance(prompt, str):
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        elif isinstance(prompt, list):
            prompt_hash = hashlib.sha256(
                json.dumps(prompt).encode("utf-8")
            ).hexdigest()
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        # Build parameter string
        param_str = f"t{temperature}_p{top_p}_tok{max_tokens}"

        # Add optional parameters if they're non-default
        # Default values: top_k=40, presence_penalty=0.0, frequency_penalty=0.0
        if top_k is not None and top_k != 40:
            param_str += f"_tk{top_k}"
        if presence_penalty is not None and presence_penalty != 0.0:
            param_str += f"_pp{presence_penalty}"
        if frequency_penalty is not None and frequency_penalty != 0.0:
            param_str += f"_fp{frequency_penalty}"

        # Add provider preferences to cache path if specified
        if provider is not None:
            # Hash the provider dict for consistent cache keys
            provider_str = json.dumps(provider, sort_keys=True)
            provider_hash = hashlib.sha256(provider_str.encode()).hexdigest()[
                :8
            ]
            param_str += f"_provider{provider_hash}"

        # Build cache path
        cache_path = Path(self.cache_dir) / model_str
        cache_path = cache_path / param_str
        prompt_hash_start = prompt_hash[:3]
        cache_path = cache_path / prompt_hash_start / prompt_hash

        # Create directory
        cache_path.mkdir(parents=True, exist_ok=True)

        # Return file path
        return str(cache_path / f"seed_{seed:05d}.json")

    def get(
        self,
        prompt: str,
        model: str,
        provider: Optional[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> Optional[Response]:
        """Get cached response if available."""
        cache_file = self._get_cache_path(
            prompt,
            model,
            provider,
            temperature,
            top_p,
            max_tokens,
            seed,
            top_k,
            presence_penalty,
            frequency_penalty,
        )

        if not os.path.exists(cache_file):
            return None

        with open(cache_file, "r") as f:
            data = json.load(f)

        # Convert to Response object
        if "response" not in data:
            return None

        resp_data = data["response"]

        # Create Usage object
        usage_data = resp_data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        # Create Response
        return Response(
            full=resp_data.get(
                "full", resp_data.get("full_text", resp_data.get("text", ""))
            ),
            content=resp_data.get("content", resp_data.get("post", "")),
            reasoning=resp_data.get("reasoning", ""),
            finish_reason=resp_data.get("finish_reason", ""),
            provider=resp_data.get("provider", provider),
            response_id=resp_data.get("response_id", ""),
            model=resp_data.get("model", model),
            object=resp_data.get("object", ""),
            created=resp_data.get("created", 0),
            usage=usage,
            logprobs=resp_data.get("logprobs"),
            echo=resp_data.get("echo", False),
            seed=seed,
        )

    def set(
        self,
        prompt: str,
        model: str,
        provider: Optional[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        response: Response,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> bool:
        """Cache a response."""
        cache_file = self._get_cache_path(
            prompt,
            model,
            provider,
            temperature,
            top_p,
            max_tokens,
            seed,
            top_k,
            presence_penalty,
            frequency_penalty,
        )

        # Prepare cache data
        cache_data = {
            "seed": seed,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "model": model,
            "provider": provider,
            "response": response.to_dict(),
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        return True

    def get_cache_dir(
        self,
        prompt: str,
        model: str,
        provider: Optional[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> str:
        """Get the cache directory for a given configuration."""
        model_str = (
            model.replace("/", "-").replace(":", "").replace("@", "-at-")
        )

        # Hash prompt only
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:32]

        # Build parameter string
        param_str = f"t{temperature}_p{top_p}_tok{max_tokens}"

        # Add optional parameters if they're non-default
        if top_k is not None and top_k != 40:
            param_str += f"_tk{top_k}"
        if presence_penalty is not None and presence_penalty != 0.0:
            param_str += f"_pp{presence_penalty}"
        if frequency_penalty is not None and frequency_penalty != 0.0:
            param_str += f"_fp{frequency_penalty}"

        # Add provider preferences to cache path if specified
        if provider is not None:
            # Hash the provider dict for consistent cache keys
            provider_str = json.dumps(provider, sort_keys=True)
            provider_hash = hashlib.sha256(provider_str.encode()).hexdigest()[
                :8
            ]
            param_str += f"_provider{provider_hash}"

        cache_path = Path(self.cache_dir) / model_str
        cache_path = cache_path / param_str
        # Use same two-level structure as _get_cache_path for consistency
        prompt_hash_start = prompt_hash[:3]
        cache_path = cache_path / prompt_hash_start / prompt_hash

        return str(cache_path)


class ResponseCacheSQL:
    """SQLite-based cache for LLM responses.

    Uses a single SQLite database file instead of millions of JSON files.
    Provides fast lookups with proper indexing and handles concurrent access.
    """

    def __init__(
        self, cache_dir: str = ".rollouts", model: Optional[str] = None
    ):
        """Initialize the SQLite cache.

        Args:
            cache_dir: Directory for the SQLite database file
            model: Model name to use for database filename (optional)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Use model name for database filename if provided
        if model:
            # Clean model name for filesystem compatibility
            db_name = (
                model.replace("/", "-").replace(":", "").replace("@", "-at-")
            )
            db_name = db_name.replace(" ", "-").lower()
            self.db_path = self.cache_dir / f"{db_name}.db"
        else:
            self.db_path = self.cache_dir / "cache.db"

        # Thread-local storage for connections
        self._local = threading.local()

        # Initialize database schema
        self._init_db()

        # Optionally configure SQLite for better performance
        self._optimize_sqlite()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,  # 30 second timeout for locks
                check_same_thread=False,
            )
            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    @contextmanager
    def _get_cursor(self):
        """Context manager for database operations."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _init_db(self):
        """Initialize database schema."""
        # Use direct connection for initialization
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()

            # Main cache table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    cache_key TEXT PRIMARY KEY,
                    prompt_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    accessed_at INTEGER NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """
            )

            # Indexes for fast lookups
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prompt_model
                ON responses(prompt_hash, model)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_params
                ON responses(params_hash)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created
                ON responses(created_at)
            """
            )

            # Optional: Table for storing metadata/statistics
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_stats (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """
            )

            conn.commit()
        finally:
            conn.close()

    def _optimize_sqlite(self):
        """Apply SQLite optimizations for better performance."""
        # Use direct connection for optimization
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()

            # Increase cache size (negative value = KB)
            cursor.execute("PRAGMA cache_size = -64000")  # 64MB cache

            # Faster but less safe (still quite safe with WAL)
            cursor.execute("PRAGMA synchronous = NORMAL")

            # Enable memory mapping for faster reads
            cursor.execute("PRAGMA mmap_size = 268435456")  # 256MB

            # Optimize for SSDs
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.execute("PRAGMA wal_autocheckpoint = 1000")

            conn.commit()
        finally:
            conn.close()

    def _compute_cache_key(
        self,
        prompt: Union[str, List[dict]],
        model: str,
        provider: Optional[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> tuple[str, str, str]:
        """Compute cache key components.

        Returns:
            Tuple of (full_cache_key, prompt_hash, params_hash)
        """
        # Hash the prompt
        if isinstance(prompt, str):
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        elif isinstance(prompt, list):
            prompt_hash = hashlib.sha256(
                json.dumps(prompt).encode("utf-8")
            ).hexdigest()
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        # Create parameters dictionary for hashing
        params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "provider": provider,
        }

        # Remove None values and hash
        params = {k: v for k, v in params.items() if v is not None}
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]

        # Combine into full cache key
        cache_key = f"{model}:{prompt_hash[:16]}:{params_hash}:{seed}"

        return cache_key, prompt_hash, params_hash

    def get(
        self,
        prompt: str,
        model: str,
        provider: Optional[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> Optional[Response]:
        """Get cached response if available."""
        cache_key, prompt_hash, params_hash = self._compute_cache_key(
            prompt,
            model,
            provider,
            temperature,
            top_p,
            max_tokens,
            seed,
            top_k,
            presence_penalty,
            frequency_penalty,
        )

        with self._get_cursor() as cursor:
            # Get cached response and update access stats
            cursor.execute(
                """
                UPDATE responses 
                SET accessed_at = ?, access_count = access_count + 1
                WHERE cache_key = ?
                RETURNING response_json
            """,
                (int(time.time()), cache_key),
            )

            row = cursor.fetchone()
            if not row:
                return None

            # Parse JSON and create Response object
            data = json.loads(row[0])
            return self._dict_to_response(data, model, provider, seed)

    def set(
        self,
        prompt: str,
        model: str,
        provider: Optional[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        response: Response,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> bool:
        """Cache a response."""
        cache_key, prompt_hash, params_hash = self._compute_cache_key(
            prompt,
            model,
            provider,
            temperature,
            top_p,
            max_tokens,
            seed,
            top_k,
            presence_penalty,
            frequency_penalty,
        )

        # Prepare response data
        response_data = {
            "response": response.to_dict(),
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "provider": provider,
        }

        response_json = json.dumps(response_data)
        current_time = int(time.time())

        with self._get_cursor() as cursor:
            # UPSERT: Insert or replace existing
            cursor.execute(
                """
                INSERT OR REPLACE INTO responses 
                (cache_key, prompt_hash, model, params_hash, seed, 
                 response_json, created_at, accessed_at, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 
                    COALESCE((SELECT access_count FROM responses WHERE cache_key = ?), 0) + 1)
            """,
                (
                    cache_key,
                    prompt_hash,
                    model,
                    params_hash,
                    seed,
                    response_json,
                    current_time,
                    current_time,
                    cache_key,
                ),
            )

        return True

    def _dict_to_response(
        self,
        data: Dict[str, Any],
        model: str,
        provider: Optional[Dict[str, Any]],
        seed: int,
    ) -> Response:
        """Convert dictionary data to Response object."""
        resp_data = data["response"]

        # Create Usage object
        usage_data = resp_data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        # Create Response
        return Response(
            full=resp_data.get(
                "full", resp_data.get("full_text", resp_data.get("text", ""))
            ),
            content=resp_data.get("content", resp_data.get("post", "")),
            reasoning=resp_data.get("reasoning", ""),
            finish_reason=resp_data.get("finish_reason", ""),
            provider=resp_data.get("provider", provider),
            response_id=resp_data.get("response_id", ""),
            model=resp_data.get("model", model),
            object=resp_data.get("object", ""),
            created=resp_data.get("created", 0),
            usage=usage,
            logprobs=resp_data.get("logprobs"),
            echo=resp_data.get("echo", False),
            seed=seed,
        )

    def get_cache_dir(
        self,
        prompt: str,
        model: str,
        provider: Optional[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> str:
        """Get the cache directory (returns database path for compatibility)."""
        return str(self.db_path)

    def clear_old_entries(self, days: int = 30):
        """Clear cache entries older than specified days."""
        cutoff_time = int(time.time()) - (days * 86400)

        with self._get_cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM responses 
                WHERE accessed_at < ?
            """,
                (cutoff_time,),
            )

            # Run VACUUM to reclaim space (optional)
            cursor.execute("VACUUM")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._get_cursor() as cursor:
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(LENGTH(response_json)) as total_size,
                    AVG(access_count) as avg_access_count,
                    MAX(access_count) as max_access_count
                FROM responses
            """
            )

            row = cursor.fetchone()
            return {
                "total_entries": row[0] or 0,
                "total_size_mb": (row[1] or 0) / (1024 * 1024),
                "avg_access_count": row[2] or 0,
                "max_access_count": row[3] or 0,
                "db_path": str(self.db_path),
            }

    def export_to_json(self, output_dir: str, limit: Optional[int] = None):
        """Export cache entries to JSON files (for backward compatibility)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with self._get_cursor() as cursor:
            query = "SELECT cache_key, response_json FROM responses"
            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query)

            for cache_key, response_json in cursor:
                file_path = output_path / f"{cache_key}.json"
                with open(file_path, "w") as f:
                    f.write(response_json)
