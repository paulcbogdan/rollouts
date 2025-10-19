"""Cache migration utilities for converting between JSON and SQLite formats."""

import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from tqdm import tqdm
import shutil

from .cache import ResponseCacheSQL, ResponseCacheJson
from .datatypes import Response, Usage


def _detect_models_in_cache(cache_path: Path, format: str) -> list[str]:
    """Detect all models in a cache directory.

    Args:
        cache_path: Path to cache directory
        format: Cache format ("json" or "sql")

    Returns:
        List of model names found
    """
    models = []

    if format == "json":
        # In JSON cache, models are top-level directories
        # The directory names are filesystem-safe versions of model names
        for path in cache_path.iterdir():
            if path.is_dir() and not path.name.startswith('.'):
                # The directory name is the filesystem-safe model name
                # We'll use it as-is since the migration will handle it correctly
                models.append(path.name)

    elif format == "sql":
        # In SQLite cache, models are .db files
        for db_file in cache_path.glob("*.db"):
            if db_file.name != "cache.db":  # Skip default cache
                # The .db filename is already the model name
                # (with slashes replaced by dashes)
                models.append(db_file.stem)

    return sorted(models)


def _migrate_all_models(
    source_path: Path,
    target_path: Path,
    source_format: str,
    target_format: str,
    overwrite: bool,
    verbose: bool
) -> Dict[str, Any]:
    """Migrate all models found in the source cache."""

    # Detect all models in source
    models = _detect_models_in_cache(source_path, source_format)

    if not models:
        if verbose:
            print(f"No models found in {source_path}")
        return {
            "total_models": 0,
            "models_migrated": [],
            "models_failed": [],
            "total_migrated": 0,
            "total_failed": 0
        }

    if verbose:
        print(f"Found {len(models)} model(s) to migrate:")
        for model in models:
            print(f"  • {model}")
        print()

    # Migrate each model
    results = {
        "total_models": len(models),
        "models_migrated": [],
        "models_failed": [],
        "total_migrated": 0,
        "total_failed": 0,
        "total_skipped": 0
    }

    for model_dir_name in models:
        # For JSON format, the model is a directory name that needs conversion
        # For SQL format, it's already a proper model name (sort of)
        if source_format == "json":
            # The directory name is filesystem-safe, we use it as-is for migration
            model = model_dir_name
        else:
            # For SQL, the filename might have dashes instead of slashes
            model = model_dir_name

        if verbose:
            print(f"\nMigrating model: {model}")
            print("-" * 40)

        try:
            if source_format == "json" and target_format == "sql":
                model_result = _migrate_json_to_sql(
                    source_path, target_path, model, overwrite, verbose
                )
            elif source_format == "sql" and target_format == "json":
                model_result = _migrate_sql_to_json(
                    source_path, target_path, model, overwrite, verbose
                )
            else:
                raise ValueError(f"Invalid format combination: {source_format} -> {target_format}")

            results["models_migrated"].append(model)
            results["total_migrated"] += model_result.get("migrated", 0)
            results["total_failed"] += model_result.get("failed", 0)
            results["total_skipped"] += model_result.get("skipped", 0)

            if verbose:
                print(f"  ✓ Migrated {model_result.get('migrated', 0)} entries")

        except Exception as e:
            if verbose:
                print(f"  ✗ Failed to migrate {model}: {e}")
            results["models_failed"].append(model)

    if verbose:
        print("\n" + "=" * 50)
        print("Migration Summary:")
        print(f"  • Models processed: {results['total_models']}")
        print(f"  • Models succeeded: {len(results['models_migrated'])}")
        print(f"  • Models failed: {len(results['models_failed'])}")
        print(f"  • Total entries migrated: {results['total_migrated']}")
        print(f"  • Total entries failed: {results['total_failed']}")
        print(f"  • Total entries skipped: {results['total_skipped']}")

    return results


def migrate_cache(
    source_dir: str=".rollouts",
    target_dir: Optional[str]=None,
    source_format: Literal["json", "sql"] = "json",
    target_format: Literal["json", "sql"] = "sql",
    model: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Migrate cache between JSON and SQLite formats.

    Args:
        source_dir: Directory containing the source cache (default: ".rollouts")
        target_dir: Directory for the target cache (default: auto-generated based on format)
        source_format: Format of source cache ("json" or "sql")
        target_format: Format of target cache ("json" or "sql")
        model: Model name (optional - if None, migrates all models)
        overwrite: Whether to overwrite existing target cache
        verbose: Whether to print progress information

    Returns:
        Dictionary with migration statistics

    Examples:
        # JSON to SQLite for specific model
        migrate_cache(source_dir=".rollouts", target_dir=".rollouts_sql",
                     source_format="json", target_format="sql",
                     model="openai/gpt-4")

        # Migrate ALL models from JSON to SQLite (auto target dir)
        migrate_cache(source_format="json", target_format="sql",
                     model=None)  # Creates .rollouts_sql automatically

        # SQLite to JSON (for backup/export)
        migrate_cache(source_dir=".rollouts_sql", target_dir=".rollouts_backup",
                     source_format="sql", target_format="json",
                     model="openai/gpt-4")
    """
    if source_format == target_format:
        raise ValueError(f"Source and target formats cannot be the same: {source_format}")

    source_path = Path(source_dir)

    # Default target_dir to source_dir (in-place migration)
    if target_dir is None:
        target_dir = source_dir
        if verbose:
            print(f"Migrating in-place within: {target_dir}")

    target_path = Path(target_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    # If model is None, detect and migrate all models
    if model is None:
        return _migrate_all_models(
            source_path, target_path, source_format, target_format, overwrite, verbose
        )

    # Check if target exists and handle overwrite
    if source_format == "json" and target_format == "sql":
        return _migrate_json_to_sql(
            source_path, target_path, model, overwrite, verbose
        )
    elif source_format == "sql" and target_format == "json":
        return _migrate_sql_to_json(
            source_path, target_path, model, overwrite, verbose
        )
    else:
        raise ValueError(f"Invalid format combination: {source_format} -> {target_format}")


def _migrate_json_to_sql(
    source_path: Path,
    target_path: Path,
    model: str,
    overwrite: bool,
    verbose: bool
) -> Dict[str, Any]:
    """Migrate from JSON cache to SQLite.

    Args:
        source_path: Path to source JSON cache directory
        target_path: Path to target SQLite cache directory
        model: Model name (required for SQLite)
        overwrite: Whether to overwrite existing data
        verbose: Whether to print progress

    Returns:
        Migration statistics
    """

    # Initialize SQLite cache
    sql_cache = ResponseCacheSQL(str(target_path), model=model)

    # Check if database already has data
    stats = sql_cache.get_stats()
    if stats['total_entries'] > 0 and not overwrite:
        raise FileExistsError(
            f"Target SQLite database already contains {stats['total_entries']} entries. "
            f"Use overwrite=True to replace existing data."
        )

    if overwrite and stats['total_entries'] > 0:
        if verbose:
            print(f"Warning: Overwriting {stats['total_entries']} existing entries in SQLite database")
        # Clear existing data
        conn = sqlite3.connect(str(sql_cache.db_path))
        conn.execute("DELETE FROM responses")
        conn.commit()
        conn.close()

    # Find JSON cache files for this specific model
    # The model directory name is the filesystem-safe version
    model_dir_name = model.replace("/", "-").replace(":", "").replace("@", "-at-")
    model_path = source_path / model_dir_name

    if model_path.exists():
        # Look for JSON files only in the model's directory
        json_files = list(model_path.rglob("*.json"))
    else:
        # Fallback: look for all JSON files (for backward compatibility)
        json_files = list(source_path.rglob("*.json"))

    if not json_files:
        if verbose:
            print(f"No JSON cache files found for model {model}")
        return {"migrated": 0, "failed": 0, "skipped": 0}

    if verbose:
        print(f"Found {len(json_files)} JSON cache files to migrate")
        json_files = tqdm(json_files, desc="Migrating JSON to SQLite")

    migrated = 0
    failed = 0
    skipped = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract response data
            if 'response' not in data:
                skipped += 1
                continue

            resp_data = data['response']

            # Create Response object
            usage_data = resp_data.get('usage', {})
            usage = Usage(
                prompt_tokens=usage_data.get('prompt_tokens', 0),
                completion_tokens=usage_data.get('completion_tokens', 0),
                total_tokens=usage_data.get('total_tokens', 0),
            )

            response = Response(
                full=resp_data.get('full', resp_data.get('full_text', resp_data.get('text', ''))),
                content=resp_data.get('content', resp_data.get('post', '')),
                reasoning=resp_data.get('reasoning', ''),
                finish_reason=resp_data.get('finish_reason', ''),
                provider=resp_data.get('provider', data.get('provider')),
                response_id=resp_data.get('response_id', ''),
                model=resp_data.get('model', data.get('model', model or '')),
                object=resp_data.get('object', ''),
                created=resp_data.get('created', 0),
                usage=usage,
                logprobs=resp_data.get('logprobs'),
                echo=resp_data.get('echo', False),
                seed=data.get('seed', 0),
            )

            # Get parameters from cache data
            prompt = data.get('prompt', '')
            cache_model = data.get('model', model or '')
            temperature = data.get('temperature', 0.7)
            top_p = data.get('top_p', 1.0)
            max_tokens = data.get('max_tokens', 100)
            seed = data.get('seed', 0)
            provider = data.get('provider')

            # Cache the response in SQLite
            success = sql_cache.set(
                prompt=prompt,
                model=cache_model,
                provider=provider,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                response=response,
                top_k=data.get('top_k'),
                presence_penalty=data.get('presence_penalty'),
                frequency_penalty=data.get('frequency_penalty'),
            )

            if success:
                migrated += 1
            else:
                failed += 1

        except Exception as e:
            if verbose:
                print(f"Failed to migrate {json_file}: {e}")
            failed += 1

    final_stats = sql_cache.get_stats()

    if verbose:
        print(f"\nMigration complete:")
        print(f"  • Migrated: {migrated} entries")
        print(f"  • Failed: {failed} entries")
        print(f"  • Skipped: {skipped} entries")
        print(f"  • Database size: {final_stats['total_size_mb']:.2f} MB")
        print(f"  • Database path: {sql_cache.db_path}")

    return {
        "migrated": migrated,
        "failed": failed,
        "skipped": skipped,
        "db_path": str(sql_cache.db_path),
        "total_entries": final_stats['total_entries'],
        "size_mb": final_stats['total_size_mb']
    }


def _migrate_sql_to_json(
    source_path: Path,
    target_path: Path,
    model: str,
    overwrite: bool,
    verbose: bool
) -> Dict[str, Any]:
    """Migrate from SQLite cache to JSON.

    Args:
        source_path: Path to source SQLite cache directory
        target_path: Path to target JSON cache directory
        model: Model name
        overwrite: Whether to overwrite existing data
        verbose: Whether to print progress

    Returns:
        Migration statistics
    """

    # Find SQLite database file
    if model:
        db_name = model.replace("/", "-").replace(":", "").replace("@", "-at-").replace(" ", "-").lower()
        db_path = source_path / f"{db_name}.db"
    else:
        db_path = source_path / "cache.db"

    if not db_path.exists():
        # Try to find any .db file
        db_files = list(source_path.glob("*.db"))
        if not db_files:
            raise FileNotFoundError(f"No SQLite database found in {source_path}")
        db_path = db_files[0]
        if verbose:
            print(f"Using database: {db_path.name}")

    # Check if target directory exists
    if target_path.exists() and not overwrite:
        # Check if there are existing JSON files
        existing_json = list(target_path.rglob("*.json"))
        if existing_json:
            raise FileExistsError(
                f"Target directory contains {len(existing_json)} JSON files. "
                f"Use overwrite=True to replace existing data."
            )

    if target_path.exists() and overwrite:
        if verbose:
            print(f"Warning: Overwriting existing JSON cache in {target_path}")
        shutil.rmtree(target_path)

    # Initialize JSON cache
    json_cache = ResponseCacheJson(str(target_path))

    # Connect to SQLite database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get all entries
    cursor.execute("SELECT response_json FROM responses")
    rows = cursor.fetchall()

    if not rows:
        if verbose:
            print(f"No entries found in SQLite database")
        conn.close()
        return {"migrated": 0, "failed": 0, "skipped": 0}

    if verbose:
        print(f"Found {len(rows)} entries to migrate from SQLite")
        rows = tqdm(rows, desc="Migrating SQLite to JSON")

    migrated = 0
    failed = 0
    skipped = 0

    for row in rows:
        try:
            # Parse response JSON
            data = json.loads(row[0])

            # Extract response data
            if 'response' not in data:
                skipped += 1
                continue

            resp_data = data['response']

            # Create Response object
            usage_data = resp_data.get('usage', {})
            usage = Usage(
                prompt_tokens=usage_data.get('prompt_tokens', 0),
                completion_tokens=usage_data.get('completion_tokens', 0),
                total_tokens=usage_data.get('total_tokens', 0),
            )

            response = Response(
                full=resp_data.get('full', resp_data.get('full_text', resp_data.get('text', ''))),
                content=resp_data.get('content', resp_data.get('post', '')),
                reasoning=resp_data.get('reasoning', ''),
                finish_reason=resp_data.get('finish_reason', ''),
                provider=resp_data.get('provider', data.get('provider')),
                response_id=resp_data.get('response_id', ''),
                model=resp_data.get('model', data.get('model', '')),
                object=resp_data.get('object', ''),
                created=resp_data.get('created', 0),
                usage=usage,
                logprobs=resp_data.get('logprobs'),
                echo=resp_data.get('echo', False),
                seed=data.get('seed', 0),
            )

            # Cache the response in JSON format
            success = json_cache.set(
                prompt=data.get('prompt', ''),
                model=data.get('model', ''),
                provider=data.get('provider'),
                temperature=data.get('temperature', 0.7),
                top_p=data.get('top_p', 1.0),
                max_tokens=data.get('max_tokens', 100),
                seed=data.get('seed', 0),
                response=response,
                top_k=data.get('top_k'),
                presence_penalty=data.get('presence_penalty'),
                frequency_penalty=data.get('frequency_penalty'),
            )

            if success:
                migrated += 1
            else:
                failed += 1

        except Exception as e:
            if verbose:
                print(f"Failed to migrate entry: {e}")
            failed += 1

    conn.close()

    # Count final JSON files
    json_files = list(target_path.rglob("*.json"))

    if verbose:
        print(f"\nMigration complete:")
        print(f"  • Migrated: {migrated} entries")
        print(f"  • Failed: {failed} entries")
        print(f"  • Skipped: {skipped} entries")
        print(f"  • Total JSON files: {len(json_files)}")
        print(f"  • Target directory: {target_path}")

    return {
        "migrated": migrated,
        "failed": failed,
        "skipped": skipped,
        "target_dir": str(target_path),
        "total_files": len(json_files)
    }


def auto_migrate(
    cache_dir: str = ".rollouts",
    model: Optional[str] = None,
    to_format: Literal["json", "sql"] = "sql",
    backup: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Automatically detect and migrate cache format.

    Args:
        cache_dir: Cache directory to check/migrate
        model: Model name (optional - if None, migrates all models)
        to_format: Target format to migrate to
        backup: Whether to create backup before migration
        overwrite: Whether to overwrite existing cache
        verbose: Whether to print progress

    Returns:
        Migration statistics

    Examples:
        # Auto-migrate specific model to SQLite
        from rollouts.migrate import auto_migrate
        auto_migrate(model="openai/gpt-4")

        # Auto-migrate ALL models to SQLite
        auto_migrate(model=None)  # Migrates all models found
    """
    cache_path = Path(cache_dir)

    # Detect current format
    has_json = bool(list(cache_path.rglob("*.json"))) if cache_path.exists() else False
    has_sql = bool(list(cache_path.glob("*.db"))) if cache_path.exists() else False

    if not has_json and not has_sql:
        if verbose:
            print(f"No cache found in {cache_dir}")
        return {"status": "no_cache"}

    current_format = "json" if has_json else "sql"

    if current_format == to_format:
        if verbose:
            print(f"Cache is already in {to_format} format")
        return {"status": "already_migrated"}

    # Create backup if requested
    if backup:
        backup_dir = f"{cache_dir}_backup_{current_format}"
        if verbose:
            print(f"Creating backup in {backup_dir}")
        shutil.copytree(cache_dir, backup_dir, dirs_exist_ok=True)

    # Perform in-place migration (same directory)
    return migrate_cache(
        source_dir=cache_dir,
        target_dir=cache_dir,  # Same directory - they don't conflict!
        source_format=current_format,
        target_format=to_format,
        model=model,
        overwrite=overwrite,
        verbose=verbose
    )

