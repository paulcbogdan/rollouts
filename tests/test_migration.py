"""Test cache migration functionality."""

import json
from pathlib import Path
import shutil
import pytest

from rollouts import RolloutsClient, migrate_cache, auto_migrate
from rollouts.datatypes import Response, Usage
from rollouts.cache import ResponseCacheJson, ResponseCacheSQL


@pytest.fixture
def sample_response():
    """Create a sample response for testing."""
    return Response(
        full="The answer is 42",
        content="42",
        reasoning="Deep thought",
        finish_reason="stop",
        provider={"order": ["openai"]},
        response_id="test-123",
        model="openai/gpt-4",
        object="chat.completion",
        created=1234567890,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        logprobs=None,
        echo=False,
        seed=42
    )


@pytest.fixture
def setup_json_cache(tmp_path, sample_response):
    """Create a sample JSON cache for testing."""
    cache_dir = tmp_path / "json_cache"
    json_cache = ResponseCacheJson(str(cache_dir))

    test_prompts = [
        "What is 2+2?",
        "Explain quantum computing",
        "Write a haiku about coding",
        "What is the meaning of life?",
        "How does photosynthesis work?"
    ]

    for i, prompt in enumerate(test_prompts):
        response = Response(
            full=f"Response to: {prompt}",
            content=f"Answer {i+1}",
            reasoning=f"Reasoning for {prompt}",
            finish_reason="stop",
            provider={"order": ["openai"]},
            response_id=f"test-{i}",
            model="openai/gpt-4",
            object="chat.completion",
            created=1234567890 + i,
            usage=Usage(prompt_tokens=10+i, completion_tokens=20+i, total_tokens=30+2*i),
            logprobs=None,
            echo=False,
            seed=42 + i
        )

        json_cache.set(
            prompt=prompt,
            model="openai/gpt-4",
            provider={"order": ["openai"]},
            temperature=0.7,
            top_p=1.0,
            max_tokens=100,
            seed=42 + i,
            response=response
        )

    return cache_dir, len(test_prompts)


def test_json_to_sql_migration(tmp_path, setup_json_cache):
    """Test migrating from JSON to SQLite."""
    json_cache_dir, num_entries = setup_json_cache
    sql_cache_dir = tmp_path / "sql_cache"

    # Perform migration
    result = migrate_cache(
        source_dir=str(json_cache_dir),
        target_dir=str(sql_cache_dir),
        source_format="json",
        target_format="sql",
        model="openai/gpt-4",
        overwrite=False,
        verbose=False
    )

    assert result['migrated'] == num_entries
    assert result['failed'] == 0

    # Verify SQLite database was created
    db_path = Path(result['db_path'])
    assert db_path.exists()

    # Test that data can be retrieved
    sql_cache = ResponseCacheSQL(str(sql_cache_dir), model="openai/gpt-4")
    retrieved = sql_cache.get(
        prompt="What is 2+2?",
        model="openai/gpt-4",
        provider={"order": ["openai"]},
        temperature=0.7,
        top_p=1.0,
        max_tokens=100,
        seed=42
    )

    assert retrieved is not None
    assert retrieved.content == "Answer 1"


def test_sql_to_json_migration(tmp_path, setup_json_cache):
    """Test migrating from SQLite back to JSON."""
    json_cache_dir, num_entries = setup_json_cache
    sql_cache_dir = tmp_path / "sql_cache"
    json_export_dir = tmp_path / "json_export"

    # First migrate to SQL
    migrate_cache(
        source_dir=str(json_cache_dir),
        target_dir=str(sql_cache_dir),
        source_format="json",
        target_format="sql",
        model="openai/gpt-4",
        verbose=False
    )

    # Then migrate back to JSON
    result = migrate_cache(
        source_dir=str(sql_cache_dir),
        target_dir=str(json_export_dir),
        source_format="sql",
        target_format="json",
        model="openai/gpt-4",
        overwrite=False,
        verbose=False
    )

    assert result['migrated'] > 0
    assert result['failed'] == 0

    # Verify JSON files were created
    json_files = list(json_export_dir.rglob("*.json"))
    assert len(json_files) > 0

    # Test that data can be retrieved from JSON cache
    json_cache = ResponseCacheJson(str(json_export_dir))
    retrieved = json_cache.get(
        prompt="What is 2+2?",
        model="openai/gpt-4",
        provider={"order": ["openai"]},
        temperature=0.7,
        top_p=1.0,
        max_tokens=100,
        seed=42
    )

    assert retrieved is not None
    assert retrieved.content == "Answer 1"


def test_overwrite_protection(tmp_path, setup_json_cache):
    """Test that overwrite protection works."""
    json_cache_dir, _ = setup_json_cache
    sql_cache_dir = tmp_path / "sql_cache"

    # First migration
    migrate_cache(
        source_dir=str(json_cache_dir),
        target_dir=str(sql_cache_dir),
        source_format="json",
        target_format="sql",
        model="openai/gpt-4",
        verbose=False
    )

    # Try to migrate again without overwrite (should fail)
    with pytest.raises(FileExistsError):
        migrate_cache(
            source_dir=str(json_cache_dir),
            target_dir=str(sql_cache_dir),
            source_format="json",
            target_format="sql",
            model="openai/gpt-4",
            overwrite=False,
            verbose=False
        )

    # Now try with overwrite=True (should succeed)
    result = migrate_cache(
        source_dir=str(json_cache_dir),
        target_dir=str(sql_cache_dir),
        source_format="json",
        target_format="sql",
        model="openai/gpt-4",
        overwrite=True,
        verbose=False
    )

    assert result['migrated'] > 0


def test_auto_migrate(tmp_path, setup_json_cache):
    """Test automatic migration detection."""
    json_cache_dir, _ = setup_json_cache
    auto_dir = tmp_path / "auto_test"

    # Copy JSON cache to auto directory
    shutil.copytree(json_cache_dir, auto_dir)

    # Test auto-migration
    result = auto_migrate(
        cache_dir=str(auto_dir),
        model="openai/gpt-4",
        to_format="sql",
        backup=True,
        verbose=False
    )

    # Check that backup was created
    backup_dir = Path(f"{auto_dir}_backup_json")
    assert backup_dir.exists()

    # Check that SQL cache was created IN THE SAME directory
    assert auto_dir.exists()
    assert list(auto_dir.glob("*.db"))  # SQLite files in same dir


def test_migration_with_cache_objects(tmp_path, sample_response):
    """Test migration using cache objects directly."""
    json_cache_dir = tmp_path / "json_direct"
    sql_cache_dir = tmp_path / "sql_direct"

    # Create JSON cache with direct cache object
    json_cache = ResponseCacheJson(str(json_cache_dir))
    json_cache.set(
        prompt="What is the meaning of life?",
        model="openai/gpt-4",
        provider=None,
        temperature=0.7,
        top_p=1.0,
        max_tokens=100,
        seed=1337,
        response=sample_response
    )

    # Migrate to SQLite
    result = migrate_cache(
        source_dir=str(json_cache_dir),
        target_dir=str(sql_cache_dir),
        source_format="json",
        target_format="sql",
        model="openai/gpt-4",
        verbose=False
    )

    assert result['migrated'] == 1

    # Verify data in SQLite
    sql_cache = ResponseCacheSQL(str(sql_cache_dir), model="openai/gpt-4")
    retrieved = sql_cache.get(
        prompt="What is the meaning of life?",
        model="openai/gpt-4",
        provider=None,
        temperature=0.7,
        top_p=1.0,
        max_tokens=100,
        seed=1337
    )

    assert retrieved is not None
    assert retrieved.content == "42"


def test_invalid_format_combination():
    """Test that invalid format combinations raise errors."""
    # Only test that same format is invalid
    with pytest.raises(ValueError, match="Source and target formats cannot be the same"):
        migrate_cache(
            source_dir=".",
            target_dir="./different",
            source_format="json",
            target_format="json",
            model="test"
        )


def test_nonexistent_source_directory():
    """Test that nonexistent source directory raises error."""
    with pytest.raises(FileNotFoundError, match="Source directory does not exist"):
        migrate_cache(
            source_dir="/nonexistent/path",
            target_dir=".",
            source_format="json",
            target_format="sql",
            model="test"
        )


def test_empty_cache_directory(tmp_path):
    """Test migration with empty cache directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    target_dir = tmp_path / "target"

    result = migrate_cache(
        source_dir=str(empty_dir),
        target_dir=str(target_dir),
        source_format="json",
        target_format="sql",
        model="openai/gpt-4",
        verbose=False
    )

    assert result['migrated'] == 0
    assert result['failed'] == 0
    assert result['skipped'] == 0


def test_migrate_all_models(tmp_path):
    """Test migrating all models with model=None."""
    source_dir = tmp_path / "multi_model_cache"
    target_dir = tmp_path / "multi_model_sql"

    # Create cache with multiple models
    models = ["openai/gpt-4", "anthropic/claude-3", "deepseek/r1"]
    test_prompts = ["Question 1", "Question 2"]

    for model in models:
        json_cache = ResponseCacheJson(str(source_dir))
        for i, prompt in enumerate(test_prompts):
            response = Response(
                full=f"Response from {model}",
                content=f"Answer {i+1}",
                reasoning="",
                finish_reason="stop",
                provider=None,
                response_id=f"{model}-{i}",
                model=model,
                object="chat.completion",
                created=1234567890 + i,
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                logprobs=None,
                echo=False,
                seed=42 + i
            )

            json_cache.set(
                prompt=prompt,
                model=model,
                provider=None,
                temperature=0.7,
                top_p=1.0,
                max_tokens=100,
                seed=42 + i,
                response=response
            )

    # Migrate all models with model=None
    result = migrate_cache(
        source_dir=str(source_dir),
        target_dir=str(target_dir),
        source_format="json",
        target_format="sql",
        model=None,  # Migrate all models
        verbose=False
    )

    # Check results
    assert "total_models" in result
    assert result['total_models'] == len(models)
    assert len(result['models_migrated']) == len(models)
    assert result['total_migrated'] == len(models) * len(test_prompts)

    # Verify database files were created for each model
    db_files = list(target_dir.glob("*.db"))
    assert len(db_files) == len(models)