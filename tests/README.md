# Rollouts Test Suite

Comprehensive test suite for the rollouts package using pytest.

## Test Coverage

- **test_config.py**: Tests for Config class validation and methods
- **test_datatypes.py**: Tests for Response, Rollouts, and Usage dataclasses
- **test_cache.py**: Tests for response caching functionality
- **test_think_handler.py**: Tests for think token handling and model detection
- **test_rate_limiter.py**: Tests for rate limiting functionality
- **test_client.py**: Tests for the main RolloutsClient class
- **test_openrouter.py**: Tests for OpenRouter API integration
- **test_integration.py**: End-to-end integration tests

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage report
pytest --cov=rollouts --cov-report=html

# Run specific test file
pytest tests/test_client.py

# Run specific test class
pytest tests/test_client.py::TestRolloutsClientInit

# Run specific test
pytest tests/test_client.py::TestRolloutsClientInit::test_client_creation_minimal

# Run with verbose output
pytest -v

# Run only fast tests (skip integration)
pytest -m "not integration"
```

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `temp_cache_dir`: Temporary directory for cache testing
- `mock_api_key`: Mock API key
- `sample_config`: Sample Config object
- `sample_response`: Sample Response object
- `sample_response_with_reasoning`: Response with reasoning
- `mock_httpx_client`: Mock HTTP client
- `mock_env_api_key`: Mock environment API key
- `sample_rollouts`: Sample Rollouts object

## Writing New Tests

1. Create test file with `test_` prefix
2. Create test class with `Test` prefix
3. Create test methods with `test_` prefix
4. Use fixtures from conftest.py
5. Mock external dependencies (API calls, file I/O)
6. Test both success and failure cases

Example:
```python
class TestNewFeature:
    def test_feature_success(self, sample_config):
        # Test successful case
        assert feature(sample_config) == expected
        
    def test_feature_failure(self):
        # Test failure case
        with pytest.raises(ValueError):
            feature(invalid_input)
```