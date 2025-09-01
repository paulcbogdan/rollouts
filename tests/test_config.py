"""
Tests for the Config class.
"""

import pytest
from rollouts import Config


class TestConfig:
    """Test suite for Config class."""
    
    def test_config_creation_minimal(self):
        """Test creating a config with minimal parameters."""
        config = Config(model="test/model")
        assert config.model == "test/model"
        assert config.temperature == 0.7
        assert config.top_p == 0.95
        assert config.max_tokens == 4096
        
    def test_config_creation_full(self):
        """Test creating a config with all parameters."""
        config = Config(
            model="test/model",
            temperature=1.2,
            top_p=0.9,
            max_tokens=2000,
            seed=42,
            stream=True,
            top_k=50,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            repetition_penalty=1.1,
            min_p=0.05,
            top_a=0.1,
            max_retries=50,
            timeout=600,
            verbose=True,
            use_cache=False,
            cache_dir="custom_cache",
            requests_per_minute=100
        )
        
        assert config.model == "test/model"
        assert config.temperature == 1.2
        assert config.top_p == 0.9
        assert config.max_tokens == 2000
        assert config.seed == 42
        assert config.stream is True
        assert config.top_k == 50
        assert config.frequency_penalty == 0.5
        assert config.presence_penalty == 0.3
        assert config.repetition_penalty == 1.1
        assert config.min_p == 0.05
        assert config.top_a == 0.1
        assert config.max_retries == 50
        assert config.timeout == 600
        assert config.verbose is True
        assert config.use_cache is False
        assert config.cache_dir == "custom_cache"
        assert config.requests_per_minute == 100
        
    def test_config_no_model_raises_error(self):
        """Test that creating a config without a model raises an error."""
        with pytest.raises(ValueError, match="model parameter is required"):
            Config()
            
    def test_config_temperature_validation(self):
        """Test temperature parameter validation."""
        # Valid temperatures
        Config(model="test", temperature=0.0)
        Config(model="test", temperature=2.0)
        
        # Invalid temperatures
        with pytest.raises(ValueError, match="temperature must be between"):
            Config(model="test", temperature=-0.1)
        with pytest.raises(ValueError, match="temperature must be between"):
            Config(model="test", temperature=2.1)
            
    def test_config_top_p_validation(self):
        """Test top_p parameter validation."""
        # Valid top_p
        Config(model="test", top_p=0.01)
        Config(model="test", top_p=1.0)
        
        # Invalid top_p
        with pytest.raises(ValueError, match="top_p must be between"):
            Config(model="test", top_p=0.0)
        with pytest.raises(ValueError, match="top_p must be between"):
            Config(model="test", top_p=1.1)
            
    def test_config_max_tokens_validation(self):
        """Test max_tokens parameter validation."""
        # Valid max_tokens
        Config(model="test", max_tokens=1)
        Config(model="test", max_tokens=10000)
        
        # Invalid max_tokens
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            Config(model="test", max_tokens=0)
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            Config(model="test", max_tokens=-1)
            
    def test_config_penalty_validation(self):
        """Test frequency and presence penalty validation."""
        # Valid penalties
        Config(model="test", frequency_penalty=-2.0)
        Config(model="test", frequency_penalty=2.0)
        Config(model="test", presence_penalty=-2.0)
        Config(model="test", presence_penalty=2.0)
        
        # Invalid penalties
        with pytest.raises(ValueError, match="frequency_penalty must be between"):
            Config(model="test", frequency_penalty=-2.1)
        with pytest.raises(ValueError, match="frequency_penalty must be between"):
            Config(model="test", frequency_penalty=2.1)
        with pytest.raises(ValueError, match="presence_penalty must be between"):
            Config(model="test", presence_penalty=-2.1)
        with pytest.raises(ValueError, match="presence_penalty must be between"):
            Config(model="test", presence_penalty=2.1)
            
    def test_config_repetition_penalty_validation(self):
        """Test repetition_penalty validation."""
        # Valid
        Config(model="test", repetition_penalty=0.1)
        Config(model="test", repetition_penalty=2.0)
        
        # Invalid
        with pytest.raises(ValueError, match="repetition_penalty must be between"):
            Config(model="test", repetition_penalty=0.0)
        with pytest.raises(ValueError, match="repetition_penalty must be between"):
            Config(model="test", repetition_penalty=2.1)
            
    def test_config_top_k_validation(self):
        """Test top_k validation."""
        # Valid
        Config(model="test", top_k=1)
        Config(model="test", top_k=1000)
        
        # Invalid
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            Config(model="test", top_k=0)
            
    def test_config_min_p_validation(self):
        """Test min_p validation."""
        # Valid
        Config(model="test", min_p=0.0)
        Config(model="test", min_p=1.0)
        
        # Invalid
        with pytest.raises(ValueError, match="min_p must be between"):
            Config(model="test", min_p=-0.1)
        with pytest.raises(ValueError, match="min_p must be between"):
            Config(model="test", min_p=1.1)
            
    def test_config_top_a_validation(self):
        """Test top_a validation."""
        # Valid
        Config(model="test", top_a=0.0)
        Config(model="test", top_a=1.0)
        
        # Invalid
        with pytest.raises(ValueError, match="top_a must be between"):
            Config(model="test", top_a=-0.1)
        with pytest.raises(ValueError, match="top_a must be between"):
            Config(model="test", top_a=1.1)
            
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Config(
            model="test/model",
            temperature=1.2,
            seed=42
        )
        
        d = config.to_dict()
        assert d["model"] == "test/model"
        assert d["temperature"] == 1.2
        assert d["seed"] == 42
        assert "top_p" in d
        assert "max_tokens" in d
        
    def test_config_to_api_params(self):
        """Test converting config to API parameters."""
        config = Config(
            model="test/model",
            temperature=1.2,
            seed=42,
            verbose=True,
            use_cache=True,
            cache_dir="custom",
            max_retries=50,
            timeout=300,
            requests_per_minute=60
        )
        
        params = config.to_api_params()
        
        # Client-only settings should be removed
        assert "verbose" not in params
        assert "use_cache" not in params
        assert "cache_dir" not in params
        assert "max_retries" not in params
        assert "timeout" not in params
        assert "requests_per_minute" not in params
        
        # API parameters should be present
        assert params["model"] == "test/model"
        assert params["temperature"] == 1.2
        assert params["seed"] == 42
        
        # None values should be removed
        assert "logit_bias" not in params
        assert "top_logprobs" not in params
        
    def test_config_copy_with(self):
        """Test creating a copy with overrides."""
        config = Config(
            model="test/model",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Create copy with overrides
        new_config = config.copy_with(
            temperature=1.2,
            seed=42
        )
        
        # Original should be unchanged
        assert config.temperature == 0.7
        assert config.seed is None
        
        # New config should have overrides
        assert new_config.temperature == 1.2
        assert new_config.seed == 42
        assert new_config.max_tokens == 1000  # Inherited
        assert new_config.model == "test/model"  # Inherited