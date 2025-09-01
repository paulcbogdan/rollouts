"""
Tests for datatype classes (Response, Rollouts, Usage).
"""

import pytest
from rollouts import Response, Rollouts, Usage


class TestUsage:
    """Test suite for Usage dataclass."""
    
    def test_usage_creation(self):
        """Test creating a Usage object."""
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
        
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30
        
    def test_usage_to_dict(self):
        """Test converting Usage to dictionary."""
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
        
        d = usage.to_dict()
        assert d["prompt_tokens"] == 10
        assert d["completion_tokens"] == 20
        assert d["total_tokens"] == 30


class TestResponse:
    """Test suite for Response dataclass."""
    
    def test_response_creation_minimal(self):
        """Test creating a Response with minimal fields."""
        response = Response(
            full="Test response text"
        )
        
        assert response.full == "Test response text"
        assert response.content == ""
        assert response.reasoning == ""
        assert response.finish_reason == ""
        
    def test_response_creation_full(self):
        """Test creating a Response with all fields."""
        usage = Usage(10, 20, 30)
        response = Response(
            full="Full text",
            content="Content text",
            reasoning="Reasoning text",
            finish_reason="stop",
            provider={"name": "openai"},
            response_id="resp-123",
            model="gpt-3.5-turbo",
            object="chat.completion",
            created=1234567890,
            usage=usage,
            logprobs={"tokens": []},
            echo=True,
            seed=42,
            completed_reasoning=True
        )
        
        assert response.full == "Full text"
        assert response.content == "Content text"
        assert response.reasoning == "Reasoning text"
        assert response.finish_reason == "stop"
        assert response.provider == {"name": "openai"}
        assert response.response_id == "resp-123"
        assert response.model == "gpt-3.5-turbo"
        assert response.object == "chat.completion"
        assert response.created == 1234567890
        assert response.usage == usage
        assert response.logprobs == {"tokens": []}
        assert response.echo is True
        assert response.seed == 42
        assert response.completed_reasoning is True
        
    def test_response_with_reasoning_format(self):
        """Test Response with reasoning in the expected format."""
        reasoning = "Let me think about this"
        content = "The answer is 42"
        full_text = f"{reasoning}\n</think>\n{content}"
        
        response = Response(
            full=full_text,
            content=content,
            reasoning=reasoning
        )
        
        assert response.full == full_text
        assert response.content == content
        assert response.reasoning == reasoning
        
    def test_response_to_dict(self):
        """Test converting Response to dictionary."""
        usage = Usage(10, 20, 30)
        response = Response(
            full="Full text",
            content="Content",
            reasoning="Reasoning",
            finish_reason="stop",
            model="test-model",
            usage=usage,
            seed=42
        )
        
        d = response.to_dict()
        assert d["full"] == "Full text"
        assert d["content"] == "Content"
        assert d["reasoning"] == "Reasoning"
        assert d["finish_reason"] == "stop"
        assert d["model"] == "test-model"
        assert d["usage"]["total_tokens"] == 30
        assert d["seed"] == 42


class TestRollouts:
    """Test suite for Rollouts class."""
    
    def test_rollouts_creation(self, sample_response):
        """Test creating a Rollouts object."""
        rollouts = Rollouts(
            prompt="Test prompt",
            num_responses=3,
            temperature=0.7,
            top_p=0.95,
            max_tokens=100,
            model="test-model",
            responses=[sample_response, sample_response],
            cache_dir="/tmp/cache",
            logprobs_enabled=False,
            echo_enabled=False
        )
        
        assert rollouts.prompt == "Test prompt"
        assert rollouts.num_responses == 3
        assert rollouts.temperature == 0.7
        assert rollouts.top_p == 0.95
        assert rollouts.max_tokens == 100
        assert rollouts.model == "test-model"
        assert len(rollouts.responses) == 2
        assert rollouts.cache_dir == "/tmp/cache"
        assert rollouts.logprobs_enabled is False
        assert rollouts.echo_enabled is False
        
    def test_rollouts_len(self, sample_response):
        """Test __len__ method of Rollouts."""
        rollouts = Rollouts(
            prompt="Test",
            num_responses=5,
            temperature=0.7,
            top_p=0.95,
            max_tokens=100,
            model="test",
            responses=[sample_response] * 3
        )
        
        assert len(rollouts) == 3
        
    def test_rollouts_getitem(self, sample_response, sample_response_with_reasoning):
        """Test __getitem__ method of Rollouts."""
        responses = [sample_response, sample_response_with_reasoning]
        rollouts = Rollouts(
            prompt="Test",
            num_responses=2,
            temperature=0.7,
            top_p=0.95,
            max_tokens=100,
            model="test",
            responses=responses
        )
        
        assert rollouts[0] == sample_response
        assert rollouts[1] == sample_response_with_reasoning
        assert rollouts[-1] == sample_response_with_reasoning
        
        # Test slicing
        sliced = rollouts[0:2]
        assert len(sliced) == 2
        
    def test_rollouts_iter(self, sample_response):
        """Test __iter__ method of Rollouts."""
        responses = [sample_response] * 3
        rollouts = Rollouts(
            prompt="Test",
            num_responses=3,
            temperature=0.7,
            top_p=0.95,
            max_tokens=100,
            model="test",
            responses=responses
        )
        
        count = 0
        for response in rollouts:
            assert response == sample_response
            count += 1
        assert count == 3
        
    def test_rollouts_get_texts(self, sample_response, sample_response_with_reasoning):
        """Test get_texts method."""
        rollouts = Rollouts(
            prompt="Test",
            num_responses=2,
            temperature=0.7,
            top_p=0.95,
            max_tokens=100,
            model="test",
            responses=[sample_response, sample_response_with_reasoning]
        )
        
        texts = rollouts.get_texts()
        assert len(texts) == 2
        assert texts[0] == sample_response.full
        assert texts[1] == sample_response_with_reasoning.full
        
    def test_rollouts_get_reasonings(self, sample_response, sample_response_with_reasoning):
        """Test get_reasonings method."""
        rollouts = Rollouts(
            prompt="Test",
            num_responses=2,
            temperature=0.7,
            top_p=0.95,
            max_tokens=100,
            model="test",
            responses=[sample_response, sample_response_with_reasoning]
        )
        
        reasonings = rollouts.get_reasonings()
        assert len(reasonings) == 2
        assert reasonings[0] == ""  # sample_response has no reasoning
        assert reasonings[1] == sample_response_with_reasoning.reasoning
        
    def test_rollouts_get_contents(self, sample_response, sample_response_with_reasoning):
        """Test get_contents method."""
        rollouts = Rollouts(
            prompt="Test",
            num_responses=2,
            temperature=0.7,
            top_p=0.95,
            max_tokens=100,
            model="test",
            responses=[sample_response, sample_response_with_reasoning]
        )
        
        contents = rollouts.get_contents()
        assert len(contents) == 2
        assert contents[0] == sample_response.content
        assert contents[1] == sample_response_with_reasoning.content
        
    def test_rollouts_get_total_tokens(self, sample_response, sample_response_with_reasoning):
        """Test get_total_tokens method."""
        rollouts = Rollouts(
            prompt="Test",
            num_responses=2,
            temperature=0.7,
            top_p=0.95,
            max_tokens=100,
            model="test",
            responses=[sample_response, sample_response_with_reasoning]
        )
        
        total = rollouts.get_total_tokens()
        # sample_response has 15 tokens, sample_response_with_reasoning has 35
        assert total == 50
        
    def test_rollouts_get_finish_reasons(self, sample_response):
        """Test get_finish_reasons method."""
        response_length = Response(
            full="Test",
            finish_reason="length"
        )
        
        rollouts = Rollouts(
            prompt="Test",
            num_responses=2,
            temperature=0.7,
            top_p=0.95,
            max_tokens=100,
            model="test",
            responses=[sample_response, response_length]
        )
        
        reasons = rollouts.get_finish_reasons()
        assert reasons == {"stop": 1, "length": 1}
        
    def test_rollouts_repr(self, sample_response):
        """Test __repr__ method."""
        rollouts = Rollouts(
            prompt="Test prompt",
            num_responses=3,
            temperature=0.7,
            top_p=0.95,
            max_tokens=100,
            model="test-model",
            responses=[sample_response] * 2
        )
        
        repr_str = repr(rollouts)
        assert "Rollouts" in repr_str
        assert "num_responses=3" in repr_str
        assert "actual=2" in repr_str
        assert "model='test-model'" in repr_str