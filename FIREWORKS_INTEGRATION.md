# Fireworks API Integration with Logprobs Support

## Overview

This implementation provides a complete Fireworks API integration for the rollouts package, maintaining the same interface and style as the existing OpenRouter implementation while adding comprehensive logprobs support.

## Files Created

### 1. `rollouts/fireworks.py`
The main Fireworks API provider implementation that handles:
- API communication using Fireworks completions endpoint
- Automatic retry logic with exponential backoff
- Model name conversion (e.g., "qwen-15b" → Fireworks format)
- Message formatting and prompt conversion
- Logprobs extraction and parsing
- Echo mode support for prompt token logprobs

### 2. `rollouts/fw_datatypes.py`
Extended data models with logprobs support:
- `Logprobs`: Dataclass for storing token log probabilities
  - `tokens`: List of token strings
  - `token_logprobs`: Log probabilities for each token
  - `top_logprobs`: Top alternative tokens and their probabilities
  - `text_offset`: Character offsets in the text
  - Helper methods to separate prompt and generation logprobs
- `FwResponse`: Extended response with logprobs and echo support
- `FwRollouts`: Collection of Fireworks responses with logprobs metadata

### 3. `rollouts/fw_client.py`
The main client interface that:
- Maintains the same API as `RolloutsClient`
- Always enables logprobs (configurable 1-5 alternatives)
- Supports echo mode for prompt token logprobs
- Includes automatic caching with logprobs-aware cache keys
- Provides both sync and async interfaces
- Shows progress bars for multiple samples

### 4. `rollouts/fw_cache.py`
Extended caching system that:
- Includes logprobs and echo parameters in cache keys
- Ensures responses with different logprobs settings are cached separately
- Maintains compatibility with existing cache structure

## Key Features

### Logprobs Support
- **Full logprobs data**: Get log probabilities for all tokens
- **Echo mode**: Include prompt tokens in logprobs output
- **Top alternatives**: Up to 5 alternative tokens per position
- **Separated access**: Helper methods to get prompt vs generation logprobs

### Compatibility
- **Same interface**: Drop-in replacement for `RolloutsClient`
- **Caching**: Automatic response caching with logprobs awareness
- **Progress tracking**: Visual progress bars for multiple generations
- **Error handling**: Robust retry logic and error recovery

### Model Support
The implementation supports various models available on Fireworks:
- `qwen-15b`: DeepSeek R1 Distill Qwen 1.5B
- `llama-v3p1-8b`: Llama 3.1 8B
- `gpt-oss-20b`: GPT-OSS 20B
- Any other Fireworks-supported model

## Usage Example

```python
from rollouts.fw_client import FireworksClient

# Create client with logprobs enabled
client = FireworksClient(
    model="qwen-15b",
    temperature=0.7,
    logprobs=5,    # Get top 5 alternatives
    echo=True      # Include prompt logprobs
)

# Generate responses with logprobs
rollouts = client.generate("What is 2+2?", n_samples=5)

# Access logprobs data
for response in rollouts:
    if response.logprobs:
        # Get logprobs for generated tokens only
        gen_logprobs = response.get_generation_logprobs()

        # Get logprobs for prompt tokens
        prompt_logprobs = response.get_prompt_logprobs()

        # Access raw logprobs data
        tokens = response.logprobs.tokens
        token_logprobs = response.logprobs.token_logprobs
        top_alternatives = response.logprobs.top_logprobs
```

## Environment Setup

Set your Fireworks API key:
```bash
export FIREWORKS_API_KEY="your-key-here"
```

## Differences from OpenRouter Implementation

1. **API Endpoint**: Uses Fireworks completions API instead of chat API
2. **Logprobs**: Full support including echo mode (OpenRouter doesn't support this)
3. **Model Names**: Automatic conversion to Fireworks format
4. **Seed Support**: Seeds are used for caching but not sent to API (Fireworks limitation)
5. **Cache Directory**: Uses `.fw_rollouts` by default (vs `.rollouts`)

## Python Version Compatibility

The implementation includes fallbacks for older Python versions:
- Python 3.6: Handles missing dataclasses (requires backport)
- Older tqdm: Falls back to regular tqdm if asyncio module unavailable

## Testing

Run syntax checks:
```bash
python -m py_compile rollouts/fireworks.py rollouts/fw_*.py
```

Run the example:
```bash
python example_fireworks.py
```

## Value Proposition

The main value of using Fireworks over OpenRouter is the comprehensive logprobs support:
- Get probability distributions for all tokens
- Understand model confidence at each step
- Access alternative token choices
- Analyze prompt token probabilities with echo mode

This is particularly useful for:
- Research on model behavior
- Uncertainty quantification
- Token-level analysis
- Debugging generation issues
- Building more sophisticated sampling strategies