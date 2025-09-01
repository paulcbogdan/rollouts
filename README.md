# Rollouts

A high-quality Python package for generating multiple LLM responses with built-in resampling, caching, and provider abstraction.

## Features

- **Simple Interface**: Both synchronous and asynchronous APIs
- **Multiple Providers**: Support for OpenRouter, Fireworks, Together, and more
- **Smart Caching**: Automatic response caching to reduce API costs
- **Parameter Override**: Override any setting at generation time
- **Presets**: Built-in presets for common use cases
- **Type Safety**: Full type hints and dataclass models
- **Production Ready**: Comprehensive error handling and retries

## Installation

```bash
pip install rollouts
```

## Examples

See `example.py` for comprehensive examples of all package features:

```bash
# Set your API key
export OPENROUTER_API_KEY="your-key-here"

# Run the examples
python example.py
```

## Quick Start

### Synchronous Usage

```python
from rollouts import RolloutsClient

# Create client with default settings
client = RolloutsClient(
    model="qwen/qwen3-30b-a3b",
    temperature=0.7,
    max_tokens=1000
)

# Generate multiple responses
rollouts = client.generate("What is the meaning of life?", n_samples=5)

# Access responses
for response in rollouts:
    print(response.full)
```

### Asynchronous Usage

```python
import asyncio
from rollouts import RolloutsClient

async def main():
    client = RolloutsClient(model="qwen/qwen3-30b-a3b")
    
    # Generate responses for multiple prompts concurrently
    results = await asyncio.gather(
        client.agenerate("Explain quantum computing", n_samples=3),
        client.agenerate("Write a haiku", n_samples=5, temperature=1.2)
    )
    
    for rollouts in results:
        print(f"Generated {len(rollouts)} responses")

asyncio.run(main())
```

### Using Presets

```python
from rollouts import create_client

# Create client with a preset configuration
client = create_client(
    model="qwen/qwen3-30b-a3b",
    preset="creative"  # High temperature, more diverse outputs
)

responses = client.generate("Write a story", n_samples=3)
```

Available presets:
- `deterministic`: Temperature 0, best for factual responses
- `focused`: Low temperature (0.3), focused but not rigid
- `balanced`: Medium temperature (0.7), good default
- `creative`: High temperature (1.2), diverse outputs

## Thinking Injection (Advanced)

Some models support "thinking injection" where you can control the reasoning process by injecting partial thoughts:

```python
# Works with DeepSeek R1, QwQ, Qwen models
prompt = "Calculate 10*5 <think>Let me calculate: 10*5="
result = client.generate(prompt, n_samples=1)
# Model continues from "=" and completes the calculation
```

**Supported models:**
- ✅ DeepSeek R1 and variants
- ✅ QwQ models
- ✅ Qwen models
- ✅ Claude/Anthropic models
- ❌ GPT-OSS models (no injection support on OpenRouter)
- ❌ Gemini thinking models (internal reasoning only)

For more details, see the `THINK_INJECTION.md` documentation.

## Advanced Usage

### Parameter Override

Override any default setting at generation time:

```python
client = RolloutsClient(model="qwen/qwen3-30b-a3b", temperature=0.7)

# Override temperature for this specific generation
rollouts = client.generate(
    "Be creative!",
    n_samples=5,
    temperature=1.5,  # Override default
    max_tokens=2000   # Override default
)
```

### Custom Configuration

```python
from rollouts import RolloutsClient, Config

# Create custom configuration
config = Config(
    model="qwen/qwen3-30b-a3b",
    temperature=0.8,
    top_p=0.95,
    max_tokens=2000,
    presence_penalty=0.1,
    frequency_penalty=0.1
)

# Use configuration
client = RolloutsClient(**config.to_dict())
```

### Caching

Responses are automatically cached to disk:

```python
client = RolloutsClient(
    model="qwen/qwen3-30b-a3b",
    use_cache=True,  # Default
    cache_dir="my_cache"  # Custom cache directory
)

# First call: generates responses
rollouts1 = client.generate("What is 2+2?", n_samples=3)

# Second call: uses cached responses (instant)
rollouts2 = client.generate("What is 2+2?", n_samples=3)
```

### OpenRouter Implicit Prompt Caching

In addition to this package's local response caching, OpenRouter provides automatic server-side prompt caching for many models. This can significantly reduce costs on repeated API calls with similar prompts:

- **Cost savings**: Cache reads are typically charged at 0.25x to 0.5x the original input token price
- **Automatic**: Most models (OpenAI, DeepSeek, Grok, Gemini 2.5) enable caching automatically with no configuration needed
- **Smart routing**: OpenRouter automatically routes to the same provider to maximize cache hits

This server-side caching works independently from this package's local cache. While our local cache eliminates API calls entirely for identical requests, OpenRouter's prompt caching reduces costs when you make similar (but not identical) requests. For full details on pricing and supported models, see [OpenRouter's Prompt Caching documentation](https://openrouter.ai/docs/features/prompt-caching).

## API Reference

### RolloutsClient

Main client class for generating responses.

**Parameters:**
- `model` (str, required): Model identifier
- `temperature` (float): Sampling temperature (0.0-2.0)
- `top_p` (float): Nucleus sampling parameter
- `max_tokens` (int): Maximum tokens to generate
- `top_k` (int): Top-k sampling parameter
- `presence_penalty` (float): Presence penalty (-2.0 to 2.0)
- `frequency_penalty` (float): Frequency penalty (-2.0 to 2.0)
- `api_key` (str): API key (uses env variable if None)
- `use_cache` (bool): Enable caching
- `verbose` (bool): Print debug information

### Rollouts

Container for multiple responses.

**Attributes:**
- `prompt`: The input prompt
- `responses`: List of Response objects
- `num_responses`: Number of responses requested
- `temperature`, `top_p`, `max_tokens`: Generation parameters
- `model`: Model information

**Methods:**
- `get_texts()`: Get all full response texts (includes reasoning + content)
- `get_reasonings()`: Get reasoning portions only
- `get_contents()`: Get content portions only (post-reasoning text)

### Response

Individual response from the model.

**Key Fields:**
- `full`: The complete response text, formatted as `reasoning_text + "\n</think>\n" + content_text`
- `content`: The post-reasoning text (what comes after `</think>`)
- `reasoning`: The reasoning/thinking text (what comes before `</think>`)
- `usage`: Token usage statistics
- `finish_reason`: Why the response ended (e.g., "stop", "length")

**Understanding the Think Token Format:**

The `full` field is always structured with a `</think>` separator between reasoning and content:
```
reasoning_text
</think>
content_text
```

This format is used consistently even for models that don't natively use `<think>` tags:
- **Models with native think support** (DeepSeek R1, QwQ, Qwen): The reasoning appears naturally
- **GPT-OSS models**: OpenRouter returns reasoning in a separate field, which we format into this structure
- **Models without reasoning**: The `full` field contains just the content (no reasoning section)

**Important Note for GPT-OSS Models:**

GPT-OSS models (like `gpt-oss-20b` and `gpt-oss-120b`) use OpenAI's Harmony format internally. On OpenRouter:
- Reasoning is returned in a separate `reasoning` field by the API
- You cannot inject or control thinking tokens for these models
- The `</think>` separator is added by this library for consistency
- If you need to control reasoning, use models like DeepSeek R1 or QwQ instead

Example accessing Response fields:
```python
for response in rollouts:
    print(f"Full response: {response.full}")
    print(f"Just content: {response.content}")
    print(f"Just reasoning: {response.reasoning}")
    print(f"Tokens used: {response.usage.total_tokens}")
```

## API Key Configuration

There are three ways to provide API keys:

### 1. Environment Variable (recommended for development)
```bash
export OPENROUTER_API_KEY="your-key-here"
```

### 2. Pass to Client (recommended for production)
```python
client = RolloutsClient(
    model="qwen/qwen3-30b-a3b",
    api_key="your-key-here"
)
```

### 3. Pass at Generation Time (for per-request keys)
```python
client = RolloutsClient(model="qwen/qwen3-30b-a3b")
responses = client.generate(
    "Your prompt",
    n_samples=5,
    api_key="different-key-here"  # Overrides any default
)
```

**Note:** API keys are never cached or stored to disk.

## Known Limitations

### Logprobs Not Supported

This package does not currently support logprobs (log probabilities). If you try to use `top_logprobs`, you'll get a `NotImplementedError`:

```python
# This will raise an error:
client = RolloutsClient(
    model="openai/gpt-3.5-turbo",
    top_logprobs=5  # ❌ Not supported
)
```

**Why?** OpenRouter's implementation of logprobs appears inconsistent across different providers. Based on examination of multiple providers, the logprobs functionality doesn't work reliably through OpenRouter's API. Until this is resolved upstream, this feature is not implemented in this package.

If you need logprobs, you may need to use the providers' APIs directly rather than through OpenRouter.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.