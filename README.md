# Rollouts

`rollouts` is python package for conveniently interacting with the OpenRouter API. The package provides three notable features:

- You can generate multiple LLM responses ("rollouts") concurrently for the same prompt
- The package will automatically cache responses. The first time you call `client.generate('your prompt', n_samples=2)`, two jsons will be saved with the model response to each. If you make the same call, those jsons will be loaded
- You can easily insert text into a model's reasoning. If you call `client.generate('What is 5*10?\n<think>\n5*1')`, this will insert `\n5*1'` into the model's reasoning, which will continue with `"0..."`

Examples are provided below, and additional examples are shown in `example.py`.

## Paper

This code is meant to help with implementing the chain-of-thought resampling techniques described in this paper:

Bogdan, P.C.\*, Macar, U.\*, Nanda, N.°, & Conmy, A.° (2025). Thought Anchors: Which LLM Reasoning Steps Matter?. arXiv preprint arXiv:2506.19143. [PDF](https://arxiv.org/pdf/2506.19143)

## Installation

```bash
pip install rollouts
```

## Quick Start

```bash
# Set your API key
export OPENROUTER_API_KEY="your-key-here"
```

### Synchronous Usage

Model responses are always via the chat-completions API.

```python
from rollouts import RolloutsClient

# Create client with default settings
client = RolloutsClient(
    model="qwen/qwen3-30b-a3b",
    temperature=0.7,
    max_tokens=1000
) 

# Generate multiple responses (one prompt sampled concurrently). This runs on seeds from 0 to n_samples (e.g., 0, 1, 2, 3, 4)
rollouts = client.generate("What is the meaning of life?", n_samples=5)

# Access responses
for response in rollouts:
    print(f"Reasoning: {response.reasoning=}") # reasoning text if reasoning model; None if non-reasoning model
    print(f"Content: {response.content=}") # post-reasoning output (or just output if not a reasoning model)
    print(f"Response: {response.full=}") # "{reasoning}</think>{content}" if reasoning exists and completed; "{reasoning}" if reasoning not completed; "{content}" if non-reasoning model or if reasoning is hidden
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

### Thinking Injection

For models using <think> tags, you can insert thoughts and continue the chain-of-thought from there (this works for Deepseek, Qwen, QwQ, Anthropic, and presumably other models). 

```python
prompt = "Calculate 10*5\n<think>\nLet me calculate: 10*5="
result = client.generate(prompt, n_samples=1)
# Model continues from "=" ("50" would be the next two tokens)
```

I believe `"<think>"` is normally surrounded by `"\n"` for chat completions by default. You probably should do this.

Importantly, you should avoid ending inserted thoughts with a trailing space (`" "`). Doing so will often cause tokenization issues, as most models tokenize words with a space prefix (e.g., `" Hello"`). When you insert thoughts with a trailing space, a model would need to introduce a double-space typo to continue with a word. Models hate typos and will thus be strongly biased toward continuing with tokens that don't have a space prefix (e.g., `"0"`).

Inserting thoughts does not work for:
- Models where true thinking tokens are hidden (Gemini and OpenAI)
- GPT-OSS-20b/120b, which use a different reasoning template; I tried to get the GPT-OSS template working, but I'm not sure it's possible with OpenRouter

## Parameter Override

The default OpenRouter settings are used, but you can override these either when defining the client or when generating responses. The logprobs parameter is not supported here; from what I can tell, it is unreliable on OpenRouter

```python
client = RolloutsClient(model="qwen/qwen3-30b-a3b", temperature=0.7)

# Override temperature for this specific generation
rollouts = client.generate(
    "Be creative!",
    n_samples=5,
    temperature=1.5,  # Override default
    max_tokens=2000   # Override default
)

result = client.generate(prompt, top_p=0.99)
```

### Progress Bar

A progress bar automatically appears when generating multiple responses (n_samples > 1):

```python
client = RolloutsClient(
    model="qwen/qwen3-30b-a3b",
    progress_bar=True  # Default, can be disabled
)

# Shows a progress bar for multiple samples
rollouts = client.generate("Write a story", n_samples=5)

# No progress bar for single sample (even if enabled)
rollout = client.generate("Quick answer", n_samples=1)

# Disable progress bar for a specific request
rollouts = client.generate("Silent generation", n_samples=10, progress_bar=False)
```

The progress bar:
- Only appears when n_samples > 1
- Shows the number of responses being generated
- Automatically disappears when complete
- Can be disabled globally (in client init) or per-request

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

**Cache Behavior:**
- Responses are cached in a hierarchical directory structure: `.rollouts/model/parameters/prompt_hash_prefix/prompt_hash/seed_00000.json`
- Each unique combination of prompt, model, and parameters gets its own cache location
- The prompt hash is split across two directory levels (`prompt_hash_prefix/prompt_hash`) as this helps performance when you have responses saved for >100k prompts. `prompt_hash_prefix` is just the first three hex digits of the prompt hash
- If a cached response has `finish_reason="error"`, it will not be loaded and is instead regenerated on the next request
- To clear the cache, simply delete the cache directory or specific subdirectories/files

## API Key Configuration

There are three ways to provide API keys:

### 1. Environment Variable
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

## Additional Notes

### Progress Bar
A progress bar appears when generating multiple responses (`n_samples > 1`). You can disable it by setting `progress_bar=False` either when creating the client or for individual requests.

### Rate Limiting
You can limit the requests per minute when defining your client using the `requests_per_minute` parameter (token bucket rate limiter):

```python
client = RolloutsClient(
    model="qwen/qwen3-30b-a3b",
    requests_per_minute=60  # Limit to 60 requests per minute
)
```