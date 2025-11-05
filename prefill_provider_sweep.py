from rollouts import RolloutsClient
import os
import requests


def get_all_providers(regenerate=False):
    if not regenerate:
        providers = [
            "ai21",
            "aion-labs",
            "alibaba",
            "amazon-bedrock",
            "anthropic",
            "atlas-cloud",
            "atoma",
            "avian",
            "azure",
            "baseten",
            "cerebras",
            "chutes",
            "cirrascale",
            "clarifai",
            "cloudflare",
            "cohere",
            "crofai",
            "crusoe",
            "deepinfra",
            "deepseek",
            "enfer",
            "fake-provider",
            "featherless",
            "fireworks",
            "friendli",
            "gmicloud",
            "google-ai-studio",
            "google-vertex",
            "groq",
            "hyperbolic",
            "inception",
            "inference-net",
            "infermatic",
            "inflection",
            "klusterai",
            "lambda",
            "liquid",
            "mancer",
            "meta",
            "minimax",
            "mistral",
            "modelrun",
            "modular",
            "moonshotai",
            "morph",
            "ncompass",
            "nebius",
            "nextbit",
            "nineteen",
            "novita",
            "nvidia",
            "open-inference",
            "openai",
            "parasail",
            "perplexity",
            "phala",
            "relace",
            "sambanova",
            "siliconflow",
            "stealth",
            "switchpoint",
            "targon",
            "together",
            "ubicloud",
            "venice",
            "wandb",
            "xai",
            "z-ai",
        ]
        return providers

    API_KEY = os.getenv("OPENROUTER_API_KEY")
    resp = requests.get(
        "https://openrouter.ai/api/v1/providers",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30,
    )
    providers = sorted(p["slug"] for p in resp.json()["data"])
    return providers


if __name__ == "__main__":
    # model = "deepseek/deepseek-r1-0528"
    model = "qwen/qwen3-32b"
    # model = "z-ai/glm-4.6"
    client = RolloutsClient(
        model=model,
        temperature=0.7,
        max_tokens=100,
        max_retries=1,
    )
    prompt = "What is 2 x 4 = ?<think>2 x 4 = "
    providers = get_all_providers()
    provider2output = {}
    for provider in providers:
        rollouts = client.generate(
            prompt,
            n_samples=1,
            provider={"only": [provider]},
            max_tokens=10,
            verbose=False,
        )
        if not len(rollouts):
            continue
        print(f"{provider}: {rollouts[0]=}")
        provider2output[provider] = rollouts[0].reasoning

    for provider, output in provider2output.items():
        if output[0] == "8":
            print(f"Good! {provider}, {output=}")
