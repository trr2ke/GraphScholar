"""Provider factory — instantiate a BaseLLMProvider from a name string.

Usage:
    from src.api import get_provider
    provider = get_provider("claude", {"model": "claude-opus-4-8"})
    print(provider.complete([{"role": "user", "content": "hello"}]))
"""
from .base import BaseLLMProvider
from .claude import ClaudeProvider
from .openai_provider import OpenAIProvider

# Maps provider name → constructor
_REGISTRY: dict[str, type] = {
    "claude": ClaudeProvider,
    "openai": OpenAIProvider,
    "kimi": OpenAIProvider,   # OpenAI-compatible; set base_url in config
    "glm4": OpenAIProvider,   # OpenAI-compatible; set base_url in config
}


def get_provider(name: str, config: dict) -> BaseLLMProvider:
    """Return an initialised LLM provider.

    Args:
        name: Provider name — "claude", "openai", "kimi", or "glm4".
        config: Dict of provider kwargs forwarded to the constructor.
                For Claude: model, api_key.
                For OpenAI-compatible: model, embedding_model, api_key, base_url.

    Returns:
        A ready-to-use BaseLLMProvider instance.

    Raises:
        ValueError: If `name` is not a recognised provider.
    """
    name = name.lower()
    if name not in _REGISTRY:
        supported = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown provider '{name}'. Choose from: {supported}")

    cls = _REGISTRY[name]

    # Pull only the kwargs the constructor accepts to avoid unexpected-keyword errors
    if cls is ClaudeProvider:
        kwargs = {k: config[k] for k in ("model", "api_key") if k in config}
    else:
        kwargs = {
            k: config[k]
            for k in ("model", "embedding_model", "api_key", "base_url")
            if k in config
        }

    return cls(**kwargs)
