"""Base LLM provider interface for GraphScholar.

All providers implement this ABC so the pipeline can swap between
Claude, OpenAI, Kimi, GLM4, etc. purely via config.yaml.
"""
from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Abstract base class every LLM provider must implement."""

    @abstractmethod
    def complete(self, messages: list[dict], **kwargs) -> str:
        """Send a chat-completion request and return the response text.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str} dicts.
            **kwargs: Provider-specific overrides (max_tokens, temperature, etc.).

        Returns:
            The model's text response as a plain string.
        """
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into dense float vectors.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of float vectors, one per input text, in the same order.
        """
        ...
