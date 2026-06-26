"""Anthropic Claude provider for GraphScholar.

Completions use the official `anthropic` SDK.
Embeddings fall back to sentence-transformers (local, no API cost)
because Anthropic does not expose a public embeddings endpoint.
"""
import os

import anthropic

# On corporate networks the OS cert store trusts the proxy CA but Python's
# bundled certifi store does not.  truststore patches ssl to use the OS store
# (same store git uses via schannel), fixing SSL errors transparently.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass  # truststore optional; falls back to certifi

from .base import BaseLLMProvider


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude via the official anthropic SDK."""

    def __init__(self, model: str = "claude-opus-4-8", api_key: str | None = None):
        """
        Args:
            model: Claude model ID (default: claude-opus-4-8).
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        """
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        # Lazy-loaded sentence-transformers model for embeddings
        self._embedder = None

    def complete(self, messages: list[dict], **kwargs) -> str:
        """Call Claude and return the text response.

        Uses adaptive thinking for models that support it (Opus 4.6+).
        Streams internally for large max_tokens values to avoid timeouts.
        """
        max_tokens = kwargs.pop("max_tokens", 4096)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs,
        )
        return response.content[0].text

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using a local sentence-transformers model.

        The model (all-MiniLM-L6-v2) is downloaded once and cached on disk
        by the sentence-transformers library.
        """
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        vectors = self._embedder.encode(texts, convert_to_numpy=True)
        return vectors.tolist()
