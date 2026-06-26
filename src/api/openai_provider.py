"""OpenAI-compatible provider for GraphScholar.

Covers OpenAI, Kimi (api.moonshot.cn), GLM4 (open.bigmodel.cn),
and any other service that speaks the OpenAI REST API.
Point `base_url` in config.yaml at the alternative endpoint.
"""
import os

from openai import OpenAI

from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible chat + embeddings provider."""

    def __init__(
        self,
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Args:
            model: Chat model ID.
            embedding_model: Embeddings model ID.
            api_key: API key. Falls back to OPENAI_API_KEY env var.
            base_url: Override for non-OpenAI endpoints (Kimi, GLM4, etc.).
                      Falls back to OPENAI_BASE_URL env var when None.
        """
        self.model = model
        self.embedding_model = embedding_model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            # None → OpenAI default; env var OPENAI_BASE_URL also honoured by SDK
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def complete(self, messages: list[dict], **kwargs) -> str:
        """Call the chat completions endpoint and return the response text."""
        max_tokens = kwargs.pop("max_tokens", 4096)
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the provider's embeddings endpoint."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]
