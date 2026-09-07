from typing import List

import ollama

from .config import (
    EMBEDDING_MODEL,
    OLLAMA_HOST,
)


class EmbeddingService:
    """Generates embeddings using Ollama."""

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = EMBEDDING_MODEL,
    ):
        self.model = model

        self.client = ollama.Client(
            host=host
        )

    def embed(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts."""

        if not texts:
            return []

        response = self.client.embed(
            model=self.model,
            input=texts,
        )

        if hasattr(response, "embeddings"):
            return response.embeddings

        if isinstance(response, dict):
            return response.get(
                "embeddings",
                []
            )

        raise RuntimeError(
            "Unexpected embedding response from Ollama."
        )

    def embed_one(
        self,
        text: str,
    ) -> List[float]:
        """Generate one embedding."""

        embeddings = self.embed(
            [text]
        )

        if not embeddings:
            raise RuntimeError(
                "Ollama returned no embedding."
            )

        return embeddings[0]