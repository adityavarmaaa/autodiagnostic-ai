from typing import Any, Dict, Optional

import ollama

from .config import LLM_MODEL, OLLAMA_HOST


class OllamaService:
    """Handles communication with the local Ollama server."""

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = LLM_MODEL,
    ):
        self.host = host
        self.model = model

        self.client = ollama.Client(
            host=self.host
        )

    def check_connection(self) -> bool:
        """Check whether Ollama is running."""

        try:
            self.client.list()
            return True

        except Exception:
            return False

    def list_models(self):
        """Return models available in Ollama."""

        response = self.client.list()

        if hasattr(response, "models"):
            return response.models

        if isinstance(response, dict):
            return response.get("models", [])

        return []

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Send a chat request to Ollama."""

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": temperature,
            },
        )

        if hasattr(response, "message"):

            message = response.message

            if hasattr(message, "content"):
                return message.content

            if isinstance(message, dict):
                return message.get(
                    "content",
                    "",
                )

        if isinstance(response, dict):

            return response.get(
                "message",
                {},
            ).get(
                "content",
                "",
            )

        return str(response)


def get_ollama_service() -> OllamaService:
    """Create an Ollama service."""

    return OllamaService()