import ollama

from .config import LLM_MODEL, OLLAMA_HOST


class OllamaService:
    """
    Fast local Ollama service for AutoDiag AI.
    """

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

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """
        Generate a diagnostic response using Ollama.
        """

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
                # Keep responses focused
                "temperature": temperature,

                # Faster generation
                "num_predict": 220,

                # Smaller context for lower latency
                "num_ctx": 2048,
            },

            # Keep the model loaded in memory
            keep_alive=-1,
        )

        # -------------------------------------------------
        # Ollama response object
        # -------------------------------------------------

        if hasattr(response, "message"):

            message = response.message

            if hasattr(message, "content"):
                return message.content

            if isinstance(message, dict):
                return message.get(
                    "content",
                    "",
                )

        # -------------------------------------------------
        # Dictionary response fallback
        # -------------------------------------------------

        if isinstance(response, dict):

            message = response.get(
                "message",
                {},
            )

            if isinstance(message, dict):

                return message.get(
                    "content",
                    "",
                )

        # -------------------------------------------------
        # Final fallback
        # -------------------------------------------------

        return str(response)