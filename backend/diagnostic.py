from typing import Dict, List

from .llm import OllamaService
from .prompts import (
    SYSTEM_PROMPT,
    build_diagnostic_prompt,
)
from .rag import RAGService, build_context


class DiagnosticService:
    """
    Main AutoDiag AI diagnostic service.

    It retrieves relevant technical information
    and sends that information to the local LLM.
    """

    def __init__(self):
        self.rag = RAGService()
        self.llm = OllamaService()

    def analyze(
        self,
        vehicle: str,
        complaint: str,
        dtc: str = "",
        top_k: int = 5,
    ) -> Dict:
        """
        Analyze an automotive complaint.
        """

        vehicle = vehicle.strip()
        complaint = complaint.strip()
        dtc = dtc.strip()

        if not vehicle:
            raise ValueError(
                "Vehicle information is required."
            )

        if not complaint:
            raise ValueError(
                "Complaint is required."
            )

        # Build the search query.
        search_query = self._build_search_query(
            vehicle=vehicle,
            complaint=complaint,
            dtc=dtc,
        )

        # Retrieve relevant technical information.
        results = self.rag.search(
            query=search_query,
            top_k=top_k,
        )

        # Convert retrieved chunks into LLM context.
        context = build_context(
            results
        )

        # Build the final prompt.
        user_prompt = build_diagnostic_prompt(
            vehicle=vehicle,
            complaint=complaint,
            dtc=dtc,
            context=context,
        )

        # Generate the diagnostic response.
        answer = self.llm.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        return {
            "vehicle": vehicle,
            "complaint": complaint,
            "dtc": dtc,
            "answer": answer,
            "sources": results,
            "search_query": search_query,
        }

    @staticmethod
    def _build_search_query(
        vehicle: str,
        complaint: str,
        dtc: str,
    ) -> str:
        """
        Create the semantic-search query.
        """

        parts = [
            vehicle,
            complaint,
        ]

        if dtc:
            parts.append(dtc)

        return " | ".join(parts)

    @staticmethod
    def format_sources(
        sources: List[Dict],
    ) -> str:
        """
        Format retrieved sources for display.
        """

        if not sources:
            return "No sources found."

        lines = []

        for index, source in enumerate(
            sources,
            start=1,
        ):
            lines.append(
                f"{index}. "
                f"{source['source']} "
                f"(page {source['page']})"
            )

        return "\n".join(lines)