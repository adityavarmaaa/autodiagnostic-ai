import os
from typing import Dict, List

import requests

from .documents import clean_text, chunk_text


class WebSearchService:

    def __init__(self):

        self.api_key = os.getenv(
            "TAVILY_API_KEY",
            ""
        ).strip()

        self.timeout = int(
            os.getenv(
                "WEB_REQUEST_TIMEOUT",
                "10"
            )
        )

        self.max_results = int(
            os.getenv(
                "WEB_MAX_RESULTS",
                "3"
            )
        )

        self.endpoint = (
            "https://api.tavily.com/search"
        )

    # =====================================================
    # QUERY NORMALIZATION
    # =====================================================

    def _normalize_query(
        self,
        query: str,
    ) -> str:
        """
        Keep the original vehicle + complaint.

        The important rule here is:
        NEVER replace the user's complaint with a
        specification-only query.
        """

        query = query.strip()

        if not query:
            return query

        normalized = query.lower()

        # -------------------------------------------------
        # Common Maruti spelling
        # -------------------------------------------------

        normalized = normalized.replace(
            "maruthi",
            "maruti"
        )

        # -------------------------------------------------
        # e VITARA
        # -------------------------------------------------
        #
        # Keep the original query and add problem-focused
        # search terms.
        # -------------------------------------------------

        if (
            "e vitara" in normalized
            or "evitara" in normalized
        ):

            # Keep the original query exactly in the search.
            #
            # Example:
            #
            # "Maruthi e Vitara" "battery warning light"
            #
            return (
                f'"{query}" '
                '"Maruti Suzuki e VITARA" '
                "common problems "
                "owner complaints "
                "reported issues "
                "faults "
                "warning signs "
                "troubleshooting "
                "repair "
                "service issues "
                "user experiences "
                "forum"
            )

        # -------------------------------------------------
        # General vehicle query
        # -------------------------------------------------

        automotive_terms = [
            "car",
            "vehicle",
            "suv",
            "sedan",
            "hatchback",
            "engine",
            "battery",
            "brake",
            "transmission",
            "diagnostic",
            "dtc",
            "mileage",
            "service",
            "repair",
            "fault",
            "warning",
            "problem",
            "problems",
            "issue",
            "issues",
            "complaint",
            "complaints",
        ]

        contains_auto_term = any(
            term in normalized
            for term in automotive_terms
        )

        if contains_auto_term:

            return (
                f'"{query}" '
                "automotive "
                "common problems "
                "owner complaints "
                "reported issues "
                "troubleshooting "
                "repair "
                "service "
                "forum"
            )

        return query

    # =====================================================
    # SEARCH
    # =====================================================

    def search(
        self,
        query: str,
        limit: int | None = None,
    ) -> List[Dict]:

        if not self.api_key:

            raise RuntimeError(
                "TAVILY_API_KEY is missing from .env"
            )

        original_query = query

        query = self._normalize_query(
            query
        )

        print()
        print("=" * 60)
        print("[Web Search]")
        print(
            f"Original query : {original_query}"
        )
        print(
            f"Search query   : {query}"
        )
        print("=" * 60)

        limit = (
            self.max_results
            if limit is None
            else max(
                1,
                min(
                    limit,
                    10,
                )
            )
        )

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": limit,
            "include_answer": False,
            "include_raw_content": True,
        }

        response = requests.post(
            self.endpoint,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        results = data.get(
            "results",
            []
        )

        print(
            f"[Web Search] Results received: "
            f"{len(results)}"
        )

        return results

    # =====================================================
    # DOCUMENT COLLECTION
    # =====================================================

    def collect_documents(
        self,
        query: str,
        limit: int | None = None,
    ) -> List[Dict]:

        results = self.search(
            query=query,
            limit=limit,
        )

        documents = []

        for index, result in enumerate(
            results,
            start=1,
        ):

            title = (
                result.get(
                    "title",
                    f"Web Source {index}",
                )
                or f"Web Source {index}"
            ).strip()

            url = (
                result.get(
                    "url",
                    "",
                )
                or ""
            ).strip()

            raw_content = (
                result.get(
                    "raw_content",
                    "",
                )
                or ""
            ).strip()

            content = (
                result.get(
                    "content",
                    "",
                )
                or ""
            ).strip()

            # -------------------------------------------------
            # Prefer raw content when it is larger.
            # -------------------------------------------------

            if len(raw_content) > len(content):

                text = raw_content

            else:

                text = content

            text = clean_text(
                text
            )

            print()
            print(
                f"[Web Source {index}]"
            )

            print(
                f"Title: {title}"
            )

            print(
                f"URL: {url}"
            )

            print(
                f"Content length: {len(text)}"
            )

            # -------------------------------------------------
            # Skip pages where Tavily returned no content.
            # -------------------------------------------------

            if not text:

                print(
                    "[Web Search] "
                    "Skipping empty source."
                )

                continue

            documents.append(
                {
                    "title": title,
                    "url": url,
                    "text": text,
                }
            )

        print()
        print(
            f"[Web Search] Valid documents: "
            f"{len(documents)}"
        )

        return documents

    # =====================================================
    # CHUNK COLLECTION
    # =====================================================

    def collect_chunks(
        self,
        query: str,
        limit: int | None = None,
    ) -> List[Dict]:

        documents = self.collect_documents(
            query=query,
            limit=limit,
        )

        chunks = []

        for document in documents:

            page_chunks = chunk_text(
                document["text"]
            )

            for index, chunk in enumerate(
                page_chunks
            ):

                chunk = chunk.strip()

                if not chunk:
                    continue

                chunks.append(
                    {
                        "text": chunk,

                        "source": document[
                            "title"
                        ],

                        "url": document[
                            "url"
                        ],

                        "title": document[
                            "title"
                        ],

                        "page": "web",

                        "chunk_index": index,

                        "type": "web",

                        "knowledge_status": "web",
                    }
                )

        print()

        print(
            f"[Web Search] Generated "
            f"{len(chunks)} chunks."
        )

        return chunks