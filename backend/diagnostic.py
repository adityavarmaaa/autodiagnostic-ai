from typing import Dict, List

from .config import TOP_K, WEB_MAX_RESULTS
from .documents import chunk_text
from .llm import OllamaService
from .prompts import (
    SYSTEM_PROMPT,
    build_diagnostic_prompt,
)
from .rag import RAGService, build_context
from .web_search import WebSearchService


class DiagnosticService:
    """
    Automotive diagnostic pipeline.

    1. Search persistent local Knowledge Base.
    2. Filter local results by vehicle.
    3. If useful local knowledge exists, use it.
    4. Otherwise search the web.
    5. Extract and chunk web pages in memory.
    6. Keep source URL/title with every chunk.
    7. Persist web chunks into ChromaDB.
    8. Send relevant evidence to Ollama.
    """

    def __init__(self):
        self.rag = RAGService()
        self.llm = OllamaService()
        self.web_search = WebSearchService()

    # =====================================================
    # MAIN ANALYSIS
    # =====================================================

    def analyze(
        self,
        vehicle: str,
        complaint: str,
        dtc: str = "",
        top_k: int = TOP_K,
    ) -> Dict:

        vehicle = vehicle.strip()
        complaint = complaint.strip()
        dtc = dtc.strip()

        if not vehicle:
            raise ValueError(
                "Vehicle information is required."
            )

        if not complaint:
            raise ValueError(
                "Customer complaint is required."
            )

        # -------------------------------------------------
        # Build search query
        # -------------------------------------------------

        search_query = self._build_search_query(
            vehicle=vehicle,
            complaint=complaint,
            dtc=dtc,
        )

        print()
        print("=" * 60)
        print("[AutoDiag] NEW DIAGNOSTIC REQUEST")
        print("=" * 60)
        print(f"Vehicle   : {vehicle}")
        print(f"Complaint : {complaint}")
        print(f"DTC       : {dtc or 'None'}")
        print(f"Query     : {search_query}")
        print("=" * 60)

        # =================================================
        # 1. SEARCH PERSISTENT KNOWLEDGE BASE
        # =================================================

        print(
            "[AutoDiag] Searching persistent Knowledge Base..."
        )

        # -------------------------------------------------
        # Search inside the vehicle manufacturer's
        # organized Knowledge Base first.
        #
        # Example:
        #   BMW -> BMW 3 Series -> P0300
        #
        # Existing older knowledge without organization
        # metadata is still supported by the fallback
        # general search below.
        # -------------------------------------------------

        manufacturer = (
            self._extract_manufacturer(
                vehicle
            )
        )

        print(
            f"[AutoDiag] Knowledge group: "
            f"{manufacturer.upper()}"
        )

        local_results = (
            self.rag.search_organized(
                query=search_query,
                manufacturer=manufacturer,
                top_k=min(top_k, 3),
            )
        )

        print(
            f"[AutoDiag] Organized "
            f"{manufacturer.upper()} results: "
            f"{len(local_results)}"
        )

        # -------------------------------------------------
        # BACKWARD COMPATIBILITY
        # -------------------------------------------------
        #
        # Old PDF/TXT/web chunks may not have
        # manufacturer metadata. Keep the original
        # search as a fallback so nothing is lost.
        # -------------------------------------------------

        if not local_results:

            print(
                "[AutoDiag] No organized knowledge "
                f"found for {manufacturer.upper()}."
            )

            print(
                "[AutoDiag] Running general "
                "Knowledge Base search..."
            )

            local_results = self.rag.search(
                query=search_query,
                top_k=min(top_k, 3),
            )

        print(
            f"[AutoDiag] Raw local results: "
            f"{len(local_results)}"
        )

        # -------------------------------------------------
        # Vehicle filter
        # -------------------------------------------------

        local_results = self._filter_vehicle_results(
            results=local_results,
            vehicle=vehicle,
        )

        print(
            f"[AutoDiag] Vehicle-relevant local results: "
            f"{len(local_results)}"
        )

        used_web_search = False
        web_sources = []

        # =================================================
        # 2. USE LOCAL KNOWLEDGE IF GOOD
        # =================================================

        if self._has_useful_results(
            results=local_results,
            vehicle=vehicle,
            complaint=complaint,
            dtc=dtc,
        ):

            print(
                "[AutoDiag] ✅ Using persistent "
                "Knowledge Base."
            )

            results = local_results

        # =================================================
        # 3. WEB FALLBACK
        # =================================================

        else:

            used_web_search = True

            print(
                "[AutoDiag] ❌ Local knowledge insufficient."
            )

            print(
                "[AutoDiag] 🌐 Starting web research..."
            )

            try:

                web_documents = (
                    self.web_search.collect_documents(
                        query=search_query,
                        limit=WEB_MAX_RESULTS,
                    )
                )

            except Exception as error:

                print(
                    f"[AutoDiag] Web search failed: {error}"
                )

                web_documents = []

            print(
                f"[AutoDiag] Web documents received: "
                f"{len(web_documents)}"
            )

            web_chunks = []

            # =================================================
            # PROCESS EVERY WEB DOCUMENT
            # =================================================

            for index, document in enumerate(
                web_documents,
                start=1,
            ):

                title = document.get(
                    "title",
                    f"Web source {index}",
                ).strip()

                url = document.get(
                    "url",
                    "",
                ).strip()

                text = document.get(
                    "text",
                    "",
                ).strip()

                print()
                print(
                    f"[Web {index}] Title: {title}"
                )

                print(
                    f"[Web {index}] URL: {url}"
                )

                # -------------------------------------------------
                # ALWAYS preserve source link
                # -------------------------------------------------

                web_sources.append(
                    {
                        "title": title,
                        "url": url,
                    }
                )

                # -------------------------------------------------
                # If no text came back, skip chunks.
                # The URL still remains visible in the UI.
                # -------------------------------------------------

                if not text:

                    print(
                        f"[Web {index}] No extracted text."
                    )

                    continue

                # -------------------------------------------------
                # Chunk the page in memory
                # -------------------------------------------------

                chunks = chunk_text(
                    text
                )

                print(
                    f"[Web {index}] "
                    f"Extracted chunks: {len(chunks)}"
                )

                # -------------------------------------------------
                # Process chunks
                # -------------------------------------------------

                for chunk_index, chunk in enumerate(
                    chunks
                ):

                    chunk = chunk.strip()

                    if len(chunk) < 100:
                        continue

                    searchable_text = (
                        f"{title} "
                        f"{url} "
                        f"{chunk}"
                    )

                    # ---------------------------------------------
                    # Vehicle relevance
                    # ---------------------------------------------

                    if not self._vehicle_matches_text(
                        vehicle=vehicle,
                        text=searchable_text,
                    ):

                        continue

                    web_chunks.append(
                        {
                            "text": chunk,
                            "source": title,
                            "url": url,
                            "title": title,
                            "page": "web",
                            "chunk_index": chunk_index,
                            "type": "web",
                            "knowledge_status": "web",
                        }
                    )

            # =================================================
            # IF STRICT FILTER REMOVED EVERYTHING
            # =================================================

            if not web_chunks:

                print(
                    "[AutoDiag] Strict web vehicle "
                    "filter returned 0 chunks."
                )

                # Use the first useful chunks from the
                # search results rather than returning
                # nothing. This prevents the UI from saying
                # "No relevant sources" when the search
                # actually found sources.

                for document in web_documents:

                    title = document.get(
                        "title",
                        "Web source",
                    )

                    url = document.get(
                        "url",
                        "",
                    )

                    text = document.get(
                        "text",
                        "",
                    ).strip()

                    if not text:
                        continue

                    chunks = chunk_text(
                        text
                    )

                    for chunk_index, chunk in enumerate(
                        chunks
                    ):

                        chunk = chunk.strip()

                        if len(chunk) < 100:
                            continue

                        web_chunks.append(
                            {
                                "text": chunk,
                                "source": title,
                                "url": url,
                                "title": title,
                                "page": "web",
                                "chunk_index": chunk_index,
                                "type": "web",
                                "knowledge_status": "web",
                            }
                        )

                        if len(web_chunks) >= 3:
                            break

                    if len(web_chunks) >= 3:
                        break

            # =================================================
            # PERSIST WEB KNOWLEDGE
            # =================================================
            #
            # THIS IS THE IMPORTANT NEW PART.
            #
            # Every newly retrieved web chunk is embedded
            # and stored in ChromaDB.
            #
            # Future questions can retrieve this knowledge
            # even when the original web result is gone.
            # =================================================

            if web_chunks:

                print()
                print(
                    "[AutoDiag] 💾 Persisting web knowledge..."
                )

                try:

                    stored_count = (
                        self.rag.ingest_organized_external_chunks(
                            chunks=web_chunks,
                            manufacturer=manufacturer,
                            vehicle=vehicle,
                            complaint=complaint,
                            dtc=dtc,
                            source_type="web",
                        )
                    )

                    print(
                        f"[AutoDiag] ✅ Persisted "
                        f"{stored_count} web chunks "
                        f"into ChromaDB."
                    )

                except Exception as error:

                    print(
                        "[AutoDiag] ⚠️ Web knowledge "
                        f"persistence failed: {error}"
                    )

            else:

                print(
                    "[AutoDiag] ⚠️ No web chunks "
                    "available to persist."
                )

            results = web_chunks[:3]

            print(
                f"[AutoDiag] Final web chunks: "
                f"{len(results)}"
            )

        # =================================================
        # 4. FINAL RESULTS
        # =================================================

        results = results[:3]

        print(
            f"[AutoDiag] Final evidence chunks: "
            f"{len(results)}"
        )

        # =================================================
        # 5. BUILD LLM CONTEXT
        # =================================================

        context = build_context(
            results
        )

        # =================================================
        # 6. BUILD PROMPT
        # =================================================

        user_prompt = build_diagnostic_prompt(
            vehicle=vehicle,
            complaint=complaint,
            dtc=dtc,
            context=context,
        )

        # =================================================
        # 7. OLLAMA
        # =================================================

        print(
            "[AutoDiag] Sending evidence to Ollama..."
        )

        answer = self.llm.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        print(
            "[AutoDiag] ✅ Diagnosis complete."
        )

        # =================================================
        # 8. RETURN RESULT
        # =================================================

        return {
            "vehicle": vehicle,
            "complaint": complaint,
            "dtc": dtc,
            "answer": answer,
            "sources": results,
            "web_sources": web_sources,
            "used_web_search": used_web_search,
            "search_query": search_query,
        }

    # =====================================================
    # MANUFACTURER EXTRACTION
    # =====================================================

    @staticmethod
    def _extract_manufacturer(
        vehicle: str,
    ) -> str:
        """
        Extract the manufacturer from the vehicle text.

        Examples:

            BMW 3 Series 2024
                -> bmw

            Maruthi Baleno 2024
                -> maruti

            Hyundai i20 2022
                -> hyundai

        The normalized manufacturer is used as the
        Knowledge Base organization key.
        """

        normalized = (
            " ".join(
                vehicle.lower().split()
            )
            .replace("-", " ")
        )

        aliases = [
            ("mercedes benz", "mercedes benz"),
            ("land rover", "land rover"),
            ("aston martin", "aston martin"),
            ("rolls royce", "rolls royce"),
            ("maruti suzuki", "maruti"),
            ("maruthi suzuki", "maruti"),
            ("volkswagen", "volkswagen"),
            ("mahindra", "mahindra"),
            ("chevrolet", "chevrolet"),
            ("mitsubishi", "mitsubishi"),
            ("toyota", "toyota"),
            ("hyundai", "hyundai"),
            ("renault", "renault"),
            ("nissan", "nissan"),
            ("honda", "honda"),
            ("skoda", "skoda"),
            ("tata", "tata"),
            ("kia", "kia"),
            ("jeep", "jeep"),
            ("volvo", "volvo"),
            ("audi", "audi"),
            ("lexus", "lexus"),
            ("jaguar", "jaguar"),
            ("porsche", "porsche"),
            ("suzuki", "suzuki"),
            ("maruti", "maruti"),
            ("maruthi", "maruti"),
            ("bmw", "bmw"),
            ("ford", "ford"),
            ("fiat", "fiat"),
            ("mg", "mg"),
        ]

        # Longest aliases first so "maruti suzuki"
        # is matched before "maruti".
        aliases.sort(
            key=lambda item: len(item[0]),
            reverse=True,
        )

        for alias, canonical in aliases:

            if (
                normalized == alias
                or normalized.startswith(
                    f"{alias} "
                )
            ):
                return canonical

        # -------------------------------------------------
        # Safe fallback
        # -------------------------------------------------

        words = normalized.split()

        if words:
            return words[0]

        return "unknown"


    # =====================================================
    # SEARCH QUERY
    # =====================================================

    @staticmethod
    def _build_search_query(
        vehicle: str,
        complaint: str,
        dtc: str = "",
    ) -> str:

        parts = [
            f'"{vehicle}"',
            f'"{complaint}"',
        ]

        if dtc:
            parts.append(
                f'"{dtc}"'
            )

        parts.append(
            "automotive diagnostic technical repair"
        )

        return " ".join(parts)

    # =====================================================
    # VEHICLE TOKENS
    # =====================================================

    @staticmethod
    def _vehicle_tokens(
        vehicle: str,
    ) -> List[str]:

        generic_words = {
            "car",
            "vehicle",
            "model",
            "year",
            "motor",
            "automotive",
            "petrol",
            "diesel",
            "gasoline",
            "automatic",
            "manual",
        }

        tokens = []

        normalized = (
            vehicle
            .lower()
            .replace("-", " ")
        )

        for word in normalized.split():

            cleaned = "".join(
                char
                for char in word
                if char.isalnum()
            )

            if not cleaned:
                continue

            if cleaned.isdigit():
                continue

            if len(cleaned) < 3:
                continue

            if cleaned in generic_words:
                continue

            tokens.append(cleaned)

        return tokens

    # =====================================================
    # VEHICLE MATCHING
    # =====================================================

    @classmethod
    def _vehicle_matches_text(
        cls,
        vehicle: str,
        text: str,
    ) -> bool:

        tokens = cls._vehicle_tokens(
            vehicle
        )

        if not tokens:
            return False

        normalized_text = (
            text.lower()
            .replace("-", " ")
            .replace("/", " ")
        )

        # Manufacturer match
        manufacturer = tokens[0]

        if manufacturer in normalized_text:
            return True

        # Model match
        for token in tokens[1:]:

            if token in normalized_text:
                return True

        return False

    # =====================================================
    # LOCAL RESULT FILTER
    # =====================================================

    @classmethod
    def _filter_vehicle_results(
        cls,
        results: List[Dict],
        vehicle: str,
    ) -> List[Dict]:

        filtered = []

        for result in results:

            searchable_text = (
                f"{result.get('source', '')} "
                f"{result.get('title', '')} "
                f"{result.get('text', '')}"
            )

            if cls._vehicle_matches_text(
                vehicle=vehicle,
                text=searchable_text,
            ):

                filtered.append(
                    result
                )

            else:

                print(
                    "[AutoDiag] Rejected local source:",
                    result.get(
                        "source",
                        "Unknown",
                    ),
                )

        return filtered

    # =====================================================
    # LOCAL RELEVANCE CHECK
    # =====================================================

    @classmethod
    def _has_useful_results(
        cls,
        results: List[Dict],
        vehicle: str,
        complaint: str,
        dtc: str,
    ) -> bool:

        if not results:
            return False

        complaint_words = {
            word.lower()
            for word in complaint.split()
            if len(word) >= 4
        }

        dtc_words = {
            word.lower()
            for word in dtc.split()
            if len(word) >= 3
        }

        stop_words = {
            "with",
            "when",
            "while",
            "from",
            "that",
            "this",
            "engine",
            "vehicle",
            "automotive",
            "diagnosis",
            "repair",
            "technical",
            "information",
        }

        complaint_words -= stop_words

        for result in results:

            text = result.get(
                "text",
                "",
            ).lower()

            if len(text) < 150:
                continue

            searchable = (
                f"{result.get('source', '')} "
                f"{text}"
            )

            if not cls._vehicle_matches_text(
                vehicle=vehicle,
                text=searchable,
            ):
                continue

            complaint_matches = sum(
                1
                for word in complaint_words
                if word in text
            )

            dtc_match = bool(
                dtc_words
                and any(
                    word in text
                    for word in dtc_words
                )
            )

            if dtc_match:
                return True

            if complaint_matches >= 2:
                return True

        return False