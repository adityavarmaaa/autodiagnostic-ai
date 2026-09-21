from pathlib import Path
from typing import Dict, List
import hashlib
import json

import chromadb

from .config import CHROMA_PATH, TOP_K
from .documents import (
    create_document_chunks,
    create_text_chunks,
)
from .embeddings import EmbeddingService


COLLECTION_NAME = "autodiag_documents"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANUALS_DIR = (
    PROJECT_ROOT
    / "data"
    / "manuals"
)

MANIFEST_PATH = (
    CHROMA_PATH
    / "knowledge_manifest.json"
)


class RAGService:
    """
    Persistent Knowledge Base + RAG service.

    Important behavior:

    1. Local PDF/TXT files are ingested into ChromaDB.
    2. Once ingested, knowledge remains in ChromaDB.
    3. Deleting the original PDF/TXT does NOT delete the
       stored knowledge.
    4. Changed documents are re-indexed.
    5. New documents are automatically indexed.
    6. Web/external knowledge can also be stored persistently.
    """

    def __init__(
        self,
        chroma_path: Path = CHROMA_PATH,
    ):
        self.embedding_service = EmbeddingService()

        self.client = chromadb.PersistentClient(
            path=str(chroma_path)
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "description": (
                        "AutoDiag AI persistent "
                        "automotive knowledge base"
                    )
                },
            )
        )

    # =====================================================
    # BASIC
    # =====================================================

    def count(self) -> int:
        return self.collection.count()

    # =====================================================
    # ID HELPERS
    # =====================================================

    @staticmethod
    def _stable_id(
        prefix: str,
        source: str,
        index: int,
        text: str,
    ) -> str:
        """
        Generate a deterministic ID.

        Python's built-in hash() is intentionally not stable
        between processes, so SHA-256 is used instead.
        """

        raw = (
            f"{prefix}|"
            f"{source}|"
            f"{index}|"
            f"{text}"
        )

        digest = hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

        return f"{prefix}_{digest}"

    # =====================================================
    # SOURCE DELETION
    # =====================================================

    def _delete_source_chunks(
        self,
        source: str,
    ) -> None:
        """
        Delete existing chunks belonging to ONE source.

        This is used only when replacing/updating a source.

        It is NOT called when a local file disappears.
        """

        try:
            self.collection.delete(
                where={
                    "source": source,
                }
            )

            print(
                f"[Knowledge Base] Removed old chunks "
                f"for: {source}"
            )

        except Exception as error:
            print(
                f"[Knowledge Base] Could not remove "
                f"old chunks for {source}: {error}"
            )

    # =====================================================
    # PDF INGESTION
    # =====================================================

    def ingest_pdf(
        self,
        pdf_path: str | Path,
    ) -> Dict:
        """
        Ingest a PDF into the persistent Knowledge Base.

        The original PDF can later be deleted without
        deleting the stored knowledge.
        """

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        chunks = create_document_chunks(
            pdf_path
        )

        if not chunks:
            raise ValueError(
                f"No readable text found in "
                f"{pdf_path.name}. "
                "The PDF may be scanned/image-only."
            )

        source = pdf_path.name

        # If the document is being re-indexed,
        # remove its previous version first.
        self._delete_source_chunks(
            source
        )

        ids = []
        documents = []
        metadatas = []

        for index, item in enumerate(chunks):
            text = item["text"]

            chunk_id = self._stable_id(
                prefix="pdf",
                source=source,
                index=index,
                text=text,
            )

            ids.append(chunk_id)

            documents.append(text)

            metadatas.append(
                {
                    "source": source,
                    "page": item["page"],
                    "chunk_index": index,
                    "type": "pdf",
                    "knowledge_status": "persistent",
                    "title": source,
                    "url": "",
                }
            )

        print(
            f"[Knowledge Base] Embedding "
            f"{len(documents)} PDF chunks..."
        )

        embeddings = (
            self.embedding_service.embed(
                documents
            )
        )

        if len(embeddings) != len(
            documents
        ):
            raise RuntimeError(
                "Embedding count does not "
                "match document count."
            )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        print(
            f"[Knowledge Base] Successfully "
            f"stored {len(documents)} chunks "
            f"from {source}."
        )

        return {
            "source": source,
            "chunks": len(documents),
            "type": "pdf",
        }

    # =====================================================
    # TEXT INGESTION
    # =====================================================

    def ingest_text(
        self,
        text: str,
        source: str,
    ) -> int:
        """
        Persist text knowledge into ChromaDB.

        This can be used for:
        - local TXT files
        - web pages
        - trusted technical sources
        """

        text = text.strip()
        source = source.strip()

        if not text:
            return 0

        if not source:
            source = "Unknown source"

        chunks = create_text_chunks(
            text=text,
            source=source,
        )

        if not chunks:
            return 0

        # Replace previous version of this source.
        self._delete_source_chunks(
            source
        )

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        ids = []
        metadatas = []

        for index, chunk in enumerate(
            chunks
        ):
            chunk_text = chunk["text"]

            chunk_id = self._stable_id(
                prefix="text",
                source=source,
                index=index,
                text=chunk_text,
            )

            ids.append(chunk_id)

            metadatas.append(
                {
                    "source": source,
                    "page": "",
                    "chunk_index": index,
                    "type": "web",
                    "knowledge_status": "persistent",
                    "title": source,
                    "url": "",
                }
            )

        print(
            f"[Knowledge Base] Embedding "
            f"{len(documents)} text chunks..."
        )

        embeddings = (
            self.embedding_service.embed(
                documents
            )
        )

        if len(embeddings) != len(
            documents
        ):
            raise RuntimeError(
                "Embedding count does not "
                "match text document count."
            )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        print(
            f"[Knowledge Base] Stored "
            f"{len(documents)} persistent "
            f"chunks from: {source}"
        )

        return len(documents)

    # =====================================================
    # EXTERNAL / WEB KNOWLEDGE INGESTION
    # =====================================================

    def ingest_external_chunks(
        self,
        chunks: List[Dict],
        source_type: str = "web",
    ) -> int:
        """
        Persist externally fetched knowledge.

        Used for:
        - Tavily web results
        - API results
        - other external sources

        Once stored, the source file/page does not need
        to be available again for future RAG retrieval.
        """

        if not chunks:
            return 0

        documents = []
        metadatas = []
        ids = []

        for index, chunk in enumerate(chunks):

            text = (
                chunk.get("text", "")
                or ""
            ).strip()

            if not text:
                continue

            source = (
                chunk.get(
                    "source",
                    "External source",
                )
                or "External source"
            ).strip()

            title = (
                chunk.get(
                    "title",
                    source,
                )
                or source
            ).strip()

            url = (
                chunk.get(
                    "url",
                    "",
                )
                or ""
            ).strip()

            chunk_index = chunk.get(
                "chunk_index",
                index,
            )

            documents.append(text)

            metadatas.append(
                {
                    "source": source,
                    "title": title,
                    "url": url,
                    "page": "web",
                    "chunk_index": chunk_index,
                    "type": source_type,
                    "knowledge_status": "persistent",
                }
            )

            raw_id = (
                f"{source_type}|"
                f"{url}|"
                f"{source}|"
                f"{chunk_index}|"
                f"{text}"
            )

            digest = hashlib.sha256(
                raw_id.encode("utf-8")
            ).hexdigest()

            ids.append(
                f"{source_type}_{digest}"
            )

        if not documents:
            return 0

        print(
            f"[Knowledge Base] Embedding "
            f"{len(documents)} external chunks..."
        )

        embeddings = (
            self.embedding_service.embed(
                documents
            )
        )

        if len(embeddings) != len(
            documents
        ):
            raise RuntimeError(
                "Embedding count does not "
                "match external document count."
            )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        print(
            f"[Knowledge Base] ✅ Stored "
            f"{len(documents)} external chunks "
            f"persistently."
        )

        return len(documents)


    # =====================================================
    # ORGANIZED EXTERNAL / WEB KNOWLEDGE
    # =====================================================

    def ingest_organized_external_chunks(
        self,
        chunks: List[Dict],
        manufacturer: str,
        vehicle: str,
        complaint: str,
        dtc: str = "",
        source_type: str = "web",
    ) -> int:
        """
        Persist external knowledge with vehicle/issue metadata.

        Logical organization:

            manufacturer
                -> vehicle
                    -> issue (DTC or complaint)

        Example:

            bmw
                -> bmw 3 series 2024
                    -> P0300
                    -> engine overheating

        This does NOT create new ChromaDB collections.
        It keeps one collection and organizes documents
        using metadata, which is much easier to search.
        """

        if not chunks:
            return 0

        manufacturer = (
            manufacturer.strip().lower()
            or "unknown"
        )

        vehicle = (
            vehicle.strip()
            or "unknown vehicle"
        )

        complaint = (
            complaint.strip()
            or "unknown issue"
        )

        dtc = dtc.strip()

        issue_reference = (
            dtc
            if dtc
            else complaint
        )

        issue_key = (
            f"{manufacturer}|"
            f"{vehicle}|"
            f"{issue_reference}"
        )

        documents = []
        metadatas = []
        ids = []

        for index, chunk in enumerate(chunks):

            text = str(
                chunk.get(
                    "text",
                    "",
                )
                or ""
            ).strip()

            if not text:
                continue

            source = str(
                chunk.get(
                    "source",
                    "External source",
                )
                or "External source"
            ).strip()

            title = str(
                chunk.get(
                    "title",
                    source,
                )
                or source
            ).strip()

            url = str(
                chunk.get(
                    "url",
                    "",
                )
                or ""
            ).strip()

            page = chunk.get(
                "page",
                "web",
            )

            chunk_index = chunk.get(
                "chunk_index",
                index,
            )

            # ---------------------------------------------
            # Stable ID
            # ---------------------------------------------
            #
            # Same vehicle + same issue + same source +
            # same chunk = same ID.
            #
            # Re-checking the same issue therefore updates
            # the existing knowledge instead of blindly
            # creating another copy.
            # ---------------------------------------------

            chunk_id = self._stable_id(
                prefix="organized",
                source=(
                    f"{issue_key}|"
                    f"{url}|"
                    f"{source}"
                ),
                index=int(
                    chunk_index
                    if str(chunk_index).isdigit()
                    else index
                ),
                text=text,
            )

            documents.append(text)
            ids.append(chunk_id)

            metadatas.append(
                {
                    # Organization
                    "manufacturer": manufacturer,
                    "vehicle": vehicle,
                    "issue_key": issue_key,
                    "complaint": complaint,
                    "dtc": dtc,

                    # Source
                    "source": source,
                    "title": title,
                    "url": url,
                    "page": str(page),

                    # Chunk
                    "chunk_index": int(
                        chunk_index
                        if str(chunk_index).isdigit()
                        else index
                    ),

                    # Existing metadata contract
                    "type": source_type,
                    "source_type": source_type,
                    "knowledge_status": "persistent",
                }
            )

        if not documents:
            return 0

        print()
        print(
            "[Knowledge Base] "
            "Organizing external knowledge:"
        )
        print(
            f"  Manufacturer : {manufacturer}"
        )
        print(
            f"  Vehicle      : {vehicle}"
        )
        print(
            f"  Issue        : "
            f"{dtc or complaint}"
        )
        print(
            f"  Chunks       : {len(documents)}"
        )

        print(
            "[Knowledge Base] Embedding "
            f"{len(documents)} organized chunks..."
        )

        embeddings = (
            self.embedding_service.embed(
                documents
            )
        )

        if len(embeddings) != len(
            documents
        ):
            raise RuntimeError(
                "Embedding count does not "
                "match organized document count."
            )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        print(
            "[Knowledge Base] ✅ Stored "
            f"{len(documents)} organized "
            "chunks persistently."
        )

        print(
            "[Knowledge Base] 📁 "
            f"{manufacturer.upper()}"
        )

        print(
            "[Knowledge Base] └── "
            f"{vehicle}"
        )

        print(
            "[Knowledge Base]     └── "
            f"{dtc or complaint}"
        )

        return len(documents)

    # =====================================================
    # MANIFEST
    # =====================================================

    def _load_manifest(self) -> Dict:
        """
        Load the local-document manifest.

        IMPORTANT:
        The manifest tracks files that have been ingested.
        It does NOT control deletion of knowledge.
        """

        if not MANIFEST_PATH.exists():
            return {}

        try:
            with open(
                MANIFEST_PATH,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            if isinstance(data, dict):
                return data

        except Exception as error:
            print(
                f"[Knowledge Base] Could not read "
                f"manifest: {error}"
            )

        return {}

    def _save_manifest(
        self,
        manifest: Dict,
    ) -> None:

        MANIFEST_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = (
            MANIFEST_PATH.with_suffix(
                ".tmp"
            )
        )

        with open(
            temporary_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                manifest,
                file,
                indent=2,
                ensure_ascii=False,
            )

        temporary_path.replace(
            MANIFEST_PATH
        )

    # =====================================================
    # FILE HASH
    # =====================================================

    @staticmethod
    def _file_hash(
        path: Path,
    ) -> str:

        sha256 = hashlib.sha256()

        with open(
            path,
            "rb",
        ) as file:

            while True:

                block = file.read(
                    1024 * 1024
                )

                if not block:
                    break

                sha256.update(
                    block
                )

        return sha256.hexdigest()

    # =====================================================
    # LOCAL DOCUMENT SYNC
    # =====================================================

    def sync_local_documents(self) -> Dict:
        """
        Scan data/manuals/ and add/update knowledge.

        IMPORTANT:

        Deleted local files are NOT removed from ChromaDB.

        Example:

            hyundai_manual.pdf
                    ↓
                 ChromaDB
                    ↓
              delete PDF
                    ↓
            ChromaDB remains ✅

        This creates persistent knowledge.
        """

        MANUALS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest = self._load_manifest()

        added = []
        changed = []
        skipped = []
        failed = []

        files = []

        # -------------------------------------------------
        # Find supported local documents
        # -------------------------------------------------

        files.extend(
            MANUALS_DIR.glob("*.pdf")
        )

        files.extend(
            MANUALS_DIR.glob("*.txt")
        )

        files = sorted(
            files,
            key=lambda path: path.name.lower()
        )

        print()
        print("=" * 60)
        print(
            "AUTODIAG AI PERSISTENT "
            "KNOWLEDGE BASE SYNC"
        )
        print("=" * 60)

        print(
            f"Knowledge directory: {MANUALS_DIR}"
        )

        print(
            f"Documents found: {len(files)}"
        )

        # -------------------------------------------------
        # Process each file
        # -------------------------------------------------

        for file_path in files:

            source = file_path.name

            print()
            print(
                f"[Knowledge Base] Checking: "
                f"{source}"
            )

            try:

                current_hash = (
                    self._file_hash(
                        file_path
                    )
                )

                previous = manifest.get(
                    source
                )

                previous_hash = ""

                if isinstance(
                    previous,
                    dict,
                ):

                    previous_hash = (
                        previous.get(
                            "hash",
                            "",
                        )
                    )

                # -----------------------------------------
                # Unchanged
                # -----------------------------------------

                if (
                    previous_hash
                    and previous_hash
                    == current_hash
                ):

                    skipped.append(
                        {
                            "source": source,
                        }
                    )

                    print(
                        f"[Knowledge Base] "
                        f"Already indexed: "
                        f"{source}"
                    )

                    continue

                # -----------------------------------------
                # Determine new vs changed
                # -----------------------------------------

                is_new = (
                    previous is None
                    or not previous_hash
                )

                # -----------------------------------------
                # PDF
                # -----------------------------------------

                if (
                    file_path.suffix.lower()
                    == ".pdf"
                ):

                    result = self.ingest_pdf(
                        file_path
                    )

                    chunk_count = result[
                        "chunks"
                    ]

                    knowledge_type = "pdf"

                # -----------------------------------------
                # TXT
                # -----------------------------------------

                else:

                    text = file_path.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )

                    chunk_count = (
                        self.ingest_text(
                            text=text,
                            source=source,
                        )
                    )

                    knowledge_type = "txt"

                # -----------------------------------------
                # Update manifest
                # -----------------------------------------

                manifest[source] = {
                    "hash": current_hash,
                    "chunks": chunk_count,
                    "type": knowledge_type,
                    "knowledge_status": (
                        "persistent"
                    ),
                }

                if is_new:

                    added.append(
                        {
                            "source": source,
                            "chunks": chunk_count,
                        }
                    )

                    print(
                        f"[Knowledge Base] "
                        f"NEW: {source} "
                        f"({chunk_count} chunks)"
                    )

                else:

                    changed.append(
                        {
                            "source": source,
                            "chunks": chunk_count,
                        }
                    )

                    print(
                        f"[Knowledge Base] "
                        f"UPDATED: {source} "
                        f"({chunk_count} chunks)"
                    )

            except Exception as error:

                failed.append(
                    {
                        "source": source,
                        "error": str(error),
                    }
                )

                print(
                    f"[Knowledge Base] FAILED: "
                    f"{source}: {error}"
                )

        # -------------------------------------------------
        # IMPORTANT:
        #
        # We deliberately DO NOT inspect the manifest
        # and delete missing sources.
        #
        # That is what makes knowledge persistent.
        # -------------------------------------------------

        self._save_manifest(
            manifest
        )

        print()
        print("=" * 60)
        print(
            "PERSISTENT KNOWLEDGE BASE SYNC COMPLETE"
        )
        print("=" * 60)

        print(
            f"New documents     : {len(added)}"
        )

        print(
            f"Changed documents : {len(changed)}"
        )

        print(
            f"Unchanged         : {len(skipped)}"
        )

        print(
            f"Failed            : {len(failed)}"
        )

        print(
            f"Total KB chunks   : {self.count()}"
        )

        print("=" * 60)

        return {
            "added": added,
            "changed": changed,
            "skipped": skipped,
            "failed": failed,
            "total_chunks": self.count(),
        }


    # =====================================================
    # ORGANIZED SEARCH
    # =====================================================

    def search_organized(
        self,
        query: str,
        manufacturer: str,
        top_k: int = TOP_K,
    ) -> List[Dict]:
        """
        Search persistent knowledge only inside one
        manufacturer.

        Example:

            manufacturer = "bmw"
            query = "BMW 3 Series P0300"

        This keeps retrieval focused when the Knowledge
        Base contains many manufacturers and issues.
        """

        query = query.strip()

        manufacturer = (
            manufacturer.strip().lower()
        )

        if not query:
            return []

        if not manufacturer:
            return []

        if self.count() == 0:
            return []

        query_embedding = (
            self.embedding_service.embed_one(
                query
            )
        )

        try:
            results = self.collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=top_k,
                where={
                    "manufacturer": manufacturer
                },
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

        except Exception as error:
            print(
                "[Knowledge Base] "
                f"Organized search failed: "
                f"{error}"
            )
            return []

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        output = []

        for index, document in enumerate(
            documents
        ):

            metadata = (
                metadatas[index]
                if index < len(
                    metadatas
                )
                else {}
            )

            distance = (
                distances[index]
                if index < len(
                    distances
                )
                else None
            )

            output.append(
                {
                    "text": document,

                    "source": metadata.get(
                        "source",
                        "Unknown",
                    ),

                    "title": metadata.get(
                        "title",
                        metadata.get(
                            "source",
                            "",
                        ),
                    ),

                    "url": metadata.get(
                        "url",
                        "",
                    ),

                    "page": metadata.get(
                        "page",
                        "?",
                    ),

                    "chunk_index": metadata.get(
                        "chunk_index",
                        "?",
                    ),

                    "type": metadata.get(
                        "type",
                        "unknown",
                    ),

                    "knowledge_status": (
                        metadata.get(
                            "knowledge_status",
                            "persistent",
                        )
                    ),

                    # Organization metadata
                    "manufacturer": metadata.get(
                        "manufacturer",
                        "",
                    ),

                    "vehicle": metadata.get(
                        "vehicle",
                        "",
                    ),

                    "issue_key": metadata.get(
                        "issue_key",
                        "",
                    ),

                    "complaint": metadata.get(
                        "complaint",
                        "",
                    ),

                    "dtc": metadata.get(
                        "dtc",
                        "",
                    ),

                    "distance": distance,
                }
            )

        return output

    # =====================================================
    # SEARCH
    # =====================================================

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> List[Dict]:
        """
        Search the persistent Knowledge Base.
        """

        if not query.strip():
            return []

        if self.count() == 0:
            return []

        query_embedding = (
            self.embedding_service.embed_one(
                query
            )
        )

        results = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        output = []

        for index, document in enumerate(
            documents
        ):

            metadata = (
                metadatas[index]
                if index < len(
                    metadatas
                )
                else {}
            )

            distance = (
                distances[index]
                if index < len(
                    distances
                )
                else None
            )

            output.append(
                {
                    "text": document,

                    "source": metadata.get(
                        "source",
                        "Unknown",
                    ),

                    "title": metadata.get(
                        "title",
                        metadata.get(
                            "source",
                            "",
                        ),
                    ),

                    "url": metadata.get(
                        "url",
                        "",
                    ),

                    "page": metadata.get(
                        "page",
                        "?",
                    ),

                    "chunk_index": metadata.get(
                        "chunk_index",
                        "?",
                    ),

                    "type": metadata.get(
                        "type",
                        "unknown",
                    ),

                    "knowledge_status": (
                        metadata.get(
                            "knowledge_status",
                            "persistent",
                        )
                    ),

                    # Organization metadata.
                    # Old records simply return empty values.
                    "manufacturer": metadata.get(
                        "manufacturer",
                        "",
                    ),

                    "vehicle": metadata.get(
                        "vehicle",
                        "",
                    ),

                    "issue_key": metadata.get(
                        "issue_key",
                        "",
                    ),

                    "complaint": metadata.get(
                        "complaint",
                        "",
                    ),

                    "dtc": metadata.get(
                        "dtc",
                        "",
                    ),

                    "distance": distance,
                }
            )

        return output

    # =====================================================
    # CLEAR EVERYTHING
    # =====================================================

    def clear(self) -> None:
        """
        Explicitly clear the entire persistent KB.

        This is the ONLY operation that intentionally
        removes stored knowledge.
        """

        try:

            self.client.delete_collection(
                COLLECTION_NAME
            )

        except Exception:
            pass

        self.collection = (
            self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "description": (
                        "AutoDiag AI persistent "
                        "automotive knowledge base"
                    )
                },
            )
        )

        # Remove manifest because the KB was intentionally
        # cleared.
        try:

            if MANIFEST_PATH.exists():
                MANIFEST_PATH.unlink()

        except Exception:
            pass

        print(
            "[Knowledge Base] Persistent "
            "knowledge base cleared."
        )


# =========================================================
# CONTEXT BUILDER
# =========================================================

def build_context(
    results: List[Dict],
) -> str:
    """
    Build LLM context from retrieved persistent knowledge.
    """

    if not results:

        return (
            "No technical documents were retrieved."
        )

    sections = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        sections.append(
            f"""
SOURCE {index}

Document:
{result["source"]}

Title:
{result.get("title", "")}

URL:
{result.get("url", "")}

Page:
{result["page"]}

Type:
{result.get("type", "unknown")}

Knowledge Status:
{result.get("knowledge_status", "persistent")}

Chunk:
{result["text"]}
""".strip()
        )

    return "\n\n---\n\n".join(
        sections
    )