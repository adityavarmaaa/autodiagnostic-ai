from pathlib import Path
from typing import Dict, List

import chromadb

from .config import CHROMA_PATH, TOP_K
from .documents import create_document_chunks
from .embeddings import EmbeddingService


COLLECTION_NAME = "autodiag_documents"


class RAGService:
    """
    Handles document indexing and semantic search.
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
                        "AutoDiag AI automotive "
                        "technical knowledge base"
                    )
                },
            )
        )

    def count(self) -> int:
        """
        Return number of indexed document chunks.
        """

        return self.collection.count()

    def ingest_pdf(
        self,
        pdf_path: str | Path,
    ) -> Dict:
        """
        Read a PDF, create chunks, generate embeddings,
        and store everything in ChromaDB.
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

        ids = [
            item["id"]
            for item in chunks
        ]

        documents = [
            item["text"]
            for item in chunks
        ]

        metadatas = [
            {
                "source": item["source"],
                "page": item["page"],
                "chunk_index": item["chunk_index"],
            }
            for item in chunks
        ]

        print(
            f"Generating embeddings for "
            f"{len(documents)} chunks..."
        )

        embeddings = (
            self.embedding_service.embed(
                documents
            )
        )

        if len(embeddings) != len(documents):
            raise RuntimeError(
                "Embedding count does not match "
                "document count."
            )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return {
            "source": pdf_path.name,
            "chunks": len(chunks),
        }

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> List[Dict]:
        """
        Search the knowledge base using semantic similarity.
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
                if index < len(metadatas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            output.append(
                {
                    "text": document,
                    "source": metadata.get(
                        "source",
                        "Unknown",
                    ),
                    "page": metadata.get(
                        "page",
                        "?",
                    ),
                    "chunk_index": metadata.get(
                        "chunk_index",
                        "?",
                    ),
                    "distance": distance,
                }
            )

        return output

    def clear(self) -> None:
        """
        Delete the knowledge collection and recreate it.
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
                        "AutoDiag AI automotive "
                        "technical knowledge base"
                    )
                },
            )
        )


def build_context(
    results: List[Dict],
) -> str:
    """
    Convert retrieved documents into LLM context.
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
Document: {result['source']}
Page: {result['page']}

{result['text']}
""".strip()
        )

    return "\n\n---\n\n".join(
        sections
    )