import hashlib
import re
from pathlib import Path
from typing import Dict, List


from pypdf import PdfReader

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.
    """

    if not text:
        return ""

    text = text.replace(
        "\x00",
        " ",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def extract_pdf_pages(
    pdf_path: str | Path,
) -> List[Dict]:
    """
    Extract text from a PDF page by page.
    """

    pdf_path = Path(pdf_path)

    reader = PdfReader(
        str(pdf_path)
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        try:
            text = page.extract_text() or ""

        except Exception:
            text = ""

        text = clean_text(text)

        if not text:
            continue

        pages.append(
            {
                "page": page_number,
                "text": text,
            }
        )

    return pages


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks.
    """

    text = clean_text(text)

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(
                chunk
            )

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def create_document_chunks(
    pdf_path: str | Path,
) -> List[Dict]:
    """
    Convert a PDF into searchable chunks.
    """

    pdf_path = Path(pdf_path)

    pages = extract_pdf_pages(
        pdf_path
    )

    all_chunks = []

    for page_data in pages:

        page_chunks = chunk_text(
            page_data["text"]
        )

        for index, chunk in enumerate(
            page_chunks
        ):

            chunk_id_source = (
                f"{pdf_path.name}|"
                f"{page_data['page']}|"
                f"{index}|"
                f"{chunk}"
            )

            chunk_id = hashlib.sha256(
                chunk_id_source.encode(
                    "utf-8"
                )
            ).hexdigest()

            all_chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "source": pdf_path.name,
                    "page": page_data["page"],
                    "chunk_index": index,
                }
            )

    return all_chunks


def create_text_chunks(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> List[Dict]:
    """
    Convert web page text into RAG-ready chunks.
    """

    if not text or not text.strip():
        return []

    cleaned_text = clean_text(text)

    if not cleaned_text:
        return []

    chunks = []

    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = min(
            start + chunk_size,
            text_length,
        )

        chunk_text_value = cleaned_text[start:end].strip()

        if chunk_text_value:
            chunks.append(
                {
                    "text": chunk_text_value,
                    "source": source,
                    "page": None,
                }
            )

        if end >= text_length:
            break

        start = end - overlap

    return chunks