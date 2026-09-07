import os
from pathlib import Path

from dotenv import load_dotenv


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


# Ollama
OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama3.2:3b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "embeddinggemma"
)


# Storage
CHROMA_PATH = BASE_DIR / os.getenv(
    "CHROMA_PATH",
    "data/chroma"
)

SQLITE_PATH = BASE_DIR / os.getenv(
    "SQLITE_PATH",
    "data/autodiag.db"
)


# RAG
TOP_K = int(
    os.getenv("TOP_K", "5")
)

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "1200")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "200")
)


# Directories
MANUALS_DIR = BASE_DIR / "data" / "manuals"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def ensure_directories() -> None:
    """
    Create required application directories.
    """

    MANUALS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHROMA_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    SQLITE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )