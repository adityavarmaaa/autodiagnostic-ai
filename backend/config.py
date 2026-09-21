import os
from pathlib import Path

from dotenv import load_dotenv


# =========================================================
# BASE PROJECT DIRECTORY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


# =========================================================
# OLLAMA SETTINGS
# =========================================================

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434",
)

# Use a faster model by default.
# You can override this in .env.
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "phi3:mini",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "embeddinggemma",
)


# =========================================================
# DATABASE SETTINGS
# =========================================================

CHROMA_PATH = BASE_DIR / os.getenv(
    "CHROMA_PATH",
    "data/chroma",
)

SQLITE_PATH = BASE_DIR / os.getenv(
    "SQLITE_PATH",
    "data/autodiag.db",
)


# =========================================================
# RAG SETTINGS
# =========================================================

# Number of documents/chunks retrieved for diagnosis.
# Lower value = smaller LLM context = faster response.
TOP_K = int(
    os.getenv("TOP_K", "3")
)

# Chunk size used when processing manuals/web pages.
CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "1200")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "200")
)


# =========================================================
# LOCAL DOCUMENT DIRECTORIES
# =========================================================

MANUALS_DIR = (
    BASE_DIR
    / "data"
    / "manuals"
)

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


# =========================================================
# WEB SEARCH SETTINGS
# =========================================================

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY",
    "",
)

# Keep web fallback small for faster responses.
WEB_MAX_RESULTS = int(
    os.getenv("WEB_MAX_RESULTS", "2")
)

# Don't let a slow website hold up the entire diagnosis.
WEB_REQUEST_TIMEOUT = int(
    os.getenv("WEB_REQUEST_TIMEOUT", "8")
)


# =========================================================
# DIRECTORY INITIALIZATION
# =========================================================

def ensure_directories() -> None:
    """
    Make sure all required project directories exist.
    """

    MANUALS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHROMA_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    SQLITE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )