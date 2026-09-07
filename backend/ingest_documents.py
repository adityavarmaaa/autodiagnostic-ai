from pathlib import Path
import sys


# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from backend.config import MANUALS_DIR
from backend.rag import RAGService


def main() -> None:
    """
    Ingest all PDF files inside data/manuals.
    """

    MANUALS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_files = sorted(
        MANUALS_DIR.glob("*.pdf")
    )

    if not pdf_files:
        print(
            f"No PDF files found in: "
            f"{MANUALS_DIR}"
        )

        print(
            "Put an automotive technical PDF "
            "inside data/manuals and run this "
            "script again."
        )

        return

    rag = RAGService()

    print(
        f"Found {len(pdf_files)} PDF file(s)."
    )

    total_chunks = 0

    for pdf_path in pdf_files:

        print()
        print(
            f"Processing: {pdf_path.name}"
        )

        try:

            result = rag.ingest_pdf(
                pdf_path
            )

            chunks = result["chunks"]

            total_chunks += chunks

            print(
                f"Successfully indexed "
                f"{chunks} chunks."
            )

        except Exception as exc:

            print(
                f"ERROR processing "
                f"{pdf_path.name}:"
            )

            print(exc)

    print()
    print(
        f"Total chunks added: "
        f"{total_chunks}"
    )

    print(
        f"Total chunks in database: "
        f"{rag.count()}"
    )


if __name__ == "__main__":
    main()