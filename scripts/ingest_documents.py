from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import MANUALS_DIR
from backend.rag import RAGService


def main() -> None:
    """
    Ingest PDF and TXT knowledge files into ChromaDB.
    """

    MANUALS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_files = sorted(
        MANUALS_DIR.glob("*.pdf")
    )

    txt_files = sorted(
        MANUALS_DIR.glob("*.txt")
    )

    knowledge_files = pdf_files + txt_files

    if not knowledge_files:
        print(
            f"No PDF or TXT files found in: "
            f"{MANUALS_DIR}"
        )
        return

    rag = RAGService()

    print(
        f"Found {len(knowledge_files)} "
        f"knowledge file(s)."
    )

    total_chunks = 0

    for file_path in knowledge_files:
        print()
        print(
            f"Processing: {file_path.name}"
        )

        try:
            extension = file_path.suffix.lower()

            if extension == ".pdf":
                result = rag.ingest_pdf(
                    file_path
                )

                chunks = result["chunks"]

            elif extension == ".txt":
                text = file_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                if not text.strip():
                    raise ValueError(
                        f"Text file is empty: "
                        f"{file_path.name}"
                    )

                chunks = rag.ingest_text(
                    text=text,
                    source=file_path.name,
                )

            else:
                print(
                    f"Skipping unsupported file: "
                    f"{file_path.name}"
                )
                continue

            total_chunks += chunks

            print(
                f"Successfully indexed "
                f"{chunks} chunks."
            )

        except Exception as exc:
            print(
                f"ERROR processing "
                f"{file_path.name}:"
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