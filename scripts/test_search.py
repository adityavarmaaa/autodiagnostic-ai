from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.rag import RAGService


def main():
    rag = RAGService()

    results = rag.search(
        "P0301 engine misfire cylinder 1",
        3,
    )

    print("Results:", len(results))

    for result in results:
        print("---")
        print("Source:", result["source"])
        print("Page:", result["page"])
        print("Text:")
        print(result["text"])


if __name__ == "__main__":
    main()