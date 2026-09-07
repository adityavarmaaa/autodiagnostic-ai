from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.diagnostic import DiagnosticService


def main():
    diagnostic = DiagnosticService()

    print("=" * 60)
    print("AutoDiag AI - Diagnostic Test")
    print("=" * 60)

    result = diagnostic.analyze(
        vehicle="Hyundai i20 2019",
        complaint="Engine shaking at idle",
        dtc="P0301",
    )

    print()
    print("VEHICLE:")
    print(result["vehicle"])

    print()
    print("COMPLAINT:")
    print(result["complaint"])

    print()
    print("DTC:")
    print(result["dtc"])

    print()
    print("=" * 60)
    print("AI DIAGNOSTIC REPORT")
    print("=" * 60)

    print(result["answer"])

    print()
    print("=" * 60)
    print("SOURCES")
    print("=" * 60)

    for source in result["sources"]:
        print(
            f"- {source['source']} "
            f"(page {source['page']})"
        )


if __name__ == "__main__":
    main()