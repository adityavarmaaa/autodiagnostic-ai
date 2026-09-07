from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "manuals"
OUTPUT_FILE = OUTPUT_DIR / "test_diagnostic.pdf"


def create_pdf():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf = canvas.Canvas(
        str(OUTPUT_FILE),
        pagesize=A4,
    )

    width, height = A4

    x = 50
    y = height - 50

    lines = [
        "AutoDiag AI Test Diagnostic Document",
        "",
        "Vehicle System: Engine",
        "",
        "DTC: P0301",
        "",
        "Description:",
        "P0301 indicates that a misfire has been detected",
        "on cylinder 1.",
        "",
        "Possible diagnostic areas:",
        "- Ignition system",
        "- Fuel delivery",
        "- Injector operation",
        "- Engine mechanical condition",
        "- Air or vacuum leaks",
        "- Wiring and electrical connections",
        "",
        "Recommended diagnostic approach:",
        "1. Confirm the DTC and check for additional codes.",
        "2. Inspect ignition components associated with cylinder 1.",
        "3. Check whether the misfire follows an ignition component",
        "   when the applicable service procedure permits this test.",
        "4. Check fuel injector operation and related wiring.",
        "5. Check for intake or vacuum leaks.",
        "6. Perform appropriate engine mechanical tests if needed.",
        "",
        "Parts replacement:",
        "Do not replace an ignition coil, spark plug, injector,",
        "or other component solely because P0301 is present.",
        "Confirm the failed component through appropriate testing.",
        "",
        "Important:",
        "P0301 is a diagnostic clue and does not by itself prove",
        "that a specific component has failed.",
        "",
        "Customer explanation:",
        "The vehicle computer has detected a misfire affecting",
        "cylinder 1. We need to perform checks to determine",
        "the underlying cause before replacing parts.",
    ]

    for line in lines:

        if y < 50:
            pdf.showPage()
            y = height - 50

        pdf.drawString(
            x,
            y,
            line,
        )

        y -= 16

    pdf.save()

    print(
        f"Created test PDF: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    create_pdf()