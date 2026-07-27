import os

import fitz  # PyMuPDF

# Only a convenience default for `python src/ingest.py` with no argument; uploads and
# the compose ingest service pass an explicit path. Override with DEFAULT_DOCUMENT.
DEFAULT_DOCUMENT = os.getenv(
    "DEFAULT_DOCUMENT",
    "data/A Dual-Stream Multimodal Framework for Tomato Disease Detection in Unstructured Field Environments.pdf",
)

# Retained under the old name so existing imports keep working.
PDF_PATH = DEFAULT_DOCUMENT


def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages_text = [page.get_text() for page in doc]
    return "\n".join(pages_text)


if __name__ == "__main__":
    full_text = extract_text(PDF_PATH)

    print(f"Total characters: {len(full_text)}")
    print("--- First 500 chars ---")
    print(full_text[:500])

    with open("data/extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
