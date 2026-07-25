import fitz  # PyMuPDF

PDF_PATH = "data/A Dual-Stream Multimodal Framework for Tomato Disease Detection in Unstructured Field Environments.pdf"


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
