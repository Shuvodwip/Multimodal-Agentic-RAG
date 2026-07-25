import fitz  # PyMuPDF

PDF_PATH = "data/A Dual-Stream Multimodal Framework for Tomato Disease Detection in Unstructured Field Environments.pdf"

doc = fitz.open(PDF_PATH)

pages_text = []
for page in doc:
    pages_text.append(page.get_text())

full_text = "\n".join(pages_text)

print(f"Pages: {len(doc)}")
print(f"Total characters: {len(full_text)}")
print("--- First 500 chars ---")
print(full_text[:500])
