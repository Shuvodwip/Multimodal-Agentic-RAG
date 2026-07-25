import os

import fitz  # PyMuPDF

from extract_text import PDF_PATH

OUTPUT_DIR = "data/images"


def extract_images(pdf_path: str, output_dir: str) -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    saved = []
    for page_num, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            filename = f"page{page_num}_img{img_index}.{ext}"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "wb") as f:
                f.write(image_bytes)

            saved.append({"page": page_num, "path": filepath})

    return saved


if __name__ == "__main__":
    images = extract_images(PDF_PATH, OUTPUT_DIR)
    print(f"Images extracted: {len(images)}")
    for img in images:
        print(f"  page {img['page']}: {img['path']}")
