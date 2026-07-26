import re
import sys

import pdfplumber

from extract_text import PDF_PATH, extract_text

sys.stdout.reconfigure(encoding="utf-8")

# Captions are line-wrapped in the extracted text, so match the label then take the
# following text up to a blank line or the start of the table's own contents.
CAPTION_PATTERN = re.compile(r"TABLE\s+([IVXLC]+)\s*[:.]\s*(.{0,160})", re.DOTALL)


def extract_captions(pdf_path: str) -> list[str]:
    """Pull each table's printed caption from the PDF text, in page order.

    These are the paper's own natural-language descriptions ("Distribution of dataset
    samples across eight tomato disease classes"), which is what a question actually
    resembles — a raw markdown grid of digits is not.
    """
    text = extract_text(pdf_path)
    captions = []
    for match in CAPTION_PATTERN.finditer(text):
        label, tail = match.group(1), match.group(2)
        tail = " ".join(tail.split("\n\n")[0].split())
        captions.append(f"Table {label}: {_trim_data_bleed(tail)}".strip())
    return captions


def _trim_data_bleed(text: str, max_numeric_run: int = 2) -> str:
    """Drop the tail once the caption runs into the table's own numeric rows.

    The caption is line-wrapped in the PDF, so a fixed-width match inevitably spills
    into data. Header words are useful retrieval signal; strings of digits are not.
    """
    words = text.split()
    numeric_run = 0
    for i, word in enumerate(words):
        if any(ch.isdigit() for ch in word):
            numeric_run += 1
            if numeric_run > max_numeric_run:
                return " ".join(words[: i - max_numeric_run]).rstrip(" ,;:")
        else:
            numeric_run = 0
    return text


def table_to_markdown(table: list[list[str]]) -> str:
    rows = [[cell or "" for cell in row] for row in table]
    header, *body = rows

    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def is_real_table(table: list[list[str]]) -> bool:
    if len(table) < 3:
        return False
    if len(table[0]) < 2:
        return False
    return True


def extract_tables(pdf_path: str) -> list[dict]:
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                if not is_real_table(table):
                    continue
                results.append({"page": page_num, "markdown": table_to_markdown(table)})

    # Tables and captions both come out in page order, so they pair up positionally.
    captions = extract_captions(pdf_path)
    for i, t in enumerate(results):
        t["caption"] = captions[i] if i < len(captions) else f"Table on page {t['page']}"

    return results


if __name__ == "__main__":
    tables = extract_tables(PDF_PATH)
    print(f"Tables found: {len(tables)}")
    for t in tables:
        print(f"--- page {t['page']} ---")
        print(f"CAPTION: {t['caption']}")
        print(t["markdown"][:200])
        print()
