import re
import sys

import fitz
import pdfplumber

from extract_text import PDF_PATH

sys.stdout.reconfigure(encoding="utf-8")

# Matches the caption styles documents actually use: "TABLE IV:" (Roman, common in
# IEEE papers), "Table 1." (Arabic, most reports), "Table A2 —", and so on.
#
# Anchored to the start of a line, which is what separates a caption from a passing
# reference: captions sit on their own line, whereas "as detailed in Table I" appears
# mid-sentence. Without the anchor, body-text references get captured as captions and
# every table on the page is described by the wrong text.
CAPTION_PATTERN = re.compile(
    r"^\s*Table\s+([IVXLC]+|[A-Z]?\.?\d+[a-z]?)\s*[:.–—-]?\s+(.{0,200})",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def extract_captions_by_page(pdf_path: str) -> dict[int, list[str]]:
    """Find printed table captions, grouped by the page they appear on.

    Matching captions to tables per page rather than by global order keeps them
    aligned when a document has captionless tables, or captions the pattern misses —
    a mismatch would otherwise shift every later caption onto the wrong table.
    """
    doc = fitz.open(pdf_path)
    by_page: dict[int, list[str]] = {}

    for page_num, page in enumerate(doc, start=1):
        captions = []
        for match in CAPTION_PATTERN.finditer(page.get_text()):
            label, tail = match.group(1), match.group(2)
            tail = " ".join(tail.split("\n\n")[0].split())
            captions.append(f"Table {label}: {_trim_data_bleed(tail)}".strip().rstrip(":"))
        if captions:
            by_page[page_num] = captions

    doc.close()
    return by_page


def describe_from_headers(table: list[list[str]], page: int) -> str:
    """Synthesise a caption from the table's own header row.

    Plenty of documents have unlabelled tables. Embedding a bare grid of digits makes
    it effectively unretrievable, so give it *some* natural-language description —
    column names are the most meaningful text a table carries on its own.
    """
    header = [str(cell).strip() for cell in table[0] if cell and str(cell).strip()]
    if not header:
        return f"Table on page {page}"
    return f"Table on page {page} with columns: {', '.join(header)}"


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
    captions_by_page = extract_captions_by_page(pdf_path)
    results = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_captions = captions_by_page.get(page_num, [])
            index_on_page = 0

            for table in page.extract_tables():
                if not is_real_table(table):
                    continue

                # Pair with the caption at the same position on the same page; fall
                # back to describing the table by its own headers when unlabelled.
                if index_on_page < len(page_captions):
                    caption = page_captions[index_on_page]
                else:
                    caption = describe_from_headers(table, page_num)

                results.append(
                    {
                        "page": page_num,
                        "markdown": table_to_markdown(table),
                        "caption": caption,
                    }
                )
                index_on_page += 1

    return results


if __name__ == "__main__":
    tables = extract_tables(PDF_PATH)
    print(f"Tables found: {len(tables)}")
    for t in tables:
        print(f"--- page {t['page']} ---")
        print(f"CAPTION: {t['caption']}")
        print(t["markdown"][:200])
        print()
