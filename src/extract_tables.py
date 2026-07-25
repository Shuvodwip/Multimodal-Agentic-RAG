import sys

import pdfplumber

from extract_text import PDF_PATH

sys.stdout.reconfigure(encoding="utf-8")


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
    return results


if __name__ == "__main__":
    tables = extract_tables(PDF_PATH)
    print(f"Tables found: {len(tables)}")
    for t in tables:
        print(f"--- Table on page {t['page']} ---")
        print(t["markdown"])
        print()
