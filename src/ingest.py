"""Build the vector store from the source PDF.

Run once before serving; safe to re-run because every write is an upsert keyed by a
stable id, so repeats overwrite rather than duplicate.

    python src/ingest.py

In docker-compose this runs as the `ingest` service, writing into the volume the API
then reads. Text chunks and tables only — image embeddings live in a separate
collection that the serving path does not query.
"""

import sys

from chunk_text import splitter
from embed_chunks import model
from extract_tables import extract_tables
from extract_text import PDF_PATH, extract_text
from store_chunks import collection

sys.stdout.reconfigure(encoding="utf-8")


def ingest_text(pdf_path: str = PDF_PATH) -> int:
    chunks = splitter.split_text(extract_text(pdf_path))
    embeddings = model.encode(chunks)

    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": pdf_path, "chunk_index": i, "type": "text"} for i in range(len(chunks))
        ],
    )
    return len(chunks)


def ingest_tables(pdf_path: str = PDF_PATH) -> int:
    tables = extract_tables(pdf_path)

    # Caption first, then the grid: questions resemble a table's description, not its
    # digits, and embedding the grid alone left tables unretrievable (0/15 questions).
    documents = [f"{t['caption']}\n\n{t['markdown']}" for t in tables]
    if not documents:
        return 0

    embeddings = model.encode(documents)
    collection.upsert(
        ids=[f"table_{i}" for i in range(len(tables))],
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": pdf_path, "type": "table", "page": t["page"], "caption": t["caption"]}
            for t in tables
        ],
    )
    return len(tables)


# A PDF with no text layer (a scan or photographed pages) extracts almost nothing.
# Indexing it would "succeed" and then answer nothing, so fail loudly instead.
MIN_EXTRACTABLE_CHARS = 200


class NoExtractableTextError(RuntimeError):
    """Raised when a PDF yields too little text to index."""


def replace_document(pdf_path: str) -> dict:
    """Index a new PDF, discarding whatever was indexed before.

    The store holds one document at a time: ids are positional (chunk_0, table_0), so
    a shorter new document would otherwise leave stale chunks from the longer old one
    behind and answerable. Clearing first keeps retrieval honest.
    """
    text = extract_text(pdf_path)
    if len(text.strip()) < MIN_EXTRACTABLE_CHARS:
        raise NoExtractableTextError(
            "This PDF has almost no selectable text — it is probably a scan or images "
            "of pages. Text extraction cannot read those; the file needs OCR first."
        )

    existing = collection.get()["ids"]
    if existing:
        collection.delete(ids=existing)

    n_chunks = ingest_text(pdf_path)
    n_tables = ingest_tables(pdf_path)

    # Keyword search holds an in-memory snapshot; without this it keeps answering
    # from the previous document.
    from bm25_search import rebuild_index

    rebuild_index()

    return {"chunks": n_chunks, "tables": n_tables}


if __name__ == "__main__":
    print(f"Ingesting {PDF_PATH}")
    n_chunks = ingest_text()
    print(f"  text chunks: {n_chunks}")
    n_tables = ingest_tables()
    print(f"  tables:      {n_tables}")
    print(f"Collection now holds {collection.count()} documents.")
