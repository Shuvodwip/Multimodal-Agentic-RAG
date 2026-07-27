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


def ingest_text() -> int:
    chunks = splitter.split_text(extract_text(PDF_PATH))
    embeddings = model.encode(chunks)

    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": PDF_PATH, "chunk_index": i, "type": "text"} for i in range(len(chunks))
        ],
    )
    return len(chunks)


def ingest_tables() -> int:
    tables = extract_tables(PDF_PATH)

    # Caption first, then the grid: questions resemble a table's description, not its
    # digits, and embedding the grid alone left tables unretrievable (0/15 questions).
    documents = [f"{t['caption']}\n\n{t['markdown']}" for t in tables]
    embeddings = model.encode(documents)

    collection.upsert(
        ids=[f"table_{i}" for i in range(len(tables))],
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": PDF_PATH, "type": "table", "page": t["page"], "caption": t["caption"]}
            for t in tables
        ],
    )
    return len(tables)


if __name__ == "__main__":
    print(f"Ingesting {PDF_PATH}")
    n_chunks = ingest_text()
    print(f"  text chunks: {n_chunks}")
    n_tables = ingest_tables()
    print(f"  tables:      {n_tables}")
    print(f"Collection now holds {collection.count()} documents.")
