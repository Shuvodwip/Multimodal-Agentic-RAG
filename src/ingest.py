"""Build the vector store from a source document.

Handles PDFs, Word documents, plain text, and bare images — `loaders.py` decides how
to read each, so nothing here depends on the file format.

    python src/ingest.py [path]

Safe to re-run: every write is an upsert keyed by a stable id, so repeats overwrite
rather than duplicate. In docker-compose this runs as the `ingest` service, writing
into the volume the API then reads.
"""

import os
import sys

from chunk_text import splitter
from embed_chunks import model
from extract_text import DEFAULT_DOCUMENT
from loaders import UnsupportedFileError, check_supported, is_image, load_tables, load_text
from store_chunks import collection

sys.stdout.reconfigure(encoding="utf-8")

# A file with no text layer — a scan, or photographed pages — extracts almost nothing.
# Indexing it would "succeed" and then answer nothing, so fail loudly instead.
MIN_EXTRACTABLE_CHARS = 200

# Roughly 250 tokens — a stored image description is retrieved into every subsequent
# request, so it has to stay small against a 6000 tokens/minute allowance.
MAX_DESCRIPTION_CHARS = 1000


class NoExtractableTextError(RuntimeError):
    """Raised when a document yields too little text to index."""


def ingest_text(path: str = DEFAULT_DOCUMENT) -> int:
    text = load_text(path)
    if not text.strip():
        return 0

    chunks = splitter.split_text(text)
    embeddings = model.encode(chunks)

    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": path, "chunk_index": i, "type": "text"} for i in range(len(chunks))
        ],
    )
    return len(chunks)


def ingest_tables(path: str = DEFAULT_DOCUMENT) -> int:
    tables = load_tables(path)

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
            {"source": path, "type": "table", "page": t["page"], "caption": t["caption"]}
            for t in tables
        ],
    )
    return len(tables)


def index_image_description(path: str) -> int:
    """Describe an uploaded image once and index the description as searchable text.

    A bare image carries no text, so without this the document would be unanswerable
    by the ordinary retrieval path. One vision call at upload makes it searchable for
    every later question, instead of paying for vision on each one.
    """
    from vision import VISION_ENABLED, describe_image

    if not VISION_ENABLED:
        return 0

    try:
        description = describe_image(
            path,
            "Describe this image concisely: its subject, any text, and any data shown. "
            "Keep it under 150 words.",
            max_tokens=1200,
        )
    except Exception:  # noqa: BLE001 - the image is still indexed for visual search
        return 0

    # The free tier allows only 6000 tokens per minute, and this description is fed
    # into every later request as retrieved context. An unbounded one pushes a single
    # request over that limit, which fails outright rather than degrading.
    description = description[:MAX_DESCRIPTION_CHARS]
    text = f"Description of the uploaded image {os.path.basename(path)}:\n{description}"
    collection.upsert(
        ids=["chunk_0"],
        documents=[text],
        embeddings=model.encode([text]).tolist(),
        metadatas=[{"source": path, "chunk_index": 0, "type": "text"}],
    )
    return 1


def replace_document(path: str) -> dict:
    """Index a new document, discarding whatever was indexed before.

    The store holds one document at a time: ids are positional (chunk_0, table_0), so
    a shorter new document would otherwise leave stale chunks from the longer old one
    behind and answerable. Clearing first keeps retrieval honest.
    """
    check_supported(path)

    if not is_image(path):
        text = load_text(path)
        if len(text.strip()) < MIN_EXTRACTABLE_CHARS:
            raise NoExtractableTextError(
                "This file has almost no selectable text — a scanned PDF is images of "
                "pages, which text extraction cannot read. It needs OCR first."
            )

    existing = collection.get()["ids"]
    if existing:
        collection.delete(ids=existing)

    # Figures are searchable by description, so a question about a chart can surface
    # the chart itself as evidence.
    from images import ingest_images

    n_figures = ingest_images(path)

    if is_image(path):
        n_chunks, n_tables = index_image_description(path), 0
    else:
        n_chunks = ingest_text(path)
        n_tables = ingest_tables(path)

    # Keyword search holds an in-memory snapshot; without this it keeps answering
    # from the previous document.
    from bm25_search import rebuild_index

    rebuild_index()

    return {"chunks": n_chunks, "tables": n_tables, "figures": n_figures}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOCUMENT
    print(f"Ingesting {target}")
    try:
        counts = replace_document(target)
    except (UnsupportedFileError, NoExtractableTextError) as exc:
        print(f"  {exc}")
        sys.exit(1)
    print(f"  text chunks: {counts['chunks']}")
    print(f"  tables:      {counts['tables']}")
    print(f"  figures:     {counts['figures']}")
    print(f"Collection now holds {collection.count()} documents.")
