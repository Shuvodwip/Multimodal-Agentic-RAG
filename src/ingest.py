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
from docids import document_key, make_id
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


def base_metadata(path: str) -> dict:
    return {"source": path, "document": os.path.basename(path)}


def ingest_text(path: str = DEFAULT_DOCUMENT) -> int:
    text = load_text(path)
    if not text.strip():
        return 0

    key = document_key(path)
    chunks = splitter.split_text(text)
    embeddings = model.encode(chunks)

    collection.upsert(
        ids=[make_id(key, "chunk", i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {**base_metadata(path), "chunk_index": i, "type": "text"}
            for i in range(len(chunks))
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

    key = document_key(path)
    embeddings = model.encode(documents)
    collection.upsert(
        ids=[make_id(key, "table", i) for i in range(len(tables))],
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=[
            {**base_metadata(path), "type": "table", "page": t["page"], "caption": t["caption"]}
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
        ids=[make_id(document_key(path), "chunk", 0)],
        documents=[text],
        embeddings=model.encode([text]).tolist(),
        metadatas=[{**base_metadata(path), "chunk_index": 0, "type": "text"}],
    )
    return 1


def indexed_documents() -> list[dict]:
    """Documents currently in the store, with how many passages each contributed."""
    items = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in items["metadatas"]:
        name = meta.get("document") or os.path.basename(meta.get("source", "unknown"))
        counts[name] = counts.get(name, 0) + 1
    return [{"name": name, "passages": n} for name, n in sorted(counts.items())]


def remove_document(path_or_name: str) -> int:
    """Drop every passage and figure belonging to one document."""
    from images import remove_document_images

    key = document_key(path_or_name)
    stale = [i for i in collection.get()["ids"] if i.startswith(f"{key}__")]
    if stale:
        collection.delete(ids=stale)

    removed_figures = remove_document_images(key)

    from bm25_search import rebuild_index

    rebuild_index()
    return len(stale) + removed_figures


def clear_all() -> None:
    """Empty the store completely."""
    from images import clear_images

    existing = collection.get()["ids"]
    if existing:
        collection.delete(ids=existing)
    clear_images()

    from bm25_search import rebuild_index

    rebuild_index()


def add_document(path: str) -> dict:
    """Index a document alongside whatever is already stored.

    Ids are namespaced per document, so several can coexist and be searched together.
    Re-adding the same filename replaces that document's passages rather than
    duplicating them, since the key is derived from the name.
    """
    check_supported(path)

    if not is_image(path):
        text = load_text(path)
        if len(text.strip()) < MIN_EXTRACTABLE_CHARS:
            raise NoExtractableTextError(
                "This file has almost no selectable text — a scanned PDF is images of "
                "pages, which text extraction cannot read. It needs OCR first."
            )

    # Clear this document's previous passages so a shorter re-upload cannot leave
    # stale trailing chunks behind; other documents are untouched.
    key = document_key(path)
    stale = [i for i in collection.get()["ids"] if i.startswith(f"{key}__")]
    if stale:
        collection.delete(ids=stale)

    # Figures are searchable by description, so a question about a chart can surface
    # the chart itself as evidence.
    from images import ingest_images

    n_figures = ingest_images(path)

    if is_image(path):
        n_chunks, n_tables = index_image_description(path), 0
    else:
        n_chunks = ingest_text(path)
        n_tables = ingest_tables(path)

    # Keyword search holds an in-memory snapshot; without this the new document is
    # invisible to keyword search until restart.
    from bm25_search import rebuild_index

    rebuild_index()

    return {"chunks": n_chunks, "tables": n_tables, "figures": n_figures}


def replace_document(path: str) -> dict:
    """Index a document as the only one in the store."""
    clear_all()
    return add_document(path)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOCUMENT

    if not target:
        # Exit cleanly rather than failing: the compose ingest service runs this on
        # every start, and an empty store is a valid state — the user uploads their own.
        print("No document to ingest. Pass a path, or set DEFAULT_DOCUMENT to seed one.")
        print(f"Collection currently holds {collection.count()} documents.")
        sys.exit(0)

    print(f"Ingesting {target}")
    try:
        counts = add_document(target)
    except (UnsupportedFileError, NoExtractableTextError) as exc:
        print(f"  {exc}")
        sys.exit(1)
    print(f"  text chunks: {counts['chunks']}")
    print(f"  tables:      {counts['tables']}")
    print(f"  figures:     {counts['figures']}")
    print(f"Collection now holds {collection.count()} documents.")
