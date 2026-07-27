"""Figure extraction, embedding, and search.

CLIP places images and text in one vector space, so a text query can find a matching
figure. That is *retrieval*, not interpretation: nothing here reads what a chart says.
Answers still come from text and captions — this surfaces the figure itself as visual
evidence the user can look at.

The model is loaded lazily (~600MB) so that importing this module stays cheap for the
text-only paths.
"""

import glob
import os
import re
import shutil
import sys

import chromadb

sys.stdout.reconfigure(encoding="utf-8")

# Kept inside the vector-store directory so figures persist with the index they belong
# to — in the container that path is the writable volume, whereas data/ is mounted
# read-only.
IMAGE_STORE_DIR = os.getenv("IMAGE_STORE_DIR", ".chroma/images")

client = chromadb.PersistentClient(path=".chroma")
collection = client.get_or_create_collection("paper_images")

_clip = None


def get_clip():
    global _clip
    if _clip is None:
        from sentence_transformers import SentenceTransformer

        _clip = SentenceTransformer("clip-ViT-B-32")
    return _clip


def page_from_filename(path: str) -> int:
    match = re.search(r"page(\d+)_img", os.path.basename(path))
    return int(match.group(1)) if match else -1


def ingest_images(pdf_path: str) -> int:
    """Extract every embedded figure from the PDF and index it for text-to-image search."""
    from PIL import Image

    from extract_images import extract_images

    # Clear previous figures: ids are positional, so a document with fewer images would
    # otherwise leave the old document's figures searchable.
    existing = collection.get()["ids"]
    if existing:
        collection.delete(ids=existing)
    shutil.rmtree(IMAGE_STORE_DIR, ignore_errors=True)

    saved = extract_images(pdf_path, IMAGE_STORE_DIR)
    if not saved:
        return 0

    paths = sorted(glob.glob(f"{IMAGE_STORE_DIR}/*"))
    if not paths:
        return 0

    images = [Image.open(p) for p in paths]
    embeddings = get_clip().encode(images)

    collection.upsert(
        ids=[f"image_{i}" for i in range(len(paths))],
        documents=paths,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[
            {"path": p, "page": page_from_filename(p), "type": "image"} for p in paths
        ],
    )
    return len(paths)


def get_figures(ids: list[str]) -> list[dict]:
    """Look up stored figures by id, preserving the order given."""
    if not ids:
        return []

    found = collection.get(ids=ids, include=["metadatas"])
    by_id = {
        doc_id: meta for doc_id, meta in zip(found["ids"], found["metadatas"])
    }
    return [
        {"id": i, "path": by_id[i].get("path"), "page": by_id[i].get("page")}
        for i in ids
        if i in by_id
    ]


def search_images(query: str, top_k: int = 2) -> list[dict]:
    """Find figures matching a text query, closest first.

    Distances come from CLIP's own space and are not comparable with the text
    retriever's scores, so image results are returned as their own ranked list rather
    than merged into one.
    """
    if collection.count() == 0:
        return []

    query_embedding = get_clip().encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding, n_results=min(top_k, collection.count())
    )

    return [
        {
            "id": results["ids"][0][i],
            "path": results["metadatas"][0][i].get("path"),
            "page": results["metadatas"][0][i].get("page"),
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]


if __name__ == "__main__":
    from extract_text import PDF_PATH

    print(f"Indexed figures: {collection.count()}")
    if collection.count() == 0:
        print(f"Ingesting figures from {PDF_PATH}")
        print(f"  indexed: {ingest_images(PDF_PATH)}")

    for query in ["confusion matrix of classification results", "a photo of a diseased leaf"]:
        print(f"\nQuery: {query}")
        for hit in search_images(query, top_k=3):
            print(f"  {hit['id']} page {hit['page']} dist={hit['distance']:.1f} {hit['path']}")
