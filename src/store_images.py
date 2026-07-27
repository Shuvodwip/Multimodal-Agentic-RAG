import os
import re

import chromadb

from embed_images import IMAGES_DIR, embed_images

client = chromadb.PersistentClient(path=".chroma")
collection = client.get_or_create_collection("document_images")


def page_from_filename(path: str) -> int:
    match = re.search(r"page(\d+)_img", os.path.basename(path))
    return int(match.group(1)) if match else -1


if __name__ == "__main__":
    records = embed_images(IMAGES_DIR)

    collection.upsert(
        ids=[f"image_{i}" for i in range(len(records))],
        documents=[r["path"] for r in records],
        embeddings=[r["embedding"].tolist() for r in records],
        metadatas=[
            {"path": r["path"], "page": page_from_filename(r["path"]), "type": "image"}
            for r in records
        ],
    )

    print(f"Images stored: {collection.count()}")
    sample = collection.get(ids=["image_0"], include=["documents", "metadatas"])
    print("--- Sample stored record (image_0) ---")
    print(sample["documents"][0])
    print(sample["metadatas"][0])
