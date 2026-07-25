import sys

from embed_chunks import model as text_model
from embed_images import clip_model
from store_chunks import collection as text_collection
from store_images import collection as image_collection

sys.stdout.reconfigure(encoding="utf-8")


def retrieve_multimodal(query: str, top_k_text: int = 3, top_k_images: int = 2) -> dict:
    text_embedding = text_model.encode([query]).tolist()
    text_results = text_collection.query(query_embeddings=text_embedding, n_results=top_k_text)

    image_embedding = clip_model.encode([query]).tolist()
    image_results = image_collection.query(
        query_embeddings=image_embedding, n_results=top_k_images
    )

    return {"text": text_results, "images": image_results}


if __name__ == "__main__":
    query = "What was the accuracy when weather data was zeroed out in the ablation study?"
    results = retrieve_multimodal(query)

    print(f"Query: {query}\n")
    print("=== Text/Table results ===")
    for i in range(len(results["text"]["ids"][0])):
        meta = results["text"]["metadatas"][0][i]
        dist = results["text"]["distances"][0][i]
        print(f"[{meta['type']}] distance={dist:.4f}")
        print(results["text"]["documents"][0][i][:200])
        print()

    print("=== Image results ===")
    for i in range(len(results["images"]["ids"][0])):
        meta = results["images"]["metadatas"][0][i]
        dist = results["images"]["distances"][0][i]
        print(f"page {meta['page']} distance={dist:.4f}: {meta['path']}")
