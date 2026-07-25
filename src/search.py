import sys

from embed_chunks import model
from store_chunks import collection

sys.stdout.reconfigure(encoding="utf-8")


def search(query: str, top_k: int = 3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results


if __name__ == "__main__":
    query = "What is the dual-stream architecture used for disease detection?"
    results = search(query)

    for i in range(len(results["ids"][0])):
        print(f"--- Result {i} (distance={results['distances'][0][i]:.4f}) ---")
        print(results["documents"][0][i])
        print(results["metadatas"][0][i])
        print()
