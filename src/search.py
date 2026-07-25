import sys

from embed_chunks import model
from store_chunks import collection

sys.stdout.reconfigure(encoding="utf-8")


def search(query: str, top_k: int = 3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results


def print_results(results):
    for i in range(len(results["ids"][0])):
        print(f"--- Result {i} (distance={results['distances'][0][i]:.4f}) ---")
        print(results["documents"][0][i])
        print(results["metadatas"][0][i])
        print()


if __name__ == "__main__":
    print("Type a question and press Enter (or 'quit' to exit).")
    while True:
        query = input("\n> ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue
        results = search(query)
        print_results(results)
