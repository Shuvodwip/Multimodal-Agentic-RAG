import sys

from retrieve import retrieve_multimodal

sys.stdout.reconfigure(encoding="utf-8")

queries = [
    "confusion matrix showing classification results per disease class",
    "dual-stream architecture diagram with visual and weather streams",
]

if __name__ == "__main__":
    for query in queries:
        print(f"Query: {query}")
        results = retrieve_multimodal(query, top_k_text=1, top_k_images=3)
        for i in range(len(results["images"]["ids"][0])):
            meta = results["images"]["metadatas"][0][i]
            dist = results["images"]["distances"][0][i]
            print(f"  page {meta['page']} distance={dist:.2f}: {meta['path']}")
        print()
