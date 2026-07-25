import sys

from hybrid_search import hybrid_search
from search import search

sys.stdout.reconfigure(encoding="utf-8")

queries = [
    "how does the model figure out if a plant is sick",  # paraphrased, no exact keywords
    "VPD Magnus-Tetens equation",  # exact technical terms
    "7.29% accuracy gain",  # exact number
]

if __name__ == "__main__":
    for query in queries:
        print(f"Query: {query}")

        print("  Vector-only top 3:")
        for doc_id in search(query, top_k=3)["ids"][0]:
            print(f"    {doc_id}")

        print("  Hybrid (RRF) top 3:")
        for r in hybrid_search(query, top_k=3):
            print(f"    {r['id']} rrf_score={r['rrf_score']:.4f}")

        print()
