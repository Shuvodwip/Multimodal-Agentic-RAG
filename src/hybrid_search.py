import sys

from bm25_search import bm25_search
from rerank import rerank
from search import search

sys.stdout.reconfigure(encoding="utf-8")


def reciprocal_rank_fusion(vector_results: dict, bm25_results: list[dict], k: int = 60) -> dict:
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, doc_id in enumerate(vector_results["ids"][0]):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        docs[doc_id] = {
            "document": vector_results["documents"][0][rank],
            "metadata": vector_results["metadatas"][0][rank],
        }

    for rank, r in enumerate(bm25_results):
        doc_id = r["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        docs[doc_id] = {"document": r["document"], "metadata": r["metadata"]}

    return scores, docs


def hybrid_search(query: str, top_k: int = 5, candidate_k: int = 10) -> list[dict]:
    """Recall a wide candidate set with hybrid search, then rerank for precision.

    RRF fuses by rank position, which makes it robust across scoring scales but also
    means a single dominant match can be outranked by items that merely placed
    moderately in both lists. Keeping RRF for recall and letting a cross-encoder choose
    the final order addresses that without discarding hybrid search's coverage.
    """
    vector_results = search(query, top_k=candidate_k)
    bm25_results = bm25_search(query, top_k=candidate_k)

    scores, docs = reciprocal_rank_fusion(vector_results, bm25_results)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    candidates = [{"id": doc_id, "rrf_score": score, **docs[doc_id]} for doc_id, score in ranked]
    return rerank(query, candidates, top_k=top_k)


if __name__ == "__main__":
    query = "Swin-Tiny Transformer patches"

    print(f"Query: {query}\n")

    print("=== Vector-only top 5 ===")
    for doc_id in search(query, top_k=5)["ids"][0]:
        print(f"  {doc_id}")

    print("\n=== BM25-only top 5 ===")
    for r in bm25_search(query, top_k=5):
        print(f"  {r['id']}")

    print("\n=== Hybrid (RRF) top 5 ===")
    for r in hybrid_search(query, top_k=5):
        print(f"  {r['id']} rrf_score={r['rrf_score']:.4f}")
