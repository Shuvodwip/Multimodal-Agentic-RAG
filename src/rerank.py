"""Cross-encoder reranking of retrieval candidates.

Vector and BM25 search score a query against *precomputed* document embeddings, which
is fast but approximate. A cross-encoder instead reads the query and a document
together in one pass, so it judges relevance directly — far more accurate, but far too
slow to run over a whole corpus. The standard arrangement, used here, is to let hybrid
search recall a wide candidate set and have the cross-encoder pick the best few.

Disable with RERANK=0 to A/B the effect through the evaluation harness.
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

RERANK_ENABLED = os.getenv("RERANK", "1").strip().lower() not in ("0", "false", "no")

# ~80MB, CPU-friendly, trained on MS MARCO passage ranking. Loaded lazily so importing
# this module stays cheap when reranking is turned off.
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Reorder candidates by cross-encoder relevance and return the best `top_k`.

    Each candidate must carry a "document" key. A `rerank_score` is attached so the
    ordering is inspectable in traces rather than being an unexplained reshuffle.
    """
    if not candidates:
        return []
    if not RERANK_ENABLED or len(candidates) == 1:
        return candidates[:top_k]

    pairs = [(query, c["document"]) for c in candidates]
    scores = _get_model().predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


if __name__ == "__main__":
    from hybrid_search import hybrid_search

    query = "What absolute accuracy gain did meteorological integration provide?"
    print(f"Query: {query}\n")
    for i, r in enumerate(hybrid_search(query, top_k=3)):
        score = r.get("rerank_score")
        label = f"rerank={score:.3f}" if score is not None else f"rrf={r['rrf_score']:.4f}"
        print(f"{i + 1}. {r['id']} ({label})")
        print(f"   {r['document'][:110].replace(chr(10), ' ')}")
