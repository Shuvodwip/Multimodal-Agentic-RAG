import re
import sys

from rank_bm25 import BM25Okapi

from store_chunks import collection as text_collection

sys.stdout.reconfigure(encoding="utf-8")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


all_items = text_collection.get(include=["documents", "metadatas"])
documents = all_items["documents"]
metadatas = all_items["metadatas"]
ids = all_items["ids"]

tokenized_corpus = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)


def bm25_search(query: str, top_k: int = 3) -> list[dict]:
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {"id": ids[i], "document": documents[i], "metadata": metadatas[i], "score": scores[i]}
        for i in ranked
    ]


if __name__ == "__main__":
    query = "Swin-Tiny Transformer patches"
    results = bm25_search(query)

    print(f"Query: {query}\n")
    for r in results:
        print(f"[{r['id']}] score={r['score']:.4f}")
        print(r["document"][:200])
        print()
