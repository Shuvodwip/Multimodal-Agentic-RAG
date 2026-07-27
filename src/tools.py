import ast
import operator
import os
import sys

from ddgs import DDGS
from langchain_core.tools import tool

from hybrid_search import hybrid_search
from images import search_images
from rerank import RERANK_ENABLED
from store_chunks import collection
from tracing import span
from vision import VISION_ENABLED, VISION_MODEL, describe_image

sys.stdout.reconfigure(encoding="utf-8")

# Enough to carry a title block, author, and opening paragraph without flooding context.
OVERVIEW_CHUNKS = 3


@tool
def retrieve_documents(query: str) -> str:
    """Retrieve relevant passages and tables from the indexed document for a given query."""
    with span("hybrid_search", as_type="retriever", input={"query": query}) as obs:
        results = hybrid_search(query, top_k=3)
        output = "\n\n".join(f"[{r['id']}]\n{r['document']}" for r in results)

        if obs:
            # Record which sources were retrieved and how they ranked, so a bad answer
            # can be traced back to whether retrieval or generation was at fault.
            # Both scores are kept: rrf_score is why a chunk was a candidate,
            # rerank_score is why it survived into the final context.
            obs.update(
                output={
                    "retrieved": [
                        {
                            "id": r["id"],
                            "rrf_score": round(r["rrf_score"], 4),
                            "rerank_score": (
                                round(r["rerank_score"], 3) if "rerank_score" in r else None
                            ),
                        }
                        for r in results
                    ]
                },
                metadata={
                    "top_k": 3,
                    "strategy": "rrf(vector+bm25) -> cross-encoder rerank",
                    "reranked": RERANK_ENABLED,
                },
            )
        return output


_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Disallowed expression: {ast.dump(node)}")


def safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '128.4 - 96.7' or '(12 / 100) * 4500'."""
    with span("calculator", as_type="tool", input={"expression": expression}) as obs:
        try:
            result = str(safe_eval(expression))
            level, message = "DEFAULT", None
        except Exception as e:
            result = f"Error: could not evaluate '{expression}' ({e})"
            level, message = "WARNING", str(e)

        if obs:
            obs.update(output=result, level=level, status_message=message)
        return result


@tool
def document_overview() -> str:
    """Show the beginning of the document and its size.

    Use for questions about the document as a whole — what it is, who it concerns,
    what it covers, or a summary. Similarity search cannot answer these: a title block
    or header contains the subject but does not *discuss* it, so nothing in the query
    matches it. This returns the opening passages directly instead of searching.
    """
    with span("document_overview", as_type="retriever") as obs:
        items = collection.get(include=["documents", "metadatas"])
        if not items["ids"]:
            return "No document is currently indexed."

        # Restore reading order: ids are positional but come back unsorted.
        def position(doc_id: str) -> tuple[int, int]:
            kind, _, number = doc_id.rpartition("_")
            return (0 if kind == "chunk" else 1, int(number) if number.isdigit() else 0)

        ordered = sorted(zip(items["ids"], items["documents"]), key=lambda p: position(p[0]))
        opening = [p for p in ordered if p[0].startswith("chunk_")][:OVERVIEW_CHUNKS]

        source = ""
        for meta in items["metadatas"]:
            if meta.get("source"):
                source = os.path.basename(meta["source"])
                break

        header = (
            f"Document: {source or 'unknown'} — {len(items['ids'])} indexed passages.\n"
            "Opening of the document:\n\n"
        )
        body = "\n\n".join(f"[{doc_id}]\n{text}" for doc_id, text in opening)

        if obs:
            obs.update(
                output={"returned": [doc_id for doc_id, _ in opening], "source": source},
                metadata={"strategy": "positional (document start)"},
            )
        return header + body


def read_best_figure(hit: dict, query: str) -> str:
    """Describe the top-matching figure, degrading to retrieval-only on any failure.

    A vision error must not fail the whole answer: the figure has still been found and
    shown, so the agent can fall back on caption text.
    """
    if not VISION_ENABLED:
        return (
            "Figure contents were not read (vision is disabled). Use retrieve_documents "
            "for the caption text."
        )

    path = hit.get("path")
    if not path or not os.path.exists(path):
        return "The figure file is unavailable, so its contents could not be read."

    with span("describe_figure", as_type="generation", input={"query": query}) as obs:
        try:
            description = describe_image(path, f"Question about this figure: {query}")
            if obs:
                obs.update(output=description, metadata={"model": VISION_MODEL})
            return f"Description of the figure on page {hit['page']}:\n{description}"
        except Exception as exc:  # noqa: BLE001 - degraded, not fatal
            if obs:
                obs.update(level="WARNING", status_message=str(exc)[:200])
            return (
                f"The figure on page {hit['page']} could not be read ({type(exc).__name__}). "
                "Use retrieve_documents for its caption instead."
            )


@tool
def find_figures(query: str) -> str:
    """Look at figures, charts, or images in the document and describe what they show.

    Use for any question about a figure's contents, trends, or appearance — this reads
    the image with a vision model, so it can report what a chart actually depicts. The
    matching figures are also displayed to the user beneath the answer.
    """
    with span("find_figures", as_type="retriever", input={"query": query}) as obs:
        hits = search_images(query, top_k=2)

        if not hits:
            if obs:
                obs.update(output="No figures indexed.", level="WARNING")
            return "This document has no indexed figures."

        pages = ", ".join(f"page {h['page']}" for h in hits)
        # The trailing id list is what the caller parses to render the images; keeping
        # it in the tool output means figures are recovered from *this* turn's messages
        # rather than by re-running the search.
        ids = ",".join(h["id"] for h in hits)

        # Only the best match is read: vision tokens are the scarcest resource here,
        # and the second hit is usually a near-duplicate or a weaker match.
        description = read_best_figure(hits[0], query)

        result = (
            f"Found {len(hits)} matching figure(s), from {pages}; they are displayed to "
            f"the user beneath your answer.\n\n{description}\n\n"
            "This description came from a model that examined the image directly, so "
            "relay its conclusion — do not re-derive it or reinterpret the numbers "
            f"yourself. Mention which page the figure came from. [figures: {ids}]"
        )
        if obs:
            obs.update(
                output={"figures": [{"id": h["id"], "page": h["page"]} for h in hits]},
                metadata={
                    "strategy": "clip text-to-image + vision description",
                    "top_k": 2,
                    "vision_model": VISION_MODEL or "disabled",
                },
            )
        return result


@tool
def web_search(query: str) -> str:
    """Search the web for background or current information not present in the indexed document."""
    with span("web_search", as_type="tool", input={"query": query}) as obs:
        results = DDGS().text(query, max_results=3)
        if not results:
            if obs:
                obs.update(output="No results found.", level="WARNING")
            return "No results found."

        output = "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)
        if obs:
            obs.update(
                output={"sources": [{"title": r["title"], "url": r["href"]} for r in results]}
            )
        return output


if __name__ == "__main__":
    print("Tool name:", retrieve_documents.name)
    print("Tool description:", retrieve_documents.description)
    print("Tool args schema:", retrieve_documents.args)
    print()

    output = retrieve_documents.invoke("What accuracy improvement did the meteorological integration provide?")
    print(output)

    print()
    print("Tool name:", calculator.name)
    print("Tool description:", calculator.description)
    print("Tool args schema:", calculator.args)
    print()

    print(calculator.invoke("85.29 - 61.76"))
    print(calculator.invoke("(7.29 / 100) * 14931"))
    print(calculator.invoke("__import__('os').system('dir')"))

    print()
    print("Tool name:", web_search.name)
    print("Tool description:", web_search.description)
    print("Tool args schema:", web_search.args)
    print()

    print(web_search.invoke("Swin Transformer tiny architecture patch size"))
