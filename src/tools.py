import ast
import operator
import sys

from ddgs import DDGS
from langchain_core.tools import tool

from hybrid_search import hybrid_search
from tracing import span

sys.stdout.reconfigure(encoding="utf-8")


@tool
def retrieve_documents(query: str) -> str:
    """Retrieve relevant passages and tables from the research paper for a given query."""
    with span("hybrid_search", as_type="retriever", input={"query": query}) as obs:
        results = hybrid_search(query, top_k=3)
        output = "\n\n".join(f"[{r['id']}]\n{r['document']}" for r in results)

        if obs:
            # Record which sources were retrieved and how they ranked, so a bad answer
            # can be traced back to whether retrieval or generation was at fault.
            obs.update(
                output={
                    "retrieved": [
                        {"id": r["id"], "rrf_score": round(r["rrf_score"], 4)} for r in results
                    ]
                },
                metadata={"top_k": 3, "strategy": "reciprocal-rank-fusion(vector+bm25)"},
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
    """Evaluate a basic arithmetic expression, e.g. '85.29 - 61.76' or '(7.29 / 100) * 14931'."""
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
def web_search(query: str) -> str:
    """Search the web for current information not found in the research paper."""
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
