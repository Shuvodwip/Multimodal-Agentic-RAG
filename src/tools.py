import ast
import operator
import sys

from langchain_core.tools import tool

from hybrid_search import hybrid_search

sys.stdout.reconfigure(encoding="utf-8")


@tool
def retrieve_documents(query: str) -> str:
    """Retrieve relevant passages and tables from the research paper for a given query."""
    results = hybrid_search(query, top_k=3)
    return "\n\n".join(f"[{r['id']}]\n{r['document']}" for r in results)


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
    try:
        return str(safe_eval(expression))
    except Exception as e:
        return f"Error: could not evaluate '{expression}' ({e})"


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
