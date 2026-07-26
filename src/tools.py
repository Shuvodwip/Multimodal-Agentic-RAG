import sys

from langchain_core.tools import tool

from hybrid_search import hybrid_search

sys.stdout.reconfigure(encoding="utf-8")


@tool
def retrieve_documents(query: str) -> str:
    """Retrieve relevant passages and tables from the research paper for a given query."""
    results = hybrid_search(query, top_k=3)
    return "\n\n".join(f"[{r['id']}]\n{r['document']}" for r in results)


if __name__ == "__main__":
    print("Tool name:", retrieve_documents.name)
    print("Tool description:", retrieve_documents.description)
    print("Tool args schema:", retrieve_documents.args)
    print()

    output = retrieve_documents.invoke("What accuracy improvement did the meteorological integration provide?")
    print(output)
