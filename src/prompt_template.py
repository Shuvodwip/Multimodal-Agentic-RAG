PROMPT_TEMPLATE = """Answer the question using only the context below. Cite the source id(s) you relied on in square brackets after each claim, like [chunk_3] or [table_1]. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, results: dict) -> str:
    chunks = results["documents"][0]
    ids = results["ids"][0]
    context = "\n\n".join(f"[{doc_id}]\n{doc}" for doc, doc_id in zip(chunks, ids))
    return PROMPT_TEMPLATE.format(context=context, question=question)


if __name__ == "__main__":
    from search import search

    question = "What is the dual-stream architecture used for disease detection?"
    results = search(question)
    prompt = build_prompt(question, results)

    print(prompt)
