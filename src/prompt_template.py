PROMPT_TEMPLATE = """Answer the question using only the context below. Cite the chunk number(s) you relied on in square brackets after each claim, like [Chunk 3]. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, results: dict) -> str:
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    context = "\n\n".join(
        f"[Chunk {meta['chunk_index']}]\n{doc}" for doc, meta in zip(chunks, metadatas)
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


if __name__ == "__main__":
    from search import search

    question = "What is the dual-stream architecture used for disease detection?"
    results = search(question)
    prompt = build_prompt(question, results)

    print(prompt)
