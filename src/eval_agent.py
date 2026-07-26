"""Evaluate the agentic pipeline with RAGAS.

Agent runs are expensive (multi-step tool loops) and the Groq free tier has a daily
token cap, so generated answers are cached to disk and re-used. Re-running only
regenerates questions that are missing from the cache.
"""

import json
import os
import re
import sys

import ragas_compat  # noqa: F401  (must precede any ragas import; registers VertexAI stub)

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness

from agent import graph
from eval_baseline import ragas_embeddings, ragas_llm

sys.stdout.reconfigure(encoding="utf-8")

EVAL_SET_PATH = "data/eval_set.json"
CACHE_PATH = "data/eval_agent_answers.json"
OUTPUT_PATH = "data/eval_agent_results.csv"


def split_tool_output(text: str) -> list[str]:
    """Split the retriever tool's joined output back into individual source chunks."""
    parts = re.split(r"\n\n(?=\[(?:chunk|table)_\d+\])", text.strip())
    return [p for p in parts if p]


def run_agent(question: str, thread_id: str) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [{"role": "user", "content": question}]}, config)

    contexts = []
    for msg in result["messages"]:
        if msg.type == "tool" and msg.name == "retrieve_documents":
            contexts.extend(split_tool_output(msg.content))

    return result["messages"][-1].content, contexts


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def generate_answers(eval_set: list[dict]) -> dict:
    """Run the agent for any question not already cached. Saves after each one so a
    rate-limit failure part-way through doesn't lose completed work."""
    cache = load_cache()

    for i, item in enumerate(eval_set):
        question = item["question"]
        if question in cache:
            print(f"[{i + 1}/{len(eval_set)}] cached | {question[:55]}...")
            continue

        answer, contexts = run_agent(question, f"eval-{i}")
        cache[question] = {"answer": answer, "contexts": contexts}
        save_cache(cache)
        print(f"[{i + 1}/{len(eval_set)}] {len(contexts)} contexts | {question[:55]}...")

    return cache


if __name__ == "__main__":
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    cache = generate_answers(eval_set)

    missing = [item["question"] for item in eval_set if item["question"] not in cache]
    if missing:
        print(f"\n{len(missing)} question(s) still ungenerated — re-run when quota allows:")
        for q in missing:
            print(f"  - {q}")
        sys.exit(1)

    samples = []
    no_retrieval = []
    for item in eval_set:
        entry = cache[item["question"]]
        if not entry["contexts"]:
            no_retrieval.append(item["question"])
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                retrieved_contexts=entry["contexts"] or ["(no documents retrieved)"],
                response=entry["answer"],
                reference=item["ground_truth"],
            )
        )

    results = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), AnswerRelevancy(strictness=1)],  # Groq rejects n>1
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    df = results.to_pandas()
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Agentic pipeline scores ===")
    print(f"Mean faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"Mean answer relevancy:  {df['answer_relevancy'].mean():.3f}")
    print(f"Questions with no retrieval call: {len(no_retrieval)}")
    for q in no_retrieval:
        print(f"  - {q}")
