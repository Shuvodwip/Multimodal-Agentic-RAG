"""Evaluate the agentic pipeline with RAGAS.

Agent runs are expensive (multi-step tool loops) and the Groq free tier has a daily
token cap, so generated answers are cached to disk and re-used. Re-running only
regenerates questions that are missing from the cache.
"""

import json
import os
import sys

import ragas_compat  # noqa: F401  (must precede any ragas import; registers VertexAI stub)

from ragas import SingleTurnSample

from agent import ask_agent
from eval_baseline import load_eval_set, report, score_samples

sys.stdout.reconfigure(encoding="utf-8")

EVAL_SET_PATH = "data/eval_set.json"
CACHE_PATH = "data/eval_agent_answers.json"
OUTPUT_PATH = "data/eval_agent_results.csv"


def run_agent(question: str, thread_id: str) -> tuple[str, list[str]]:
    """Evaluation entry point — delegates to the same traced call path the API uses."""
    answer, contexts, _trace_url = ask_agent(question, thread_id=thread_id)
    return answer, contexts


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


def build_samples(eval_set: list[dict]) -> tuple[list[SingleTurnSample], list[str]]:
    """Generate (or reuse cached) agent answers and wrap them as RAGAS samples.

    Also returns the questions the agent answered without any retrieval call, which
    is a grounding signal the metrics alone don't surface.
    """
    cache = generate_answers(eval_set)

    missing = [item["question"] for item in eval_set if item["question"] not in cache]
    if missing:
        raise RuntimeError(
            f"{len(missing)} question(s) ungenerated (likely rate-limited); "
            "re-run when quota allows — cached answers are preserved."
        )

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

    return samples, no_retrieval


if __name__ == "__main__":
    eval_set = load_eval_set()
    samples, no_retrieval = build_samples(eval_set)

    df = score_samples(samples)
    df.to_csv(OUTPUT_PATH, index=False)

    report(df, "Agentic pipeline scores")
    print(f"Questions with no retrieval call: {len(no_retrieval)}")
    for q in no_retrieval:
        print(f"  - {q}")
