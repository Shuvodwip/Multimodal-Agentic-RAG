"""Evaluate the baseline (non-agentic) RAG pipeline with RAGAS.

Also defines the shared judge LLM, embeddings, and run config used by eval_agent.py,
so both pipelines are scored under identical conditions.
"""

import json
import os
import sys

import ragas_compat  # noqa: F401  (must precede any ragas import; registers VertexAI stub)

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness
from ragas.run_config import RunConfig

from ask import ask_with_contexts
from embed_chunks import model as minilm_model

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

EVAL_SET_PATH = "data/eval_set.json"
CACHE_PATH = "data/eval_baseline_answers.json"
OUTPUT_PATH = "data/eval_baseline_results.csv"


class MiniLMEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return minilm_model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return minilm_model.encode([text])[0].tolist()


# Judge runs on a different model from the systems under test, so evaluation draws
# from its own Groq quota pool. temperature=0 keeps scores reproducible run-to-run.
JUDGE_MODEL = "openai/gpt-oss-120b"

ragas_llm = LangchainLLMWrapper(ChatGroq(model=JUDGE_MODEL, temperature=0))
ragas_embeddings = LangchainEmbeddingsWrapper(MiniLMEmbeddings())

# RAGAS defaults to high concurrency, which instantly exceeds the judge model's
# 8K tokens/minute cap and surfaces as TimeoutError -> NaN scores. Throttle workers
# and allow generous per-job time so every sample actually gets scored.
RUN_CONFIG = RunConfig(timeout=300, max_workers=1, max_retries=5, max_wait=90)

METRICS = [Faithfulness(), AnswerRelevancy(strictness=1)]  # Groq rejects n>1


def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path: str, cache: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def report(df, label: str) -> None:
    n = len(df)
    faith_ok = df["faithfulness"].notna().sum()
    rel_ok = df["answer_relevancy"].notna().sum()

    print(f"\n=== {label} ===")
    print(f"Mean faithfulness:      {df['faithfulness'].mean():.3f}  (scored {faith_ok}/{n})")
    print(f"Mean answer relevancy:  {df['answer_relevancy'].mean():.3f}  (scored {rel_ok}/{n})")
    if faith_ok < n or rel_ok < n:
        print("WARNING: some samples failed to score; means cover only the scored subset.")


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def score_samples(samples: list[SingleTurnSample]):
    """Score samples with the shared judge, metrics, and throttled run config."""
    results = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=METRICS,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RUN_CONFIG,
    )
    return results.to_pandas()


def build_samples(eval_set: list[dict]) -> list[SingleTurnSample]:
    """Generate (or reuse cached) baseline answers and wrap them as RAGAS samples."""
    cache = load_cache(CACHE_PATH)
    for i, item in enumerate(eval_set):
        question = item["question"]
        if question in cache:
            print(f"[{i + 1}/{len(eval_set)}] cached | {question[:55]}...")
            continue
        answer, contexts = ask_with_contexts(question)
        cache[question] = {"answer": answer, "contexts": contexts}
        save_cache(CACHE_PATH, cache)
        print(f"[{i + 1}/{len(eval_set)}] generated | {question[:55]}...")

    return [
        SingleTurnSample(
            user_input=item["question"],
            retrieved_contexts=cache[item["question"]]["contexts"],
            response=cache[item["question"]]["answer"],
            reference=item["ground_truth"],
        )
        for item in eval_set
    ]


if __name__ == "__main__":
    eval_set = load_eval_set()
    samples = build_samples(eval_set)
    df = score_samples(samples)
    df.to_csv(OUTPUT_PATH, index=False)
    report(df, "Baseline RAG scores")
