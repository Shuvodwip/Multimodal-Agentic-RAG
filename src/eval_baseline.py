import json
import sys

import ragas_compat  # noqa: F401  (registers the VertexAI import stub before ragas loads)

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness

from ask import ask_with_contexts
from embed_chunks import model as minilm_model

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")


class MiniLMEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return minilm_model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return minilm_model.encode([text])[0].tolist()


ragas_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.3-70b-versatile", temperature=0))
ragas_embeddings = LangchainEmbeddingsWrapper(MiniLMEmbeddings())

if __name__ == "__main__":
    with open("data/eval_set.json", encoding="utf-8") as f:
        eval_set = json.load(f)

    samples = []
    for item in eval_set:
        answer, contexts = ask_with_contexts(item["question"])
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                retrieved_contexts=contexts,
                response=answer,
                reference=item["ground_truth"],
            )
        )
        print(f"Answered: {item['question'][:60]}...")

    dataset = EvaluationDataset(samples=samples)

    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(strictness=1)],  # Groq rejects n>1
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    df = results.to_pandas()
    df.to_csv("data/eval_baseline_results.csv", index=False)

    print("\n=== Baseline RAG scores ===")
    print(f"Mean faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"Mean answer relevancy:  {df['answer_relevancy'].mean():.3f}")
