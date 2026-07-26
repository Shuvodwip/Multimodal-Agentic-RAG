import sys

from dotenv import load_dotenv
from groq import Groq

from prompt_template import build_prompt
from search import search

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

client = Groq()


def ask_with_contexts(question: str, top_k: int = 3) -> tuple[str, list[str]]:
    results = search(question, top_k)
    contexts = results["documents"][0]
    prompt = build_prompt(question, results)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content, contexts


def ask(question: str, top_k: int = 3) -> str:
    answer, _ = ask_with_contexts(question, top_k)
    return answer


if __name__ == "__main__":
    print("Type a question and press Enter (or 'quit' to exit).")
    while True:
        question = input("\n> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        answer = ask(question)
        print(f"\n{answer}")
