import sys

from dotenv import load_dotenv
from groq import Groq

from prompt_template import build_prompt
from search import search

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

client = Groq()


def ask(question: str, top_k: int = 3) -> str:
    results = search(question, top_k)
    prompt = build_prompt(question, results)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


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
