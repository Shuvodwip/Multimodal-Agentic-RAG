import sys

from agent import graph

sys.stdout.reconfigure(encoding="utf-8")

questions = [
    "What is the difference in accuracy between the Scrambled and Weather-Only ablation experiments, in percentage points?",
    "What patch size does this paper's Swin-Tiny Transformer use, and does that match the standard Swin-Tiny configuration described online?",
]

if __name__ == "__main__":
    for i, question in enumerate(questions):
        config = {"configurable": {"thread_id": f"test-{i}"}}
        print(f"Question: {question}\n")

        result = graph.invoke({"messages": [{"role": "user", "content": question}]}, config)

        for msg in result["messages"]:
            if msg.type == "ai" and msg.tool_calls:
                for call in msg.tool_calls:
                    print(f"  -> called {call['name']}({call['args']})")
            elif msg.type == "tool":
                print(f"  <- {msg.content[:150]}")

        print(f"\nAnswer: {result['messages'][-1].content}\n")
        print("=" * 80)
