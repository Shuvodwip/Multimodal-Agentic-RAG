import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import calculator, retrieve_documents, web_search

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

tools = [retrieve_documents, calculator, web_search]
llm = ChatGroq(model="llama-3.3-70b-versatile").bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "session-1"}}

    print("Type a question and press Enter (or 'quit' to exit). Memory is kept for this session.")
    while True:
        question = input("\n> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        result = graph.invoke({"messages": [{"role": "user", "content": question}]}, config)
        print(f"\n{result['messages'][-1].content}")
