from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    message: str


def greet_node(state: State) -> State:
    return {"message": f"Hello, {state['message']}!"}


builder = StateGraph(State)
builder.add_node("greet", greet_node)
builder.add_edge(START, "greet")
builder.add_edge("greet", END)
graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({"message": "world"})
    print(result)
