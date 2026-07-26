import sys
import time
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from groq import BadRequestError, RateLimitError
from langchain_core.messages import AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import calculator, retrieve_documents, web_search
from tracing import TRACING_ENABLED, flush, get_callback_handler, span, trace_url

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

# Matches the baseline RAG's model (see ask.py) so agentic-vs-baseline comparisons
# isolate the architecture rather than also changing model size. Also has a 5x larger
# daily token budget on Groq's free tier than llama-3.3-70b (500K vs 100K).
AGENT_MODEL = "llama-3.1-8b-instant"

tools = [retrieve_documents, calculator, web_search]
llm = ChatGroq(model=AGENT_MODEL).bind_tools(tools)

SYSTEM_PROMPT = (
    "You answer questions about a specific research paper. Never state or calculate with "
    "any specific number, statistic, or experiment result unless you have retrieved it in "
    "this conversation using retrieve_documents first. If you don't already have a number "
    "from a tool result, call retrieve_documents to get it before answering or calculating.\n\n"
    "If the retrieved passages do not actually contain the answer, say you don't know and "
    "state what the passages did cover. Never guess, estimate, or fill in a value that does "
    "not appear verbatim in a tool result — an honest 'I don't know' is always preferable to "
    "a plausible-sounding number."
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


MAX_RETRIES = 2
RATE_LIMIT_RETRIES = 3


def agent_node(state: State) -> State:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    bad_request_attempts = 0
    rate_limit_attempts = 0

    while True:
        try:
            return {"messages": [llm.invoke(messages)]}
        except RateLimitError:
            # Per-minute token cap (6K TPM on this model); wait for the window to roll over.
            rate_limit_attempts += 1
            if rate_limit_attempts > RATE_LIMIT_RETRIES:
                raise
            wait = 20 * rate_limit_attempts
            print(f"  [rate limited, waiting {wait}s]", flush=True)
            time.sleep(wait)
        except BadRequestError:
            # Model occasionally emits a malformed tool call the API rejects outright.
            bad_request_attempts += 1
            if bad_request_attempts > MAX_RETRIES:
                fallback = "Sorry, I had trouble forming a response to that. Could you rephrase the question?"
                return {"messages": [AIMessage(content=fallback)]}


builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


def build_config(thread_id: str, trace: bool = True) -> dict:
    """Graph config: thread_id drives conversation memory, callbacks drive tracing."""
    config = {"configurable": {"thread_id": thread_id}}
    if trace:
        handler = get_callback_handler()
        if handler:
            config["callbacks"] = [handler]
    return config


def ask_agent(question: str, thread_id: str = "cli", trace: bool = True) -> tuple[str, str | None]:
    """Answer a question, returning (answer, trace_url). trace_url is None when tracing is off.

    The whole run is wrapped in one root observation so retrieval, tool calls, and LLM
    generations appear as a single tree in Langfuse rather than separate traces.
    """
    payload = {"messages": [{"role": "user", "content": question}]}

    if not (trace and TRACING_ENABLED):
        result = graph.invoke(payload, build_config(thread_id, trace=False))
        return result["messages"][-1].content, None

    with span("agent-run", as_type="agent", input={"question": question}) as obs:
        result = graph.invoke(payload, build_config(thread_id, trace=True))
        answer = result["messages"][-1].content
        if obs:
            obs.update(output=answer)
        # Must be read inside the span context — there is no active trace once it exits.
        url = trace_url()

    return answer, url


if __name__ == "__main__":
    print("Type a question and press Enter (or 'quit' to exit). Memory is kept for this session.")
    if TRACING_ENABLED:
        print("Tracing to Langfuse is on.")

    while True:
        question = input("\n> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        answer, url = ask_agent(question, thread_id="session-1")
        print(f"\n{answer}")
        if url:
            print(f"\ntrace: {url}")

    flush()
