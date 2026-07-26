"""FastAPI service exposing the agentic RAG pipeline.

    uvicorn api:app --app-dir src --reload

Models and the vector store load once at import, so the first request is not
penalised with several seconds of setup.
"""

import sys
import time
import uuid

from fastapi import FastAPI

from agent import AGENT_MODEL, ask_agent
from schemas import AskRequest, AskResponse, HealthResponse, Source
from tracing import TRACING_ENABLED

sys.stdout.reconfigure(encoding="utf-8")

app = FastAPI(
    title="Agentic Multimodal RAG",
    description=(
        "Ask questions about an indexed research paper. Answers are grounded in "
        "retrieved passages and tables, and every response carries its sources."
    ),
    version="0.2.0",
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", model=AGENT_MODEL, tracing=TRACING_ENABLED)


@app.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(request: AskRequest) -> AskResponse:
    # A session_id maps to the agent's conversation thread, so follow-up questions
    # can resolve references. Omitting it starts an isolated one-off conversation.
    session_id = request.session_id or f"api-{uuid.uuid4().hex[:12]}"

    started = time.perf_counter()
    answer, passages, trace_url = ask_agent(request.question, thread_id=session_id)
    elapsed = time.perf_counter() - started

    return AskResponse(
        answer=answer,
        session_id=session_id,
        sources=[Source.from_passage(p) for p in passages],
        latency_seconds=round(elapsed, 3),
        trace_url=trace_url,
    )
