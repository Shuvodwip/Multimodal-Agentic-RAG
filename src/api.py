"""FastAPI service exposing the agentic RAG pipeline.

    uvicorn api:app --app-dir src --reload

Models and the vector store load once at import, so the first request is not
penalised with several seconds of setup.
"""

import sys
import time
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from agent import AGENT_MODEL, ask_agent
from tracing import TRACING_ENABLED

sys.stdout.reconfigure(encoding="utf-8")

app = FastAPI(
    title="Agentic Multimodal RAG",
    description="Ask questions about the indexed research paper.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    session_id: str
    latency_seconds: float
    trace_url: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": AGENT_MODEL, "tracing": TRACING_ENABLED}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    # A session_id maps to the agent's conversation thread, so follow-up questions
    # can resolve references. Omitting it starts an isolated one-off conversation.
    session_id = request.session_id or f"api-{uuid.uuid4().hex[:12]}"

    started = time.perf_counter()
    answer, trace_url = ask_agent(request.question, thread_id=session_id)
    elapsed = time.perf_counter() - started

    return AskResponse(
        answer=answer,
        session_id=session_id,
        latency_seconds=round(elapsed, 3),
        trace_url=trace_url,
    )
