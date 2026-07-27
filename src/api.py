"""FastAPI service exposing the agentic RAG pipeline.

    uvicorn api:app --app-dir src --reload

Models and the vector store load once at import, so the first request is not
penalised with several seconds of setup.
"""

import logging
import re
import sys
import time
import uuid

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from groq import APIConnectionError, APIStatusError, RateLimitError

from agent import AGENT_MODEL, ask_agent
from schemas import AskRequest, AskResponse, ErrorResponse, Figure, HealthResponse, Source
from tracing import TRACING_ENABLED

sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger("rag.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# Groq reports the wait in its message ("Please try again in 4m3.6s") rather than a
# structured field, so recover it for the Retry-After header when present.
RETRY_HINT = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")

app = FastAPI(
    title="Agentic Multimodal RAG",
    description=(
        "Ask questions about an indexed research paper. Answers are grounded in "
        "retrieved passages and tables, and every response carries its sources."
    ),
    version="0.3.0",
    responses={
        422: {"model": ErrorResponse, "description": "Request failed validation"},
        429: {"model": ErrorResponse, "description": "Upstream model rate limit reached"},
        502: {"model": ErrorResponse, "description": "Upstream model unavailable"},
        500: {"model": ErrorResponse, "description": "Unexpected server error"},
    },
)


def error_response(
    code: int, error: str, detail: str, retry_after: float | None = None
) -> JSONResponse:
    body = ErrorResponse(error=error, detail=detail, retry_after_seconds=retry_after)
    headers = {"Retry-After": str(int(retry_after))} if retry_after else None
    return JSONResponse(status_code=code, content=body.model_dump(), headers=headers)


def parse_retry_after(message: str) -> float | None:
    match = RETRY_HINT.search(message)
    if not match:
        return None
    minutes = float(match.group(1) or 0)
    return minutes * 60 + float(match.group(2))


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return the same error shape as every other failure, not FastAPI's default."""
    first = exc.errors()[0]
    field = ".".join(str(p) for p in first.get("loc", []) if p != "body") or "request"
    return error_response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "invalid_request",
        f"{field}: {first.get('msg', 'is invalid')}",
    )


@app.exception_handler(RateLimitError)
async def handle_rate_limit(request: Request, exc: RateLimitError) -> JSONResponse:
    retry_after = parse_retry_after(str(exc))
    logger.warning("upstream rate limit; retry_after=%s", retry_after)
    return error_response(
        status.HTTP_429_TOO_MANY_REQUESTS,
        "rate_limited",
        "The language model provider's rate limit was reached. Please retry shortly.",
        retry_after,
    )


@app.exception_handler(APIConnectionError)
async def handle_connection_error(request: Request, exc: APIConnectionError) -> JSONResponse:
    logger.error("cannot reach model provider: %s", exc)
    return error_response(
        status.HTTP_502_BAD_GATEWAY,
        "upstream_unavailable",
        "Could not reach the language model provider. Please retry shortly.",
    )


@app.exception_handler(APIStatusError)
async def handle_api_status_error(request: Request, exc: APIStatusError) -> JSONResponse:
    logger.error("model provider returned %s: %s", exc.status_code, exc)
    return error_response(
        status.HTTP_502_BAD_GATEWAY,
        "upstream_error",
        "The language model provider returned an error. Please retry shortly.",
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # Log the real cause server-side; never return internals to the caller.
    logger.exception("unhandled error answering request")
    return error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "Something went wrong answering that question.",
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
    result = ask_agent(request.question, thread_id=session_id)
    elapsed = time.perf_counter() - started

    logger.info(
        "answered session=%s in %.2fs with %d sources, %d figures",
        session_id,
        elapsed,
        len(result.passages),
        len(result.figures),
    )

    return AskResponse(
        answer=result.answer,
        session_id=session_id,
        sources=[Source.from_passage(p) for p in result.passages],
        figures=[Figure(id=f["id"], page=f["page"]) for f in result.figures],
        latency_seconds=round(elapsed, 3),
        trace_url=result.trace_url,
    )
