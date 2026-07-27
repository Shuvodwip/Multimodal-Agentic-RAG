"""Request and response schemas for the API.

Kept separate from the route handlers so the contract is readable in one place, and
so FastAPI can generate accurate OpenAPI docs from the field metadata.
"""

import re

from pydantic import BaseModel, Field, field_validator

SOURCE_ID_PATTERN = re.compile(r"^\[((?:chunk|table)_\d+)\]\s*(.*)", re.DOTALL)
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class Source(BaseModel):
    """A passage the retriever returned, so an answer can be traced to its evidence."""

    id: str = Field(description="Stored document id, e.g. chunk_40 or table_0")
    type: str = Field(description="Whether the passage came from prose or a table")
    preview: str = Field(description="Opening text of the passage")

    @classmethod
    def from_passage(cls, passage: str, preview_chars: int = 240) -> "Source":
        """Parse one '[chunk_40]\\n<text>' block emitted by the retriever tool."""
        match = SOURCE_ID_PATTERN.match(passage.strip())
        if match:
            doc_id, body = match.group(1), match.group(2)
        else:
            doc_id, body = "unknown", passage

        body = " ".join(body.split())
        preview = body[:preview_chars] + ("…" if len(body) > preview_chars else "")
        return cls(
            id=doc_id,
            type="table" if doc_id.startswith("table_") else "text",
            preview=preview,
        )


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
        description="Question to answer from the indexed paper.",
        examples=["What accuracy does the Weather-Only ablation experiment achieve?"],
    )
    session_id: str | None = Field(
        default=None,
        pattern=SESSION_ID_PATTERN,
        description=(
            "Conversation thread. Reuse it across requests for follow-up questions that "
            "refer back to earlier answers. Omit it to start an isolated conversation."
        ),
        examples=["demo-session"],
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        """min_length alone would accept a string of spaces."""
        cleaned = value.strip()
        if len(cleaned) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        return cleaned


class Figure(BaseModel):
    """A figure matched to the question by image search.

    Only its identity and page are returned — the image itself is not interpreted, and
    the API does not serve image bytes.
    """

    id: str = Field(description="Stored figure id, e.g. image_9")
    page: int = Field(description="Page of the source document the figure appears on")


class AskResponse(BaseModel):
    answer: str = Field(description="Answer grounded in the retrieved sources.")
    session_id: str = Field(description="Thread this exchange belongs to; reuse for follow-ups.")
    sources: list[Source] = Field(
        default_factory=list,
        description="Passages retrieved while answering. Empty if no retrieval was needed.",
    )
    figures: list[Figure] = Field(
        default_factory=list,
        description="Figures matching the question, when the agent searched for one.",
    )
    latency_seconds: float = Field(description="Server-side time to produce the answer.")
    trace_url: str | None = Field(
        default=None,
        description="Langfuse trace for this request, when tracing is configured.",
    )


class HealthResponse(BaseModel):
    status: str
    model: str = Field(description="Model serving agent requests.")
    tracing: bool = Field(description="Whether Langfuse tracing is active.")


class ErrorResponse(BaseModel):
    """Uniform error body, so clients parse one shape regardless of failure type."""

    error: str = Field(description="Stable machine-readable code, e.g. rate_limited.")
    detail: str = Field(description="Human-readable explanation safe to show a user.")
    retry_after_seconds: float | None = Field(
        default=None,
        description="Suggested wait before retrying, when the failure is transient.",
    )
