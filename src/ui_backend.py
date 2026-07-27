"""Backends the Streamlit UI can talk to.

Set RAG_API_URL to route questions over HTTP to the FastAPI service:

    RAG_API_URL=http://localhost:8000 streamlit run src/app.py

Unset, the UI loads the agent in-process. Both are useful: in-process is a single
container (the simplest thing to deploy), while HTTP splits frontend from backend so
they scale and deploy independently.
"""

import os
from dataclasses import dataclass

import httpx

TIMEOUT_SECONDS = 180


@dataclass
class Answer:
    text: str
    passages: list[str]
    trace_url: str | None = None


class InProcessBackend:
    """Runs the agent inside the Streamlit process."""

    name = "in-process"
    supports_upload = True

    def __init__(self):
        from agent import AGENT_MODEL, ask_agent
        from ingest import replace_document
        from store_chunks import collection

        self._ask = ask_agent
        self._replace_document = replace_document
        self.collection = collection
        self.model = AGENT_MODEL

    def ask(self, question: str, session_id: str) -> Answer:
        text, passages, trace_url = self._ask(question, thread_id=session_id)
        return Answer(text, passages, trace_url)

    def replace_document(self, pdf_path: str) -> dict:
        return self._replace_document(pdf_path)

    def describe_corpus(self) -> tuple[int, str | None]:
        count = self.collection.count()
        if not count:
            return 0, None
        sample = self.collection.get(limit=1, include=["metadatas"])
        metadatas = sample["metadatas"]
        return count, metadatas[0].get("source") if metadatas else None


class HttpBackend:
    """Calls the FastAPI service. Document management is not exposed over HTTP —
    ingestion is an operational task, not something any UI visitor should trigger."""

    name = "http"
    supports_upload = False

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.model = "unknown"
        self._client = httpx.Client(timeout=TIMEOUT_SECONDS)
        self._load_health()

    def _load_health(self) -> None:
        try:
            health = self._client.get(f"{self.base_url}/health").json()
            self.model = health.get("model", "unknown")
        except httpx.HTTPError:
            # Leave the model unknown rather than refusing to start; the UI surfaces
            # the connection failure on the first question, where it is actionable.
            pass

    def ask(self, question: str, session_id: str) -> Answer:
        response = self._client.post(
            f"{self.base_url}/ask",
            json={"question": question, "session_id": session_id},
        )
        if response.status_code != 200:
            body = response.json()
            raise RuntimeError(body.get("detail", f"API returned {response.status_code}"))

        payload = response.json()
        # Re-render the API's structured sources into the same "[id]\ntext" shape the
        # in-process path yields, so the UI stays backend-agnostic.
        passages = [f"[{s['id']}]\n{s['preview']}" for s in payload.get("sources", [])]
        return Answer(payload["answer"], passages, payload.get("trace_url"))

    def replace_document(self, pdf_path: str) -> dict:
        raise NotImplementedError("Document upload is unavailable when using the HTTP backend.")

    def describe_corpus(self) -> tuple[int, str | None]:
        # The API deliberately exposes no corpus-inspection endpoint.
        return -1, None


def build_backend():
    api_url = os.getenv("RAG_API_URL", "").strip()
    return HttpBackend(api_url) if api_url else InProcessBackend()
