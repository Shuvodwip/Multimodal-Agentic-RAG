"""API contract and failure-path tests.

The agent is stubbed out, so these run without a model provider, without network,
and without consuming any token quota.
"""

import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import APIConnectionError, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import api  # noqa: E402
from agent import AgentResult  # noqa: E402
from api import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)


def stub_answer(question, thread_id="cli", trace=True):
    return AgentResult(
        answer="0.9195",
        passages=["[chunk_38]\nThe proposed model reaches an mAP of 0.9195."],
        figures=[],
        trace_url=None,
    )


def make_rate_limit_error(message: str) -> RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
    return RateLimitError(message, response=response, body=None)


@pytest.fixture(autouse=True)
def stub_agent(monkeypatch):
    monkeypatch.setattr(api, "ask_agent", stub_answer)


def test_health_reports_model():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "model" in body


def test_ask_returns_answer_and_typed_sources():
    body = client.post("/ask", json={"question": "What is the mAP?"}).json()
    assert body["answer"] == "0.9195"
    assert body["sources"][0] == {
        "id": "chunk_38",
        "type": "text",
        "preview": "The proposed model reaches an mAP of 0.9195.",
    }
    assert body["latency_seconds"] >= 0


def test_ask_returns_figures_when_the_agent_found_them(monkeypatch):
    def with_figures(question, thread_id="cli", trace=True):
        return AgentResult(
            answer="The figure is shown below.",
            passages=[],
            figures=[{"id": "image_9", "page": 5, "path": "/tmp/page5_img0.png"}],
            trace_url=None,
        )

    monkeypatch.setattr(api, "ask_agent", with_figures)
    body = client.post("/ask", json={"question": "Show me the confusion matrix"}).json()

    # Identity and page only — the API does not serve image bytes.
    assert body["figures"] == [{"id": "image_9", "page": 5}]


def test_ask_generates_session_id_when_absent():
    body = client.post("/ask", json={"question": "What is the mAP?"}).json()
    assert body["session_id"].startswith("api-")


def test_ask_preserves_supplied_session_id():
    body = client.post("/ask", json={"question": "What is the mAP?", "session_id": "abc-1"}).json()
    assert body["session_id"] == "abc-1"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "hi"},                                   # under min_length
        {"question": "        "},                             # whitespace only
        {"question": "x" * 1001},                             # over max_length
        {"question": "valid question", "session_id": "no spaces!"},  # bad session id
        {},                                                   # missing question
    ],
)
def test_invalid_requests_are_rejected_uniformly(payload):
    response = client.post("/ask", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "invalid_request"
    assert body["detail"]


def test_rate_limit_maps_to_429_with_retry_after(monkeypatch):
    def raise_rate_limit(*args, **kwargs):
        raise make_rate_limit_error("Rate limit reached. Please try again in 2m3.5s.")

    monkeypatch.setattr(api, "ask_agent", raise_rate_limit)
    response = client.post("/ask", json={"question": "What is the mAP?"})

    assert response.status_code == 429
    assert response.json()["error"] == "rate_limited"
    assert response.json()["retry_after_seconds"] == pytest.approx(123.5)
    assert response.headers["Retry-After"] == "123"


def test_connection_error_maps_to_502(monkeypatch):
    def raise_connection_error(*args, **kwargs):
        raise APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))

    monkeypatch.setattr(api, "ask_agent", raise_connection_error)
    response = client.post("/ask", json={"question": "What is the mAP?"})

    assert response.status_code == 502
    assert response.json()["error"] == "upstream_unavailable"


def test_unexpected_error_does_not_leak_internals(monkeypatch):
    def raise_unexpected(*args, **kwargs):
        raise ValueError("chroma collection 'paper_chunks' is corrupt at /secret/path")

    monkeypatch.setattr(api, "ask_agent", raise_unexpected)
    response = client.post("/ask", json={"question": "What is the mAP?"})

    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal_error"
    assert "secret" not in body["detail"]
    assert "chroma" not in body["detail"].lower()


def test_retry_hint_parsing():
    assert api.parse_retry_after("try again in 4m3.6s") == pytest.approx(243.6)
    assert api.parse_retry_after("try again in 8.2s") == pytest.approx(8.2)
    assert api.parse_retry_after("no hint here") is None
