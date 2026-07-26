"""Langfuse tracing helpers.

Tracing is enabled only when Langfuse credentials are present, so the pipeline still
runs normally (untraced) for anyone who clones the repo without setting them up.
"""

import contextlib
import os

from dotenv import load_dotenv

load_dotenv()

TRACING_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

if TRACING_ENABLED:
    from langfuse import get_client

    langfuse_client = get_client()
else:  # pragma: no cover - depends on local env
    langfuse_client = None


@contextlib.contextmanager
def span(name: str, as_type: str = "span", **kwargs):
    """Record a Langfuse observation, or do nothing if tracing is disabled.

    Yields the observation so callers can attach output/metadata, or None when
    tracing is off — callers must handle the None case.
    """
    if not TRACING_ENABLED:
        yield None
        return

    with langfuse_client.start_as_current_observation(
        name=name, as_type=as_type, **kwargs
    ) as observation:
        yield observation


def get_callback_handler():
    """LangChain/LangGraph callback handler, or None when tracing is disabled."""
    if not TRACING_ENABLED:
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def flush() -> None:
    if TRACING_ENABLED:
        langfuse_client.flush()


def trace_url() -> str | None:
    if not TRACING_ENABLED:
        return None
    return langfuse_client.get_trace_url()
