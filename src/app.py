"""Streamlit chat UI for the agentic RAG pipeline.

    streamlit run src/app.py                                    # agent in-process
    RAG_API_URL=http://localhost:8000 streamlit run src/app.py  # via the FastAPI service

See ui_backend.py for the two backends.
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Hosted Streamlit stores configuration in st.secrets, while everything else here reads
# environment variables (so the API, CLI, and container share one mechanism). Copy them
# across before importing anything that constructs a client — the chat model is built at
# import time and raises without its key.
for _key in ("GROQ_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST",
             "VISION_MODEL", "RERANK"):
    try:
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = str(st.secrets[_key])
    except Exception:  # noqa: BLE001 - no secrets file locally, which is normal
        break

from schemas import Source  # noqa: E402  (needs the sys.path entry above)
from ui_backend import build_backend  # noqa: E402

st.set_page_config(page_title="Agentic RAG", page_icon="📄", layout="centered")


@st.cache_resource(show_spinner="Connecting to backend…")
def load_backend():
    """Built once per server process — the in-process backend loads embeddings and
    the vector store, which takes seconds."""
    return build_backend()


backend = load_backend()

if "thread_id" not in st.session_state:
    # One conversation thread per browser session, so follow-up questions resolve
    # against this user's history and not someone else's.
    st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:12]}"
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.subheader("Documents")

    if backend.supports_upload:
        docs = backend.documents()
        if docs:
            for doc in docs:
                st.caption(f"`{doc['name']}` — {doc['passages']} passages")
        else:
            st.warning("Nothing indexed yet — upload a file to begin.")

        uploads = st.file_uploader(
            "Add documents",
            type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help="PDF, Word, plain text, or images. Questions are answered across everything indexed.",
        )
        if uploads and st.button("Index these files", use_container_width=True):
            added, failed = [], []
            for upload in uploads:
                with st.spinner(f"Indexing {upload.name}…"):
                    # Preserve the extension: the loader dispatches on it.
                    ext = Path(upload.name).suffix or ".pdf"
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        tmp.write(upload.getbuffer())
                        tmp_path = tmp.name
                    # Rename so the stored document key derives from the real filename
                    # rather than the temporary one.
                    named = Path(tmp_path).with_name(upload.name)
                    try:
                        Path(tmp_path).replace(named)
                        counts = backend.add_document(str(named))
                        added.append(f"{upload.name}: {counts['chunks']} chunks, "
                                     f"{counts['tables']} tables, {counts['figures']} figures")
                    except Exception as exc:  # noqa: BLE001 - shown to the user below
                        failed.append(f"{upload.name}: {exc}")
                    finally:
                        named.unlink(missing_ok=True)
                        Path(tmp_path).unlink(missing_ok=True)

            for line in added:
                st.success(line)
            for line in failed:
                st.error(line)
            if added:
                st.session_state.messages = []
                st.rerun()

        if docs and st.button("Clear all documents", use_container_width=True):
            backend.clear_all()
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("Documents are managed server-side in this mode.")

    st.divider()
    st.caption(f"Backend: `{backend.name}`")
    st.caption(f"Model: `{backend.model}`")
    st.caption(f"Session: `{st.session_state.thread_id}`")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:12]}"
        st.rerun()


st.title("Ask your document")
st.caption(
    "Answers are grounded in retrieved passages and tables. Every response shows the "
    "sources it used, so you can check it rather than trust it."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for fig in message.get("figures", []):
            if fig.get("path") and Path(fig["path"]).exists():
                st.image(fig["path"], caption=f"Figure from page {fig['page']}")
        if message.get("sources"):
            with st.expander(f"{len(message['sources'])} sources"):
                for src in message["sources"]:
                    st.markdown(f"**`{src.id}`** · {src.type}")
                    st.caption(src.preview)
        if message.get("meta"):
            st.caption(message["meta"])


if question := st.chat_input("Ask something about the indexed document…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning…"):
            try:
                result = backend.ask(question, session_id=st.session_state.thread_id)
                answer, passages = result.text, result.passages
                figures, trace_url = result.figures, result.trace_url
            except Exception as exc:  # noqa: BLE001 - surfaced to the user below
                answer, passages, figures, trace_url = f"Something went wrong: {exc}", [], [], None

        sources = [Source.from_passage(p) for p in passages]
        st.markdown(answer)

        for fig in figures:
            if fig.get("path") and Path(fig["path"]).exists():
                st.image(fig["path"], caption=f"Figure from page {fig['page']}")

        if sources:
            with st.expander(f"{len(sources)} sources"):
                for src in sources:
                    st.markdown(f"**`{src.id}`** · {src.type}")
                    st.caption(src.preview)

        meta = f"[View trace]({trace_url})" if trace_url else ""
        if meta:
            st.caption(meta)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "figures": figures,
            "meta": meta,
        }
    )
