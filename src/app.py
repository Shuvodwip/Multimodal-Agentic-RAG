"""Streamlit chat UI for the agentic RAG pipeline.

    streamlit run src/app.py                                    # agent in-process
    RAG_API_URL=http://localhost:8000 streamlit run src/app.py  # via the FastAPI service

See ui_backend.py for the two backends.
"""

import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    st.subheader("Document")

    indexed, source = backend.describe_corpus()
    if indexed > 0:
        st.caption(f"{indexed} passages indexed from `{Path(source).name}`")
    elif indexed == 0:
        st.warning("Nothing indexed yet — upload a PDF to begin.")
    else:
        st.caption("Corpus served by the API.")

    if backend.supports_upload:
        upload = st.file_uploader("Replace with another PDF", type="pdf")
        if upload and st.button("Index this document", use_container_width=True):
            counts = None
            with st.spinner("Extracting, chunking and embedding…"):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(upload.getbuffer())
                    tmp_path = tmp.name
                try:
                    counts = backend.replace_document(tmp_path)
                except Exception as exc:  # noqa: BLE001 - shown to the user below
                    st.error(str(exc))
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

            if counts:
                st.session_state.messages = []
                st.success(f"Indexed {counts['chunks']} chunks and {counts['tables']} tables.")
                st.rerun()
    else:
        st.caption("Indexing is managed server-side in this mode.")

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
                answer, passages, trace_url = result.text, result.passages, result.trace_url
            except Exception as exc:  # noqa: BLE001 - surfaced to the user below
                answer, passages, trace_url = f"Something went wrong: {exc}", [], None

        sources = [Source.from_passage(p) for p in passages]
        st.markdown(answer)

        if sources:
            with st.expander(f"{len(sources)} sources"):
                for src in sources:
                    st.markdown(f"**`{src.id}`** · {src.type}")
                    st.caption(src.preview)

        meta = f"[View trace]({trace_url})" if trace_url else ""
        if meta:
            st.caption(meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "meta": meta}
    )
