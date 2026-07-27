"""Streamlit chat UI for the agentic RAG pipeline.

    streamlit run src/app.py

Calls the agent in-process. Step 46 adds the option to talk to the FastAPI service
over HTTP instead, which is what a split frontend/backend deployment would use.
"""

import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="Agentic RAG", page_icon="📄", layout="centered")


@st.cache_resource(show_spinner="Loading models and vector store…")
def load_backend():
    """Import once per server process — loading embeddings and Chroma takes seconds."""
    from agent import AGENT_MODEL, ask_agent
    from ingest import replace_document
    from schemas import Source
    from store_chunks import collection

    return {
        "ask": ask_agent,
        "model": AGENT_MODEL,
        "replace_document": replace_document,
        "Source": Source,
        "collection": collection,
    }


backend = load_backend()

if "thread_id" not in st.session_state:
    # One conversation thread per browser session, so follow-up questions resolve
    # against this user's history and not someone else's.
    st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:12]}"
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.subheader("Document")

    indexed = backend["collection"].count()
    if indexed:
        sample = backend["collection"].get(limit=1, include=["metadatas"])
        source = sample["metadatas"][0].get("source", "unknown") if sample["metadatas"] else "?"
        st.caption(f"{indexed} passages indexed from `{Path(source).name}`")
    else:
        st.warning("Nothing indexed yet — upload a PDF to begin.")

    upload = st.file_uploader("Replace with another PDF", type="pdf")
    if upload and st.button("Index this document", use_container_width=True):
        with st.spinner("Extracting, chunking and embedding…"):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(upload.getbuffer())
                tmp_path = tmp.name
            try:
                counts = backend["replace_document"](tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        st.session_state.messages = []
        st.success(f"Indexed {counts['chunks']} chunks and {counts['tables']} tables.")
        st.rerun()

    st.divider()
    st.caption(f"Model: `{backend['model']}`")
    st.caption(f"Session: `{st.session_state.thread_id}`")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:12]}"
        st.rerun()


st.title("Ask the paper")
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


if question := st.chat_input("e.g. What accuracy does the Weather-Only ablation achieve?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning…"):
            try:
                answer, passages, trace_url = backend["ask"](
                    question, thread_id=st.session_state.thread_id
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the user below
                answer, passages, trace_url = f"Something went wrong: {exc}", [], None

        sources = [backend["Source"].from_passage(p) for p in passages]
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
