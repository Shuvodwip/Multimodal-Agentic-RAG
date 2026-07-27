---
title: Agentic Multimodal RAG
emoji: 📄
colorFrom: indigo
colorTo: gray
sdk: streamlit
sdk_version: 1.60.0
app_file: src/app.py
pinned: false
short_description: Ask questions about your documents, with cited sources and figures
---

# Agentic Multimodal RAG

Upload PDFs, Word documents, text files, or images and ask questions about them. Every
answer shows the passages it came from, so you can check it rather than trust it.

## What it does

- **Agentic** — a LangGraph ReAct loop decides which tool each question needs:
  document retrieval, whole-document overview, figure lookup, a calculator, or web search
- **Hybrid retrieval** — BM25 keyword search fused with vector search, then reranked by
  a cross-encoder. Reranking took recall@3 from 11/14 to 14/14 on a labelled question set
- **Multimodal** — tables keep their structure, and figures are found by description and
  read by a vision model
- **Cited** — responses carry the id, type, and preview of every passage used
- **Traced** — each answer links to its full execution trace when Langfuse is configured

## Notes on this demo

- Nothing is indexed until you upload something. Storage here is ephemeral, so uploads
  are cleared when the Space restarts.
- The first question is slow: embedding, reranking, and image models load on demand.
- Figure answers are reliable descriptively but not for reading exact values off dense
  charts — see the repository's evaluation notes.
- Runs on a free-tier language model with a low per-minute token allowance, so a burst
  of large requests can be rate-limited.

Source and evaluation results: https://github.com/Shuvodwip/Multimodal-Agentic-RAG
