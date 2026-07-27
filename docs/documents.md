# Supported documents

The system indexes one PDF at a time and is not tied to any subject. Upload a report,
manual, contract, policy, or paper — nothing in the retrieval or prompting assumes a
domain.

## What works

| Content | Handling |
|---|---|
| Body text | Extracted with PyMuPDF, split into overlapping chunks, embedded for semantic search and indexed for keyword search. |
| Tables | Detected with pdfplumber and converted to markdown, so row/column structure survives for the model to read. |
| Table captions | Matched from the page's printed caption — `TABLE IV:`, `Table 1.`, `Table A2 —` all recognised. Unlabelled tables get a description synthesised from their column headers instead. |
| Figure captions | Ordinary text in the PDF, so they are indexed and retrievable like any other passage. |

Captions matter more than they appear. A table embedded as a bare grid of digits is
almost unretrievable, because questions resemble a table's *description*, not its
contents. Before captions were added, stored tables ranked in the top-3 for **0 of 15**
evaluation questions.

## What doesn't

**Image contents.** Figures, charts, and photographs are not interpreted. Their captions
are searchable, so "what does Figure 3 show?" answers from the caption — but nothing
reads the pixels. Two reasons: the upload path indexes text and tables only, and no
vision-capable model is available on the free tier this project targets. Adding one
would require a multimodal model and wiring the image collection into the retriever.

**Scanned PDFs.** Pages that are photographs of text have no text layer to extract.
Rather than indexing successfully and then answering nothing, ingestion refuses these
with an explicit message — the file needs OCR first.

**Multiple documents at once.** The store holds one document; indexing a new one
replaces it. Document ids are positional (`chunk_0`, `table_0`), so the previous
document is cleared first to stop stale passages remaining answerable.

## Verifying with your own PDF

```bash
python src/ingest.py                      # index the configured default document
streamlit run src/app.py                  # or upload any PDF from the sidebar
```

Ingestion reports how many chunks and tables it found. Zero tables on a document that
visibly has them usually means the table has no ruling lines for pdfplumber to detect —
a known limit of layout-based detection, not a failure of the pipeline.
