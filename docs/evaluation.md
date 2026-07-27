# Evaluation

The RAG pipeline is scored with RAGAS (faithfulness, answer relevancy) against a
15-question labeled set, comparing the agentic pipeline to plain baseline RAG. See
`src/eval_baseline.py`, `src/eval_agent.py`, and `src/run_eval.py`.

## Why this isn't wired into CI

Two hard constraints rule it out, not just inconvenience:

1. **The source paper can't reach CI.** `data/` — the PDF and everything derived from
   it — is deliberately excluded from git, since the paper is unpublished. Without it
   there is no vector store to retrieve from, so an eval running in CI wouldn't fail
   loudly, it would silently score against empty context and produce numbers that look
   real but mean nothing. A meaningless pass is worse than no run at all.
2. **The judge and generator both call metered APIs.** Groq's free tier has rate-limited
   this project's own manual eval runs more than once. Running the full suite on every
   push would burn quota fast and fail from throttling before it ever failed from an
   actual regression.

So evaluation stays a deliberate command you run yourself, not something automated
blindly. That's the same judgment call the CI test suite makes in reverse: `tests/`
stubs the model specifically so it *can* run on every push for free — see
`.github/workflows/ci.yml`.

## Running it before a merge

```bash
python src/run_eval.py            # generate (or reuse cached) answers, score, write report
python src/run_eval.py --reuse     # rebuild the report from existing scores only, no API calls
```

Requires `GROQ_API_KEY` in `.env` and a populated vector store (`python src/ingest.py`
if starting fresh). `LANGFUSE_*` is optional — traces just won't be produced without it.

The script exits non-zero if agent faithfulness falls below the floor set in
`FAITHFULNESS_FLOOR` (`src/run_eval.py`), so it works as a manual gate:

```bash
python src/run_eval.py && echo "safe to merge" || echo "faithfulness regressed — check data/eval_report.md"
```

Review `data/eval_report.md` for the full breakdown — summary deltas, any answers with
unsupported claims, and per-question scores against both pipelines.
