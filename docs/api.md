# API reference

Start the service:

```bash
uvicorn api:app --app-dir src --reload
```

Interactive docs are generated from the schemas at `http://127.0.0.1:8000/docs`.

---

## `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{ "status": "ok", "model": "llama-3.1-8b-instant", "tracing": true }
```

---

## `POST /ask`

| Field | Type | Notes |
|---|---|---|
| `question` | string, 3–1000 chars | Required. Whitespace-only is rejected. |
| `session_id` | string, `[A-Za-z0-9_-]{1,64}` | Optional. Reuse across requests for follow-ups; omit for an isolated conversation. |

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total number of samples in the combined dataset?"}'
```

```json
{
  "answer": "16,065",
  "session_id": "api-01fbb65a56b0",
  "sources": [
    {
      "id": "table_0",
      "type": "table",
      "preview": "Table I: Distribution of dataset samples across eight tomato disease classes…"
    },
    { "id": "chunk_20", "type": "text", "preview": "…stratified sampling split of 80/20…" }
  ],
  "latency_seconds": 1.948,
  "trace_url": "https://cloud.langfuse.com/project/…/traces/…"
}
```

Every answer carries the passages it was grounded in, so it can be verified without
opening the tracing UI. `trace_url` links to the full execution trace (retrieval scores,
tool calls, token counts) when Langfuse is configured.

### Follow-up questions

Reuse a `session_id` and the agent resolves references against conversation history:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What accuracy does the Weather-Only ablation achieve?", "session_id": "demo"}'
# -> "The Weather-Only ablation achieves 68.00% accuracy."

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does that compare to the full model?", "session_id": "demo"}'
# -> "…68.00% accuracy and 0.7400 mAP, a 17.29% accuracy gap versus the full model's 85.29%…"
```

---

## Errors

Every failure — validation included — returns the same shape:

```json
{ "error": "rate_limited", "detail": "…", "retry_after_seconds": 123.5 }
```

| Status | `error` | Cause |
|---|---|---|
| 422 | `invalid_request` | Request failed schema validation |
| 429 | `rate_limited` | Model provider rate limit; sets a `Retry-After` header |
| 502 | `upstream_unavailable` | Model provider unreachable |
| 502 | `upstream_error` | Model provider returned an error |
| 500 | `internal_error` | Unexpected fault; cause is logged server-side, never returned |

```bash
curl -i -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" -d '{"question": "  "}'
# HTTP/1.1 422
# {"error":"invalid_request","detail":"question: Value error, question must contain at least 3 non-whitespace characters","retry_after_seconds":null}
```

---

## Testing

```bash
python -m pytest tests/test_api.py -q      # stubs the agent — no network, no quota
bash scripts/smoke_test.sh                 # hits a running server — uses real quota
```
