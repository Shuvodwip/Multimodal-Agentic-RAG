#!/usr/bin/env bash
# End-to-end smoke test against a running API.
#
#   uvicorn api:app --app-dir src --port 8000     # in one terminal
#   bash scripts/smoke_test.sh                    # in another
#
# Unlike tests/test_api.py (which stubs the agent and is free to run), this exercises
# the real pipeline and therefore consumes model quota. Keep the request count small.

set -u
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
SESSION="smoke-$$"
failures=0

pass() { echo "  PASS  $1"; }
fail() { echo "  FAIL  $1"; failures=$((failures + 1)); }

# Extract a field from a JSON body without requiring jq.
field() { python -c "import json,sys; print(json.load(sys.stdin).get('$1',''))"; }

echo "Smoke testing $BASE_URL"
echo

echo "1. GET /health"
body=$(curl -s "$BASE_URL/health")
if [ "$(echo "$body" | field status)" = "ok" ]; then
  pass "health reports ok ($(echo "$body" | field model))"
else
  fail "health did not report ok: $body"
fi

echo
echo "2. POST /ask — grounded answer with sources"
body=$(curl -s -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the mAP of the proposed Swin model?","session_id":"'"$SESSION"'"}')
answer=$(echo "$body" | field answer)
sources=$(echo "$body" | python -c "import json,sys; print(len(json.load(sys.stdin).get('sources',[])))")
if echo "$answer" | grep -q "0.9195"; then
  pass "answer contains expected value 0.9195"
else
  fail "unexpected answer: $answer"
fi
if [ "$sources" -gt 0 ]; then
  pass "response carried $sources source passage(s)"
else
  fail "response carried no sources"
fi

echo
echo "3. POST /ask — follow-up reuses conversation memory"
body=$(curl -s -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"And what accuracy does it reach?","session_id":"'"$SESSION"'"}')
answer=$(echo "$body" | field answer)
if echo "$answer" | grep -qE "85\.29|0\.8529"; then
  pass "follow-up resolved without restating the subject"
else
  fail "follow-up lost context: $answer"
fi

echo
echo "4. POST /ask — invalid input is rejected uniformly"
code=$(curl -s -o /tmp/smoke_invalid.json -w "%{http_code}" -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" -d '{"question":"  "}')
err=$(field error < /tmp/smoke_invalid.json)
if [ "$code" = "422" ] && [ "$err" = "invalid_request" ]; then
  pass "blank question rejected with 422/invalid_request"
else
  fail "expected 422 invalid_request, got $code/$err"
fi

echo
if [ "$failures" -eq 0 ]; then
  echo "All smoke checks passed."
  exit 0
fi
echo "$failures smoke check(s) failed."
exit 1
