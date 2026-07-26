"""Per-request cost and latency reporting, read back from Langfuse traces.

Langfuse records token usage and latency for every generation, but ships no pricing
for Groq models (cost_details comes back empty), so cost is computed here from a
local rate table.

    python src/metrics.py              # summarise recent agent runs
    python src/metrics.py --limit 50   # over more traces
"""

import argparse
import statistics
import sys

from tracing import TRACING_ENABLED, langfuse_client

sys.stdout.reconfigure(encoding="utf-8")

# USD per 1M tokens. These are estimates for reporting only — verify against
# https://groq.com/pricing before quoting them anywhere that matters.
PRICING = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.75},
    "openai/gpt-oss-20b": {"input": 0.10, "output": 0.50},
}

# Groq bills cached prompt reads at a reduced rate; set to 1.0 to price them as
# normal input until the exact discount is confirmed.
CACHE_READ_MULTIPLIER = 1.0


def generation_cost(model: str, usage: dict) -> float | None:
    """USD cost for one generation, or None if the model has no configured rate."""
    rates = PRICING.get(model)
    if not rates or not usage:
        return None

    fresh_input = usage.get("input", 0)
    cached_input = usage.get("input_cache_read", 0)
    output = usage.get("output", 0)

    return (
        fresh_input * rates["input"]
        + cached_input * rates["input"] * CACHE_READ_MULTIPLIER
        + output * rates["output"]
    ) / 1_000_000


def summarize_trace(trace) -> dict:
    generations = [o for o in trace.observations if o.type == "GENERATION"]
    tools = [o for o in trace.observations if o.type in ("TOOL", "RETRIEVER")]

    fresh = cached = output = 0
    cost = 0.0
    priced = True

    for gen in generations:
        usage = gen.usage_details or {}
        fresh += usage.get("input", 0)
        cached += usage.get("input_cache_read", 0)
        output += usage.get("output", 0)

        gen_cost = generation_cost(gen.model, usage)
        if gen_cost is None:
            priced = False
        else:
            cost += gen_cost

    return {
        "id": trace.id,
        "latency": trace.latency or 0.0,
        "llm_calls": len(generations),
        "tool_calls": len(tools),
        "fresh_input": fresh,
        "cached_input": cached,
        "output": output,
        "cost": cost if priced else None,
    }


def main() -> int:
    if not TRACING_ENABLED:
        print("Langfuse credentials not set — nothing to report.")
        return 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--name", default="agent-run", help="trace name to filter on")
    args = parser.parse_args()

    listed = langfuse_client.api.trace.list(name=args.name, limit=args.limit).data
    if not listed:
        print(f"No traces named '{args.name}' found.")
        return 1

    rows = [summarize_trace(langfuse_client.api.trace.get(t.id)) for t in listed]

    print(f"Per-request metrics ({len(rows)} traces named '{args.name}')\n")
    header = f"{'latency':>8} {'LLM':>4} {'tool':>5} {'fresh_in':>9} {'cached':>7} {'out':>6} {'cost_usd':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        cost = f"{r['cost']:.6f}" if r["cost"] is not None else "n/a"
        print(
            f"{r['latency']:>8.2f} {r['llm_calls']:>4} {r['tool_calls']:>5} "
            f"{r['fresh_input']:>9} {r['cached_input']:>7} {r['output']:>6} {cost:>10}"
        )

    latencies = [r["latency"] for r in rows]
    costs = [r["cost"] for r in rows if r["cost"] is not None]
    total_cached = sum(r["cached_input"] for r in rows)
    total_fresh = sum(r["fresh_input"] for r in rows)

    print("\nSummary")
    print(f"  median latency:   {statistics.median(latencies):.2f}s")
    print(f"  p95 latency:      {max(latencies):.2f}s" if len(rows) < 20
          else f"  p95 latency:      {statistics.quantiles(latencies, n=20)[18]:.2f}s")
    print(f"  mean LLM calls:   {statistics.mean(r['llm_calls'] for r in rows):.1f} per request")
    print(f"  input tokens:     {total_fresh:,} fresh + {total_cached:,} cached")
    if total_fresh + total_cached:
        share = total_cached / (total_fresh + total_cached) * 100
        print(f"  prompt cache hit: {share:.1f}% of input tokens served from cache")
    if costs:
        print(f"  mean cost/req:    ${statistics.mean(costs):.6f}")
        print(f"  total cost:       ${sum(costs):.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
