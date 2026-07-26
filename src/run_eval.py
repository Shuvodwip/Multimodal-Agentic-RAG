"""Run the full evaluation suite and write a markdown report.

    python src/run_eval.py            # generate (or reuse cached) answers, score, report
    python src/run_eval.py --reuse    # rebuild the report from existing score CSVs only

Answers are cached per pipeline, so a re-run costs judge tokens only. --reuse costs
nothing and is useful for regenerating the report after editing its format.
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

import eval_agent
import eval_baseline
from agent import AGENT_MODEL
from eval_baseline import JUDGE_MODEL, load_eval_set, score_samples

sys.stdout.reconfigure(encoding="utf-8")

REPORT_PATH = "data/eval_report.md"
FAITHFULNESS_FLOOR = 0.60  # below this, the run is considered a regression


def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    return {
        "faithfulness": df["faithfulness"].mean(),
        "relevancy": df["answer_relevancy"].mean(),
        "faith_scored": int(df["faithfulness"].notna().sum()),
        "rel_scored": int(df["answer_relevancy"].notna().sum()),
        "n": n,
    }


def fmt(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.3f}"


def build_report(base: pd.DataFrame, agent: pd.DataFrame, no_retrieval: list[str]) -> str:
    b, a = summarize(base), summarize(agent)
    delta_f = a["faithfulness"] - b["faithfulness"]
    delta_r = a["relevancy"] - b["relevancy"]

    lines = [
        "# RAG Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Configuration",
        "",
        f"- Generator model (both pipelines): `{AGENT_MODEL}`",
        f"- Judge model: `{JUDGE_MODEL}` (temperature 0)",
        "- Metrics: RAGAS faithfulness, answer relevancy (`strictness=1`)",
        f"- Eval set: {b['n']} labeled question/ground-truth pairs",
        "",
        "## Summary",
        "",
        "| Pipeline | Faithfulness | Answer relevancy | Scored |",
        "|---|---|---|---|",
        f"| Baseline RAG | {fmt(b['faithfulness'])} | {fmt(b['relevancy'])} "
        f"| {b['faith_scored']}/{b['n']} |",
        f"| Agentic | {fmt(a['faithfulness'])} | {fmt(a['relevancy'])} "
        f"| {a['faith_scored']}/{a['n']} |",
        f"| **Delta** | **{delta_f:+.3f}** | **{delta_r:+.3f}** | |",
        "",
        "Faithfulness measures whether answer claims are supported by retrieved context — "
        "it is the metric that catches hallucination. Answer relevancy rewards fluent, "
        "on-topic phrasing and will score a confident wrong answer highly, so it must not "
        "be read as a correctness measure.",
        "",
    ]

    if no_retrieval:
        lines += [
            f"### Grounding warning: {len(no_retrieval)} answer(s) with no retrieval call",
            "",
            *[f"- {q}" for q in no_retrieval],
            "",
        ]

    weak = agent[agent["faithfulness"] < 1.0].sort_values("faithfulness")
    if not weak.empty:
        lines += [
            f"### Agent answers with unsupported claims ({len(weak)})",
            "",
            "| Faithfulness | Question | Answer |",
            "|---|---|---|",
        ]
        for _, row in weak.iterrows():
            q = str(row["user_input"])[:60].replace("|", "/")
            ans = str(row["response"])[:80].replace("\n", " ").replace("|", "/")
            lines.append(f"| {fmt(row['faithfulness'])} | {q} | {ans} |")
        lines.append("")

    lines += [
        "## Per-question scores",
        "",
        "| # | Question | Base faith | Agent faith | Base rel | Agent rel |",
        "|---|---|---|---|---|---|",
    ]
    for i in range(len(base)):
        q = str(base.loc[i, "user_input"])[:55].replace("|", "/")
        lines.append(
            f"| {i} | {q} | {fmt(base.loc[i, 'faithfulness'])} "
            f"| {fmt(agent.loc[i, 'faithfulness'])} | {fmt(base.loc[i, 'answer_relevancy'])} "
            f"| {fmt(agent.loc[i, 'answer_relevancy'])} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="rebuild the report from existing score CSVs without calling any model",
    )
    args = parser.parse_args()

    eval_set = load_eval_set()

    if args.reuse:
        for path in (eval_baseline.OUTPUT_PATH, eval_agent.OUTPUT_PATH):
            if not os.path.exists(path):
                print(f"ERROR: {path} not found — run without --reuse first.")
                return 1
        base_df = pd.read_csv(eval_baseline.OUTPUT_PATH)
        agent_df = pd.read_csv(eval_agent.OUTPUT_PATH)
        _, no_retrieval = eval_agent.build_samples(eval_set)
    else:
        print("--- Baseline pipeline ---")
        base_samples = eval_baseline.build_samples(eval_set)
        base_df = score_samples(base_samples)
        base_df.to_csv(eval_baseline.OUTPUT_PATH, index=False)

        print("\n--- Agentic pipeline ---")
        agent_samples, no_retrieval = eval_agent.build_samples(eval_set)
        agent_df = score_samples(agent_samples)
        agent_df.to_csv(eval_agent.OUTPUT_PATH, index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(base_df, agent_df, no_retrieval))

    agent_faith = agent_df["faithfulness"].mean()
    print(f"\nReport written to {REPORT_PATH}")
    print(f"Agent faithfulness: {agent_faith:.3f} (floor {FAITHFULNESS_FLOOR})")

    if pd.isna(agent_faith) or agent_faith < FAITHFULNESS_FLOOR:
        print("FAIL: agent faithfulness below floor.")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
