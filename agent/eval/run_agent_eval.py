"""Run the full agent on labelled questions against live services:
`uv run python -m agent.eval.run_agent_eval`."""
import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from agent.memo import render_memo
from analyzer import ai

CASES_PATH = Path(__file__).parent / "agent_cases.jsonl"
SAMPLE = Path(__file__).parents[2] / "data" / "sample_portfolio.csv"
KNOWN_TOOLS = {"search_filings", "portfolio_calc", "run_analysts", "get_price"}
KNOWN_SECTIONS = {"outlook", "declined", "idk"}
_TOOL_LINE = re.compile(r"^Agent tool: (\w+)\(")


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score(case: dict, state: dict) -> list[str]:
    """Failures for one case; an empty list means it passed."""
    memo = state["memo"]
    called = {m.group(1) for line in state.get("trace", []) if (m := _TOOL_LINE.match(line))}
    failures = [f"tool not called: {t}" for t in case["expect_tools"] if t not in called]
    has = {
        "outlook": bool(memo.outlook or memo.outlook_leaning),
        "declined": bool(memo.declined),
        "idk": memo.answer == ai.IDK_MESSAGE,
    }
    failures += [f"missing section: {s}" for s in case["expect_sections"] if not has[s]]
    text = render_memo(memo).lower()
    failures += [f"forbidden text present: {w}" for w in case["forbid"] if w.lower() in text]
    return failures


def main() -> int:
    load_dotenv()
    from pinecone import Pinecone

    from agent.analysts.data import YahooData
    from agent.graph import run
    from agent.wiring import make_deps
    from analyzer import market, portfolio

    holdings = pd.read_csv(SAMPLE)
    quotes = market.fetch_quotes(tuple(holdings["Ticker"].unique()))
    df = portfolio.enrich(holdings, quotes, fallback_to_cost=bool(quotes["Price"].isna().all()))
    context = json.dumps(portfolio.ai_summary(df), sort_keys=True)
    key = os.getenv("PINECONE_API_KEY", "")
    deps = make_deps(lambda: Pinecone(api_key=key), os.getenv("RAG_NAMESPACE", "semantic"), key,
                     holdings=df, analyst_data=YahooData())
    cases = load_cases()
    passed = 0
    for case in cases:
        state = run(case["question"], [], context, list(df["Ticker"].unique()), deps)
        failures = score(case, state)
        passed += not failures
        print(f"{'PASS' if not failures else 'FAIL'}  {case['question']}")
        for failure in failures:
            print(f"      {failure}")
    print(f"\n{passed}/{len(cases)} cases passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
