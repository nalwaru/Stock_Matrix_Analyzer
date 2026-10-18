"""Time each node on sample questions: `uv run python -m agent.eval.profile_latency`."""
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

QUESTIONS = [
    "What are Apple's main risk factors?",
    "What is Apple's current price and why did it move?",
    "What happens to my portfolio if crypto falls 30%?",
    "Is NVDA overbought and how risky is it?",
    "Should I buy Tesla?",
]
SAMPLE = Path(__file__).parents[2] / "data" / "sample_portfolio.csv"


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
    for question in QUESTIONS:
        start = time.perf_counter()
        state = run(question, [], context, list(df["Ticker"].unique()), deps)
        total = time.perf_counter() - start
        route = state["route"].route if state.get("route") else "?"
        print(f"\n{question}\n  route={route}  total={total:.1f}s")
        for node, secs in state.get("timings", []):
            print(f"    {node:<11}{secs:6.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
