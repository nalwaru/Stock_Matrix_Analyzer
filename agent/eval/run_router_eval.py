"""Score the real router against labelled questions: `uv run python -m agent.eval.run_router_eval`."""
import json
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

CASES_PATH = Path(__file__).parent / "router_cases.jsonl"
HELD = ["AAPL", "NVDA", "VOO"]


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def route_accuracy(cases: list[dict], decide: Callable[[str], str]) -> dict:
    misses = []
    for case in cases:
        got = decide(case["question"])
        if got != case["expected_route"]:
            misses.append((case["question"], case["expected_route"], got))
    total = len(cases)
    return {"accuracy": (total - len(misses)) / total if total else 0.0, "misses": misses}


def ticker_misses(cases: list[dict], decide_tickers: Callable[[str], list[str]]) -> list[tuple]:
    """Cases that pin `expected_tickers` where the router resolved different tickers."""
    misses = []
    for case in cases:
        want = case.get("expected_tickers")
        if want is not None and decide_tickers(case["question"]) != want:
            misses.append((case["question"], want, decide_tickers(case["question"])))
    return misses


def main() -> int:
    load_dotenv()
    from agent.nodes import router_node
    from agent.wiring import make_deps

    deps = make_deps(lambda: None, "semantic", pinecone_key="")

    def full(question: str):
        return router_node({"question": question, "held_tickers": HELD}, deps)["route"]

    def decide(question: str) -> str:
        return full(question).route

    result = route_accuracy(load_cases(), decide)
    print(f"router accuracy: {result['accuracy']:.0%}")
    for question, expected, got in result["misses"]:
        print(f"  MISS {question!r}: expected {expected}, got {got}")
    for question, want, got in ticker_misses(load_cases(), lambda q: full(q).tickers):
        print(f"  TICKER MISS {question!r}: expected {want}, got {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
