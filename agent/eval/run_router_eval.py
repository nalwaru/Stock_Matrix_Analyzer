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


def main() -> int:
    load_dotenv()
    from agent.nodes import router_node
    from agent.wiring import make_deps

    deps = make_deps(lambda: None, "semantic", pinecone_key="")

    def decide(question: str) -> str:
        return router_node({"question": question, "held_tickers": HELD}, deps)["route"].route

    result = route_accuracy(load_cases(), decide)
    print(f"router accuracy: {result['accuracy']:.0%}")
    for question, expected, got in result["misses"]:
        print(f"  MISS {question!r}: expected {expected}, got {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
