"""Run {fixed, semantic} x {no rerank, rerank} over the gold set: `uv run python -m rag.eval.run_eval`."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

from rag.eval import metrics
from rag.retrieve import retrieve

GOLD_PATH = Path(__file__).parent / "gold.jsonl"
K = 5
# A reranker score below this on an unanswerable question counts as a correct "can't tell".
# Starting value; calibrate against the first real run.
UNANSWERABLE_THRESHOLD = 0.2


def load_gold(path: Path = GOLD_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _ranked_keys(passages, evidence: list[str]) -> list[str]:
    """Strategy-agnostic relevance: a passage is 'e<i>' when it contains evidence phrase i."""
    keys = []
    for p in passages:
        body = _norm(p.text)
        match = next((f"e{i}" for i, phrase in enumerate(evidence) if _norm(phrase) in body), None)
        keys.append(match or p.id)
    return keys


def evaluate(pc, gold: list[dict], namespace: str, rerank: bool, k: int = K) -> dict:
    if not gold:
        raise ValueError("The gold set is empty; add questions to rag/eval/gold.jsonl first.")
    rows, rejected = [], []
    for item in gold:
        passages = retrieve(pc, item["question"], item["tickers"], namespace, rerank=rerank, top_n=k)
        if item.get("answerable", True):
            ranked = _ranked_keys(passages, item["evidence"])
            wanted = {f"e{i}" for i in range(len(item["evidence"]))}
            rows.append((metrics.hit_at_k(ranked, wanted, k), metrics.mrr(ranked, wanted),
                         metrics.recall_at_k(ranked, wanted, k)))
        else:
            rejected.append(not passages or passages[0].score < UNANSWERABLE_THRESHOLD)
    n = len(rows) or 1
    return {
        "hit@k": sum(r[0] for r in rows) / n,
        "mrr": sum(r[1] for r in rows) / n,
        "recall@k": sum(r[2] for r in rows) / n,
        # Reranker scores are only comparable to the threshold when reranking.
        "unanswerable_rejected": (sum(rejected) / len(rejected)) if rejected and rerank else None,
    }


def main() -> int:
    load_dotenv()
    gold = load_gold()
    if not gold:
        print("No gold questions found in rag/eval/gold.jsonl.")
        return 1
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    print(f"{'chunking':<10}{'rerank':<8}{'hit@5':>7}{'mrr':>7}{'recall@5':>10}{'unans.rej':>11}")
    for namespace in ("fixed", "semantic"):
        for rerank in (False, True):
            r = evaluate(pc, gold, namespace, rerank)
            rej = "n/a" if r["unanswerable_rejected"] is None else f"{r['unanswerable_rejected']:.2f}"
            print(f"{namespace:<10}{str(rerank):<8}{r['hit@k']:>7.2f}{r['mrr']:>7.2f}{r['recall@k']:>10.2f}{rej:>11}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
