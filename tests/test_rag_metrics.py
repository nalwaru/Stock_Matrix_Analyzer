import pytest

from rag.eval import metrics, run_eval
from rag.retrieve import Passage


def test_hit_at_k_basic():
    assert metrics.hit_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert metrics.hit_at_k(["a", "b", "c"], {"c"}, 2) == 0.0


def test_mrr_first_rank_and_miss():
    assert metrics.mrr(["a", "b", "c"], {"b", "c"}) == 0.5
    assert metrics.mrr(["a"], {"z"}) == 0.0


def test_recall_at_k_partial():
    assert metrics.recall_at_k(["a", "x", "b"], {"a", "b", "c", "d"}, 3) == 0.5


def test_empty_gold_does_not_divide_by_zero():
    assert metrics.recall_at_k(["a"], set(), 5) == 0.0
    assert metrics.hit_at_k(["a"], set(), 5) == 0.0 and metrics.mrr(["a"], set()) == 0.0


def passage(i, score, text="t"):
    return Passage(f"id{i}", text, "AAPL", "10-K", "2025-11-01", "Item 7", "u", score)


def test_evaluate_scores_answerable_and_unanswerable(monkeypatch):
    results = {
        "q1": [passage(1, 0.9, "Intro"), passage(2, 0.8, "Net  sales rose\n6% to $400 billion.")],
        "q2": [passage(3, 0.05)],
    }
    monkeypatch.setattr(run_eval, "retrieve", lambda pc, q, tickers, ns, rerank, top_n: results[q])
    gold = [
        {"question": "q1", "tickers": ["AAPL"], "evidence": ["NET SALES rose 6%"], "answerable": True},
        {"question": "q2", "tickers": ["AAPL"], "evidence": [], "answerable": False},
    ]
    out = run_eval.evaluate(None, gold, "fixed", rerank=True, k=5)
    assert out["hit@k"] == 1.0 and out["mrr"] == 0.5 and out["recall@k"] == 1.0
    assert out["unanswerable_rejected"] == 1.0
    assert run_eval.evaluate(None, gold, "fixed", rerank=False, k=5)["unanswerable_rejected"] is None


def test_evaluate_missing_evidence_scores_zero(monkeypatch):
    monkeypatch.setattr(run_eval, "retrieve", lambda *a, **k: [passage(1, 0.9, "unrelated")])
    gold = [{"question": "q", "tickers": ["AAPL"], "evidence": ["services revenue"], "answerable": True}]
    out = run_eval.evaluate(None, gold, "semantic", rerank=True, k=5)
    assert out["hit@k"] == 0.0 and out["mrr"] == 0.0 and out["recall@k"] == 0.0


def test_evaluate_without_gold_raises():
    with pytest.raises(ValueError):
        run_eval.evaluate(None, [], "fixed", rerank=True, k=5)
