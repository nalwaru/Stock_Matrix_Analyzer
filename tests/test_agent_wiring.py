import pytest

from agent.wiring import make_deps
from rag import retrieve as rag_retrieve
from rag import store
from tests.agent_fakes import passage


def with_score(p, score):
    from dataclasses import replace
    return replace(p, score=score)


def test_low_relevance_passages_are_dropped(monkeypatch):
    found = [with_score(passage(1), 0.9), with_score(passage(2), 0.01)]
    monkeypatch.setattr(rag_retrieve, "retrieve", lambda pc, q, t, ns, **k: found)
    deps = make_deps(lambda: object(), "semantic", pinecone_key="k")
    assert [p.id for p in deps.retrieve("q", ["AAPL"])] == ["id1"]


def test_missing_pinecone_key_raises_retrieval_error():
    deps = make_deps(lambda: object(), "semantic", pinecone_key="")
    with pytest.raises(rag_retrieve.RetrievalError):
        deps.retrieve("q", ["AAPL"])


def test_pinecone_client_failure_raises_retrieval_error():
    def broken():
        raise ValueError("bad key")

    deps = make_deps(broken, "semantic", pinecone_key="k")
    with pytest.raises(rag_retrieve.RetrievalError):
        deps.retrieve("q", ["AAPL"])


def test_deps_use_indexed_tickers_and_namespace(monkeypatch):
    seen = {}
    monkeypatch.setattr(rag_retrieve, "retrieve", lambda pc, q, t, ns, **k: seen.update(ns=ns) or [])
    deps = make_deps(lambda: object(), "fixed", pinecone_key="k")
    deps.retrieve("q", ["AAPL"])
    assert seen["ns"] == "fixed" and deps.indexed_tickers == store.INDEXED_TICKERS


def test_make_deps_passes_on_step_through():
    seen = []
    deps = make_deps(lambda: object(), "semantic", pinecone_key="k", on_step=seen.append)
    deps.on_step("x")
    assert seen == ["x"]


def test_disclosure_names_openrouter_only_when_configured(monkeypatch):
    from analyzer.ai import disclosure
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("ROUTER_MODEL", "m")
    assert "OpenRouter" in disclosure()
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert "OpenRouter" not in disclosure()
