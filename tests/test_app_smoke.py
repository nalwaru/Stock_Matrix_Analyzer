from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from analyzer import market

ROOT = Path(__file__).parent.parent


def fake_quotes(tickers):
    q = pd.DataFrame(
        {"Price": 100.0, "Prev_Close": 99.0, "Week52_High": 120.0, "Week52_Low": 80.0},
        index=pd.Index(tickers, name="Ticker"),
    )
    return q


def test_app_renders_sample_dataset(monkeypatch):
    monkeypatch.setattr(market, "fetch_quotes", fake_quotes)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.metric) == 0  # nothing until a dataset is chosen

    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert labels[:3] == ["Total Portfolio Value", "Total Unrealized P/L", "Daily P&L"]
    # 215.75 total shares in the sample x $100 fake price
    assert at.metric[0].value == "$21,575.00"


def test_app_falls_back_to_stale_prices_when_quotes_are_empty(monkeypatch):
    monkeypatch.setattr(market, "fetch_quotes", lambda tickers: market.empty_quotes())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception
    assert any("stale" in e.value.lower() for e in at.error)
    assert at.metric[0].label == "Total Portfolio Value"


def test_ask_ai_tab_exists_and_is_disabled_without_key(monkeypatch):
    monkeypatch.setattr(market, "fetch_quotes", fake_quotes)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)  # ignore the real .env
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Dashboard", "Stress Test", "Ask AI", "Export"]
    assert any("Ask AI is disabled" in i.value for i in at.info)


def _ask_ai_app(monkeypatch, retrieve_impl, indexed=None):
    from analyzer import ai
    from rag import retrieve as rag_retrieve, store

    import streamlit as st

    st.cache_resource.clear()  # app caches the Pinecone client across runs
    monkeypatch.setattr(market, "fetch_quotes", fake_quotes)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("PINECONE_API_KEY", "k")
    calls = {}

    def fake_chat(summary, history, question, model, client=None, passages=None):
        calls["passages"] = passages
        return "Because of the 8-K [AAPL 8-K, 2026-10-01, Item 2]."

    monkeypatch.setattr(ai, "chat_reply", fake_chat)
    monkeypatch.setattr(rag_retrieve, "retrieve", retrieve_impl)
    if indexed is not None:
        monkeypatch.setattr(store, "INDEXED_TICKERS", indexed)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    at.sidebar.toggle[0].set_value(True).run()
    at.chat_input[0].set_value("Why is AAPL down?").run()
    return at, calls


def _passage():
    from rag.retrieve import Passage
    return Passage("id1", "Revenue fell on weak demand.", "AAPL", "8-K", "2026-10-01", "Item 2", "http://x", 0.9)


def test_answer_shows_sources_expander(monkeypatch):
    at, calls = _ask_ai_app(monkeypatch, lambda pc, q, tickers, ns, **kw: [_passage()])
    assert not at.exception
    assert any(e.label == "Sources" for e in at.expander)
    assert calls["passages"] and calls["passages"][0].id == "id1"


def test_retrieval_error_falls_back_to_portfolio_only(monkeypatch):
    from rag.retrieve import RetrievalError

    def boom(*a, **k):
        raise RetrievalError("Filing search is unavailable right now.")

    at, calls = _ask_ai_app(monkeypatch, boom)
    assert not at.exception
    assert calls["passages"] is None
    assert any("unavailable" in c.value for c in at.caption)
    assert any("Because of the 8-K" in m.value for m in at.markdown)


def test_portfolio_without_indexed_filings_skips_retrieval(monkeypatch):
    def must_not_run(*a, **k):
        raise AssertionError("retrieve should not be called")

    at, calls = _ask_ai_app(monkeypatch, must_not_run, indexed=["ZZZZ"])
    assert not at.exception
    assert calls["passages"] is None
    assert any("filings" in c.value.lower() for c in at.caption)


def test_pinecone_client_failure_falls_back_to_portfolio_only(monkeypatch):
    import pinecone

    def broken(*a, **k):
        raise ValueError("bad api key format")

    monkeypatch.setattr(pinecone, "Pinecone", broken)
    at, calls = _ask_ai_app(monkeypatch, lambda *a, **k: [_passage()])
    assert not at.exception
    assert calls["passages"] is None
    assert any("portfolio data only" in c.value for c in at.caption)
