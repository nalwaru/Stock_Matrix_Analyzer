from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from analyzer import market

ROOT = Path(__file__).parent.parent
DASHBOARD_CHARTS = 2  # allocation donut and top-holdings bars always on the page


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


def _ask_ai_app(monkeypatch, run_impl):
    import streamlit as st
    from agent import graph

    def stream(question, history, portfolio_json, held, deps):
        state = run_impl(question, history, portfolio_json, held, deps)  # may raise: simulates a crash
        yield "router", {"trace": state["trace"][:1]}
        yield "done", state

    st.cache_resource.clear()
    monkeypatch.setattr(market, "fetch_quotes", fake_quotes)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(graph, "run_stream", stream)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    at.sidebar.toggle[0].set_value(True).run()
    at.chat_input[0].set_value("Why is AAPL down?").run()
    return at


def _passage():
    from rag.retrieve import Passage
    return Passage("id1", "Revenue fell on weak demand.", "AAPL", "8-K", "2026-10-01", "Item 2", "http://x", 0.9)


def _state(answer="Because of the 8-K [AAPL 8-K, 2026-10-01, Item 2].", graded=()):
    from agent.schemas import Memo
    memo = Memo(answer=answer, key_figures=["Net sales: $5"], sources=["AAPL 8-K, 2026-10-01, Item 2"], unverified=[])
    return {"memo": memo, "graded": list(graded), "trace": ["Router: sec_filings (r)", "Grader: kept 1 of 3 passages"]}


def test_answer_renders_memo_sources_and_agent_trace(monkeypatch):
    at = _ask_ai_app(monkeypatch, lambda q, h, pj, held, deps: _state(graded=[_passage()]))
    assert not at.exception
    labels = [e.label for e in at.expander]
    assert "Sources" in labels and "Agent trace" in labels
    assert any("Because of the 8-K" in m.value and "###" not in m.value for m in at.markdown)


def test_graph_receives_question_history_and_held_tickers(monkeypatch):
    seen = {}

    def run(question, history, portfolio_json, held, deps):
        seen.update(question=question, held=held, history=history)
        return _state()

    at = _ask_ai_app(monkeypatch, run)
    assert not at.exception
    assert seen["question"] == "Why is AAPL down?" and "AAPL" in seen["held"] and seen["history"] == []


def test_graph_exception_shows_idk_not_crash(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("unexpected")

    at = _ask_ai_app(monkeypatch, boom)
    assert not at.exception
    from analyzer import ai
    assert any(ai.IDK_MESSAGE in m.value for m in at.markdown)


def test_chat_history_resets_when_filters_change(monkeypatch):
    at = _ask_ai_app(monkeypatch, lambda *a, **k: _state())
    assert any("Because of the 8-K" in m.value for m in at.markdown)
    at.sidebar.multiselect[0].set_value(["Stocks"]).run()
    assert not any("Because of the 8-K" in m.value for m in at.markdown)


def test_charts_from_the_agent_are_rendered_and_dollars_are_escaped(monkeypatch):
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2], name="Close"))
    state = _state(answer="Price is $343.46 and the 50-day SMA is $325.69 [AAPL technical, 2026-10-18, SMA50].")
    state["charts"] = [fig]
    at = _ask_ai_app(monkeypatch, lambda *a, **k: state)
    assert not at.exception
    assert len(at.get("plotly_chart")) == DASHBOARD_CHARTS + 1  # donut + bars + the agent chart
    assert any(r"\$343.46" in m.value for m in at.markdown)


def test_charts_are_hidden_when_the_answer_is_i_dont_know(monkeypatch):
    import plotly.graph_objects as go
    from analyzer import ai

    state = _state(answer=ai.IDK_MESSAGE)
    state["charts"] = [go.Figure(go.Scatter(x=[1, 2], y=[1, 2], name="Close"))]
    at = _ask_ai_app(monkeypatch, lambda *a, **k: state)
    assert not at.exception
    assert any(ai.IDK_MESSAGE in m.value for m in at.markdown)
    assert len(at.get("plotly_chart")) == DASHBOARD_CHARTS  # only the dashboard's own charts
