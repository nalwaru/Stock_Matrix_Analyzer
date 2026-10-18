from datetime import datetime, timezone

import pandas as pd

from agent.nodes import price_node
from agent.schemas import RouteDecision
from agent.state import Quote
from agent.tools.history import price_changes
from tests.agent_fakes import make_deps

NOW = datetime(2026, 10, 17, 16, 45, tzinfo=timezone.utc)


def closes(n=30, start=100.0, step=1.0):
    idx = pd.bdate_range(end="2026-10-16", periods=n)
    return pd.Series([start + step * i for i in range(n)], index=idx, name="Close")


def test_changes_for_1d_1w_1m_with_exact_figures():
    facts = price_changes(["AAPL"], now=NOW, fetch=lambda t: closes())
    assert [f.label for f in facts] == [
        "AAPL price history, 2026-10-17, 1d", "AAPL price history, 2026-10-17, 1w",
        "AAPL price history, 2026-10-17, 1m"]
    # closes run 100..129: 1d 128->129, 1w (5 sessions back) 124->129, 1m (21 sessions back) 108->129
    assert "+0.78%" in facts[0].text and "128.00" in facts[0].text and "129.00" in facts[0].text
    assert "+4.03%" in facts[1].text and "124.00" in facts[1].text
    assert "+19.44%" in facts[2].text and "108.00" in facts[2].text


def test_negative_change_has_minus_sign():
    facts = price_changes(["AAPL"], now=NOW, fetch=lambda t: closes(step=-1.0, start=200.0))
    assert "-0.58%" in facts[0].text  # 172 -> 171


def test_short_history_marks_longer_windows_unavailable():
    facts = price_changes(["AAPL"], now=NOW, fetch=lambda t: closes(n=8))
    by_window = {f.label.rsplit(", ", 1)[1]: f.text for f in facts}
    assert "%" in by_window["1d"] and "%" in by_window["1w"]
    assert "unavailable" in by_window["1m"]


def test_fetch_failure_or_empty_gives_unavailable_facts():
    def boom(t):
        raise RuntimeError("offline")

    for fetch in (boom, lambda t: pd.Series(dtype=float)):
        facts = price_changes(["AAPL"], now=NOW, fetch=fetch)
        assert len(facts) == 1 and "unavailable" in facts[0].text and facts[0].label.endswith(", history")


def test_price_node_adds_history_facts_to_quotes():
    class FakeData:
        def closes(self, ticker):
            return closes()

    deps = make_deps(quotes=lambda t: {x: Quote(x, 1.0, "x") for x in t}, analyst_data=FakeData())
    route = RouteDecision(route="live_price", tickers=["AAPL"], reason="r")
    out = price_node({"route": route, "held_tickers": []}, deps)
    assert out["quotes"]["AAPL"].price == 1.0
    assert any(f.label.endswith(", 1w") for f in out["facts"])
