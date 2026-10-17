from datetime import datetime, timezone

import pandas as pd

from agent.nodes import price_node
from agent.prices import quotes_from_market
from agent.schemas import RouteDecision
from agent.state import Quote
from tests.agent_fakes import make_deps

NOW = datetime(2026, 10, 17, 16, 45, tzinfo=timezone.utc)


def frame(prices):
    return pd.DataFrame({"Price": prices}, index=pd.Index(list(prices), name="Ticker"))


def test_quotes_carry_fetch_time():
    quotes = quotes_from_market(["AAPL"], now=NOW, fetch=lambda t: frame({"AAPL": 250.5}))
    assert quotes["AAPL"] == Quote("AAPL", 250.5, "2026-10-17, 16:45 UTC")


def test_missing_or_nan_ticker_has_none_price():
    fetch = lambda t: frame({"AAPL": float("nan")})
    quotes = quotes_from_market(["AAPL", "NVDA"], now=NOW, fetch=fetch)
    assert quotes["AAPL"].price is None and quotes["NVDA"].price is None


def test_fetch_exception_gives_none_prices():
    def boom(t):
        raise RuntimeError("offline")

    quotes = quotes_from_market(["AAPL"], now=NOW, fetch=boom)
    assert quotes == {"AAPL": Quote("AAPL", None, "2026-10-17, 16:45 UTC")}


def test_price_node_uses_route_tickers_then_held():
    seen = []

    def fetch(tickers):
        seen.append(tickers)
        return {t: Quote(t, 1.0, "x") for t in tickers}

    deps = make_deps(quotes=fetch)
    route = lambda t: RouteDecision(route="live_price", tickers=t, reason="r")
    out = price_node({"route": route(["NVDA"]), "held_tickers": ["AAPL"]}, deps)
    assert seen[-1] == ["NVDA"] and out["quotes"]["NVDA"].price == 1.0
    price_node({"route": route([]), "held_tickers": ["AAPL"]}, deps)
    assert seen[-1] == ["AAPL"]


def test_price_node_trace_lists_unavailable_tickers():
    deps = make_deps(quotes=lambda t: {"AAPL": Quote("AAPL", None, "x")})
    out = price_node({"route": RouteDecision(route="live_price", tickers=["AAPL"], reason="r"),
                      "held_tickers": []}, deps)
    assert "unavailable" in out["trace"][0].lower() and "AAPL" in out["trace"][0]
