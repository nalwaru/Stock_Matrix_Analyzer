from types import SimpleNamespace as NS

import openai

from agent.deps import Deps
from agent.nodes import route_after_router, router_node
from agent.schemas import RouteDecision


class FakeClient:
    def __init__(self, decision=None, error=None):
        self.calls = []
        self.decision, self.error = decision, error
        self.chat = NS(completions=NS(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return NS(choices=[NS(message=NS(parsed=self.decision, refusal=None))])


def deps(client, indexed=("AAPL",)):
    return Deps(client=client, model="m", retrieve=lambda q, t: [], fetch_quotes=lambda t: {},
                indexed_tickers=list(indexed))


def state(question="q", held=("AAPL", "NVDA")):
    return {"question": question, "held_tickers": list(held)}


def decision(route, tickers=(), reason="r"):
    return RouteDecision(route=route, tickers=list(tickers), reason=reason)


def test_router_returns_structured_decision():
    client = FakeClient(decision("sec_filings", ["AAPL"]))
    out = router_node(state("What are Apple's risks?"), deps(client))
    assert out["route"].route == "sec_filings" and out["route"].tickers == ["AAPL"]
    assert "Router" in out["trace"][0]
    assert "What are Apple's risks?" in client.calls[0]["messages"][1]["content"]


def test_router_failure_falls_back_to_portfolio_only_with_trace():
    out = router_node(state(), deps(FakeClient(error=openai.OpenAIError("boom"))))
    assert out["route"].route == "portfolio_only"
    assert "routing failed" in out["trace"][0].lower()


def test_tickers_filtered_to_indexed_for_filings_routes():
    out = router_node(state(), deps(FakeClient(decision("sec_filings", ["aapl", "ZZZZ"]))))
    assert out["route"].tickers == ["AAPL"]


def test_empty_tickers_fall_back_to_held_indexed_tickers():
    out = router_node(state(), deps(FakeClient(decision("sec_filings", []))))
    assert out["route"].tickers == ["AAPL"]


def test_filings_route_without_indexed_ticker_downgrades():
    d = deps(FakeClient(decision("sec_filings", ["NVDA"])))
    out = router_node(state(), d)
    assert out["route"].route == "portfolio_only" and "no filings" in out["trace"][0].lower()
    both = router_node(state(), deps(FakeClient(decision("both", ["NVDA"]))))
    assert both["route"].route == "live_price" and both["route"].tickers == ["NVDA"]


def test_live_price_with_no_tickers_uses_held():
    out = router_node(state(), deps(FakeClient(decision("live_price", []))))
    assert out["route"].tickers == ["AAPL", "NVDA"]


def test_route_after_router_mapping():
    expected = {"sec_filings": "retrieve", "both": "retrieve", "live_price": "price",
                "portfolio_only": "tools", "analysis": "analysis", "out_of_scope": "refuse"}
    for route, node in expected.items():
        assert route_after_router({"route": decision(route)}) == node
