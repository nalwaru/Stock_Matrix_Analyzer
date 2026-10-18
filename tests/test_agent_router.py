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
    client = FakeClient(decision("agent", ["AAPL"]))
    out = router_node(state("What are Apple's risks?"), deps(client))
    assert out["route"].route == "agent" and out["route"].tickers == ["AAPL"]
    assert "Router" in out["trace"][0]
    assert "What are Apple's risks?" in client.calls[0]["messages"][1]["content"]


def test_router_failure_falls_back_to_portfolio_only_with_trace():
    out = router_node(state(), deps(FakeClient(error=openai.OpenAIError("boom"))))
    assert out["route"].route == "portfolio_only"
    assert "routing failed" in out["trace"][0].lower()


def test_unindexed_ticker_stays_on_the_agent_route():
    out = router_node(state(), deps(FakeClient(decision("agent", ["orcl"]))))
    assert out["route"].route == "agent" and out["route"].tickers == ["ORCL"]


def test_agent_with_no_tickers_defaults_to_held():
    out = router_node(state(), deps(FakeClient(decision("agent", []))))
    assert out["route"].tickers == ["AAPL", "NVDA"] and out["tickers"] == ["AAPL", "NVDA"]


def test_live_price_with_no_tickers_uses_held():
    out = router_node(state(), deps(FakeClient(decision("live_price", []))))
    assert out["route"].tickers == ["AAPL", "NVDA"]


def test_route_after_router_mapping():
    expected = {"live_price": "price", "portfolio_only": "tools", "agent": "agent",
                "out_of_scope": "refuse"}
    for route, node in expected.items():
        assert route_after_router({"route": decision(route)}) == node


def test_outlook_question_sets_wants_outlook():
    d = RouteDecision(route="agent", tickers=["AAPL"], reason="r", wants_outlook=True)
    out = router_node(state("Do you expect growth next year?"), deps(FakeClient(d)))
    assert out["route"].wants_outlook is True and "outlook" in out["trace"][0].lower()


def test_partial_decline_keeps_in_scope_route_and_declined_part():
    d = RouteDecision(route="agent", tickers=["TSLA"], reason="r", declined_part="buy advice")
    out = router_node(state("Should I buy Tesla, and what is their revenue?"), deps(FakeClient(d)))
    assert out["route"].route == "agent" and out["route"].declined_part == "buy advice"


def test_all_out_of_scope_refuses():
    out = router_node(state("Will AAPL hit 300?"), deps(FakeClient(decision("out_of_scope"))))
    assert out["route"].route == "out_of_scope" and route_after_router(out) == "refuse"


def test_router_uses_its_own_client_and_model_when_configured():
    from dataclasses import replace
    main = FakeClient()
    own = FakeClient(RouteDecision(route="live_price", tickers=["AAPL"], reason="r"))
    out = router_node(state(), replace(deps(main), router_client=own, router_model="x/y"))
    assert out["route"].route == "live_price"
    assert not main.calls and own.calls[0]["model"] == "x/y"
    assert own.calls[0]["extra_body"] == {"provider": {"require_parameters": True}}


def test_router_keeps_the_about_filings_flag():
    d = RouteDecision(route="agent", tickers=["AAPL"], reason="r", about_filings=True)
    out = router_node(state("What are Apple's 10-K risk factors?"), deps(FakeClient(d)))
    assert out["route"].about_filings is True
    assert RouteDecision(route="agent", tickers=[], reason="r").about_filings is False


def test_router_prompt_defines_about_filings():
    from agent.nodes import ROUTER_PROMPT
    assert "about_filings" in ROUTER_PROMPT


class FlakyClient:
    """Fails the first `failures` calls, then returns the decision."""

    def __init__(self, decision, failures):
        self.decision, self.failures, self.calls = decision, failures, 0
        self.chat = NS(completions=NS(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise openai.OpenAIError("transient")
        return NS(choices=[NS(message=NS(parsed=self.decision, refusal=None))])


def test_router_retries_once_after_a_transient_failure():
    client = FlakyClient(decision("agent", ["AAPL"]), failures=1)
    out = router_node(state(), deps(client))
    assert out["route"].route == "agent" and client.calls == 2


def test_router_gives_up_after_two_failures_and_falls_back():
    client = FlakyClient(decision("agent", ["AAPL"]), failures=2)
    out = router_node(state(), deps(client))
    assert out["route"].route == "portfolio_only" and client.calls == 2
    assert "routing failed" in out["trace"][0].lower()


def test_gate_falls_back_to_the_openai_model_when_the_router_model_keeps_failing():
    from dataclasses import replace
    broken = FakeClient(error=openai.OpenAIError("402"))
    main = FakeClient(decision("agent", ["AAPL"]))
    out = router_node(state(), replace(deps(main), router_client=broken, router_model="x/y"))
    assert out["route"].route == "agent" and len(broken.calls) == 2 and len(main.calls) == 1
    assert "fallback" in out["trace"][0].lower()


def test_gate_without_a_router_model_has_no_fallback_note():
    out = router_node(state(), deps(FakeClient(decision("agent", ["AAPL"]))))
    assert "fallback" not in out["trace"][0].lower()
