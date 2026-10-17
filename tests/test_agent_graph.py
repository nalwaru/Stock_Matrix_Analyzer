from agent.graph import build_graph, run
from agent.schemas import GradeResult, Memo, PassageGrade, RewriteResult, RouteDecision
from agent.state import Quote
from analyzer import ai
from tests.agent_fakes import ScriptedClient, make_deps, passage

LABEL = "AAPL 10-K, 2025-10-31, Item 1A"


def decision(route, tickers=("AAPL",)):
    return RouteDecision(route=route, tickers=list(tickers), reason="r")


def grade(relevant=True, pid="id1"):
    return GradeResult(grades=[PassageGrade(id=pid, relevant=relevant, reason="r")])


def good_memo():
    return Memo(answer=f"Risk exists [{LABEL}].", key_figures=[], sources=[LABEL], unverified=[])


def go(client, deps, question="q"):
    return run(question, [], "{}", ["AAPL"], deps)


def counting_retrieve(passages):
    calls = []

    def retrieve(query, tickers):
        calls.append(query)
        return passages

    retrieve.calls = calls
    return retrieve


def test_sec_route_relevant_passage_reaches_synthesizer():
    client = ScriptedClient(RouteDecision=decision("sec_filings"), GradeResult=grade(), Memo=good_memo())
    retrieve = counting_retrieve([passage(1)])
    state = go(client, make_deps(client, retrieve=retrieve))
    assert state["memo"].answer.startswith("Risk exists") and len(retrieve.calls) == 1
    assert any(line.startswith("Router") for line in state["trace"])


def test_sec_route_no_relevance_rewrites_twice_then_idk():
    client = ScriptedClient(
        RouteDecision=decision("sec_filings"),
        GradeResult=[grade(False), grade(False), grade(False)],
        RewriteResult=[RewriteResult(query="r1"), RewriteResult(query="r2")],
    )
    retrieve = counting_retrieve([passage(1)])
    state = go(client, make_deps(client, retrieve=retrieve))
    assert state["memo"].answer == ai.IDK_MESSAGE
    assert retrieve.calls == ["q", "r1", "r2"] and state["rewrites"] == 2
    assert "Memo" not in client.calls


def test_live_price_route_skips_retrieval():
    quote = Quote("AAPL", 250.0, "2026-10-17, 16:45 UTC")
    text = "Price is 250 [AAPL live quote, 2026-10-17, 16:45 UTC]."
    client = ScriptedClient(RouteDecision=decision("live_price"),
                            Memo=Memo(answer=text, key_figures=[], sources=[], unverified=[]))
    retrieve = counting_retrieve([passage(1)])
    state = go(client, make_deps(client, retrieve=retrieve, quotes=lambda t: {"AAPL": quote}))
    assert state["memo"].answer == text and retrieve.calls == []


def test_both_route_runs_retrieval_then_price():
    quote = Quote("AAPL", 250.0, "2026-10-17, 16:45 UTC")
    memo = Memo(answer=f"[{LABEL}] and [AAPL live quote, 2026-10-17, 16:45 UTC]",
                key_figures=[], sources=[], unverified=[])
    client = ScriptedClient(RouteDecision=decision("both"), GradeResult=grade(), Memo=memo)
    state = go(client, make_deps(client, retrieve=counting_retrieve([passage(1)]), quotes=lambda t: {"AAPL": quote}))
    assert state["memo"].answer == memo.answer
    order = [line.split(":")[0] for line in state["trace"]]
    assert order.index("Grader") < order.index("Price tool")


def test_portfolio_only_route_goes_straight_to_synthesizer():
    memo = Memo(answer="Your largest holding is AAPL.", key_figures=[], sources=[], unverified=[])
    client = ScriptedClient(RouteDecision=decision("portfolio_only", ()), Memo=memo)
    retrieve = counting_retrieve([])
    state = go(client, make_deps(client, retrieve=retrieve))
    assert state["memo"].answer == memo.answer and retrieve.calls == []


def test_out_of_scope_returns_idk_without_retrieval_or_price():
    client = ScriptedClient(RouteDecision=decision("out_of_scope"))
    retrieve = counting_retrieve([passage(1)])
    priced = []
    state = go(client, make_deps(client, retrieve=retrieve, quotes=lambda t: priced.append(t) or {}))
    assert state["memo"].answer == ai.IDK_MESSAGE and retrieve.calls == [] and priced == []
    assert client.calls == ["RouteDecision"]


def test_unexpected_exception_returns_idk_memo():
    def boom(query, tickers):
        raise RuntimeError("unexpected")

    client = ScriptedClient(RouteDecision=decision("sec_filings"))
    state = go(client, make_deps(client, retrieve=boom))
    assert state["memo"].answer == ai.IDK_MESSAGE
    assert any("unexpected" in line.lower() or "failed" in line.lower() for line in state["trace"])


def test_build_graph_compiles():
    assert build_graph(make_deps(ScriptedClient())) is not None
