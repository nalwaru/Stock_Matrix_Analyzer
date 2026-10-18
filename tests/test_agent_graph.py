from agent.graph import build_graph, run
from agent.memo import render_memo
from agent.schemas import GradeResult, Memo, PassageGrade, RewriteResult, RouteDecision, ToolPlan
from agent.state import Quote
from analyzer import ai
from tests.agent_fakes import (ScriptedClient, assistant, make_deps, passage, tool_call,
                               with_tools)
from tests.test_agent_portfolio_tools import frame

LABEL = "AAPL 10-K, 2025-10-31, Item 1A"
SEARCH = tool_call("c1", "search_filings", {"query": "risk factors", "tickers": ["AAPL"]})


def decision(route, tickers=("AAPL",), **kw):
    return RouteDecision(route=route, tickers=list(tickers), reason="r", **kw)


def grade(relevant=True, pid="id1"):
    return GradeResult(grades=[PassageGrade(id=pid, relevant=relevant, reason="r")])


def good_memo():
    return Memo(answer=f"Risk exists [{LABEL}].", key_figures=[], sources=[LABEL], unverified=[])


def go(deps, question="q"):
    return run(question, [], "{}", ["AAPL"], deps)


def counting_retrieve(passages):
    calls = []

    def retrieve(query, tickers):
        calls.append(query)
        return passages

    retrieve.calls = calls
    return retrieve


def test_agent_route_runs_agent_then_synthesizer():
    client = with_tools(ScriptedClient(RouteDecision=decision("agent"), GradeResult=grade(), Memo=good_memo()),
                        [assistant(tool_calls=[SEARCH]), assistant("done")])
    retrieve = counting_retrieve([passage(1)])
    state = go(make_deps(client, retrieve=retrieve))
    assert state["memo"].answer.startswith("Risk exists") and len(retrieve.calls) == 1
    assert [n for n, _ in state["timings"]] == ["router", "agent", "synthesize"]
    assert any(line.startswith("Agent tool: search_filings") for line in state["trace"])


def test_agent_route_with_no_relevant_passages_rewrites_once_then_idk():
    client = with_tools(
        ScriptedClient(RouteDecision=decision("agent"), GradeResult=[grade(False), grade(False)],
                       RewriteResult=RewriteResult(query="r1")),
        [assistant(tool_calls=[SEARCH]), assistant("done")])
    retrieve = counting_retrieve([passage(1)])
    state = go(make_deps(client, retrieve=retrieve))
    assert state["memo"].answer == ai.IDK_MESSAGE and retrieve.calls == ["risk factors", "r1"]
    assert "Memo" not in client.calls


def test_live_price_route_skips_retrieval_and_agent():
    quote = Quote("AAPL", 250.0, "2026-10-17, 16:45 UTC")
    text = "Price is 250 [AAPL live quote, 2026-10-17, 16:45 UTC]."
    client = ScriptedClient(RouteDecision=decision("live_price"),
                            Memo=Memo(answer=text, key_figures=[], sources=[], unverified=[]))
    retrieve = counting_retrieve([passage(1)])
    state = go(make_deps(client, retrieve=retrieve, quotes=lambda t: {"AAPL": quote}))
    assert state["memo"].answer == text and retrieve.calls == []
    assert [n for n, _ in state["timings"]] == ["router", "price", "synthesize"]
    assert client.calls == ["RouteDecision", "Memo"]


def test_agent_can_combine_filings_and_price_in_one_question():
    quote = Quote("AAPL", 250.0, "2026-10-17, 16:45 UTC")
    memo = Memo(answer=f"[{LABEL}] and [AAPL live quote, 2026-10-17, 16:45 UTC]",
                key_figures=[], sources=[], unverified=[])
    price = tool_call("c2", "get_price", {"tickers": ["AAPL"]})
    client = with_tools(ScriptedClient(RouteDecision=decision("agent"), GradeResult=grade(), Memo=memo),
                        [assistant(tool_calls=[SEARCH, price]), assistant("done")])
    state = go(make_deps(client, retrieve=counting_retrieve([passage(1)]), quotes=lambda t: {"AAPL": quote}))
    assert state["memo"].answer == memo.answer


def test_portfolio_only_keeps_three_model_calls():
    from agent.schemas import CalcCall
    plan = ToolPlan(calc_calls=[CalcCall(tool="total", ticker=None, asset_class=None, metric=None,
                                         n=None, shock_pct=None, order=None)])
    memo = Memo(answer="Your portfolio is worth 3,000.00.", key_figures=[], sources=[], unverified=[])
    client = ScriptedClient(RouteDecision=decision("portfolio_only", ()), ToolPlan=plan, Memo=memo)
    retrieve = counting_retrieve([])
    state = go(make_deps(client, retrieve=retrieve, holdings=frame()))
    assert state["memo"].answer == memo.answer and retrieve.calls == []
    assert client.calls == ["RouteDecision", "ToolPlan", "Memo"]


def test_out_of_scope_returns_idk_without_any_other_call():
    client = ScriptedClient(RouteDecision=decision("out_of_scope"))
    retrieve = counting_retrieve([passage(1)])
    priced = []
    state = go(make_deps(client, retrieve=retrieve, quotes=lambda t: priced.append(t) or {}))
    assert state["memo"].answer == ai.IDK_MESSAGE and retrieve.calls == [] and priced == []
    assert client.calls == ["RouteDecision"]


def test_partial_decline_answers_in_scope_part_and_states_the_decline():
    client = with_tools(
        ScriptedClient(RouteDecision=decision("agent", declined_part="buy advice"), GradeResult=grade(),
                       Memo=good_memo()),
        [assistant(tool_calls=[SEARCH]), assistant("done")])
    state = go(make_deps(client, retrieve=counting_retrieve([passage(1)])), "Should I buy Apple, and what are its risks?")
    assert state["memo"].declined == "buy advice"
    assert "Not answered: buy advice" in render_memo(state["memo"])


def test_unexpected_exception_returns_idk_memo():
    def boom(tickers):
        raise RuntimeError("unexpected")

    client = ScriptedClient(RouteDecision=decision("live_price"))
    state = go(make_deps(client, quotes=boom))
    assert state["memo"].answer == ai.IDK_MESSAGE
    assert any("unexpected" in line.lower() or "failed" in line.lower() for line in state["trace"])


def test_build_graph_compiles():
    assert build_graph(make_deps(ScriptedClient())) is not None


def test_unindexed_ticker_end_to_end_says_no_filings_are_available():
    call = tool_call("c1", "search_filings", {"query": "risks", "tickers": ["ORCL"]})
    client = with_tools(ScriptedClient(RouteDecision=decision("agent", ("ORCL",))), [assistant(tool_calls=[call]), assistant("done")])
    state = go(make_deps(client, retrieve=counting_retrieve([passage(1)])), "What are Oracle's main risks?")
    assert state["memo"].answer == ai.IDK_MESSAGE
    assert "No SEC filings are available for ORCL" in render_memo(state["memo"])
