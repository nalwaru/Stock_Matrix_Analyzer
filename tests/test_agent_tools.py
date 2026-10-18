import pandas as pd

from agent.agent_tools import TOOL_SPECS, run_tool
from agent.schemas import GradeResult, PassageGrade, RewriteResult
from agent.state import Quote
from tests.agent_fakes import ScriptedClient, make_deps, passage
from tests.test_agent_analysis_node import FakeData
from tests.test_agent_portfolio_tools import frame

STATE = {"question": "How did Apple do?", "today": "2026-10-18", "held_tickers": ["AAPL", "NVDA"]}


def grades(**flags):
    return GradeResult(grades=[PassageGrade(id=k, relevant=v, reason="r") for k, v in flags.items()])


def test_tool_specs_name_the_four_tools():
    assert {s["function"]["name"] for s in TOOL_SPECS} == {
        "search_filings", "portfolio_calc", "run_analysts", "get_price"}


def test_search_filings_returns_graded_passages_with_labels():
    deps = make_deps(ScriptedClient(GradeResult=grades(id1=True, id2=False)),
                     retrieve=lambda q, t: [passage(1), passage(2)])
    out = run_tool("search_filings", {"query": "net income", "tickers": ["AAPL"]}, STATE, deps)
    assert [p.id for p in out["graded"]] == ["id1"]
    assert "[AAPL 10-K, 2025-10-31, Item 1A]" in out["observation"]
    assert out["searched_filings"] is True


def test_search_filings_rewrites_once_when_nothing_relevant():
    seen = []
    deps = make_deps(
        ScriptedClient(GradeResult=[grades(id1=False), grades(id1=True)],
                       RewriteResult=RewriteResult(query="net sales")),
        retrieve=lambda q, t: seen.append(q) or [passage(1)])
    out = run_tool("search_filings", {"query": "profit", "tickers": ["AAPL"]}, STATE, deps)
    assert seen == ["profit", "net sales"] and len(out["graded"]) == 1


def test_search_filings_unindexed_ticker_returns_note():
    deps = make_deps(ScriptedClient(), retrieve=lambda q, t: [passage(1)])
    out = run_tool("search_filings", {"query": "q", "tickers": ["ORCL"]}, STATE, deps)
    assert out["graded"] == [] and "no filings are available for ORCL" in out["observation"]
    assert out["no_filings"] == ["ORCL"]


def test_portfolio_calc_runs_calculator():
    deps = make_deps(holdings=frame())
    out = run_tool("portfolio_calc", {"tool": "total"}, STATE, deps)
    assert len(out["facts"]) == 1 and out["facts"][0].label.startswith("PORTFOLIO calc")


def test_portfolio_calc_without_holdings_is_unavailable():
    out = run_tool("portfolio_calc", {"tool": "total"}, STATE, make_deps())
    assert "no portfolio is loaded" in out["observation"].lower() and not out.get("facts")


def test_run_analysts_returns_facts_and_charts():
    deps = make_deps(analyst_data=FakeData())
    out = run_tool("run_analysts", {"ticker": "NVDA", "kinds": ["technical"]}, STATE, deps)
    assert out["facts"] and out["charts"] and "NVDA technical" in out["observation"]


def test_get_price_returns_quotes():
    quote = Quote("AAPL", 250.5, "2026-10-18, 10:00 UTC")
    deps = make_deps(quotes=lambda t: {"AAPL": quote})
    out = run_tool("get_price", {"tickers": ["AAPL"]}, STATE, deps)
    assert out["quotes"]["AAPL"].price == 250.5 and "AAPL live quote" in out["observation"]


def test_unknown_tool_and_bad_args_are_tool_errors():
    deps = make_deps(holdings=frame())
    assert run_tool("nope", {}, STATE, deps)["observation"].startswith("Tool error:")
    assert run_tool("portfolio_calc", {"tool": "bogus"}, STATE, deps)["observation"].startswith("Tool error:")
    assert run_tool("search_filings", {}, STATE, deps)["observation"].startswith("Tool error:")


def test_invalid_argument_error_names_the_allowed_values():
    out = run_tool("run_analysts", {"ticker": "AAPL", "kinds": ["growth"]}, STATE, make_deps())
    text = out["observation"]
    assert text.startswith("Tool error:") and "technical" in text and "ownership" in text
