from agent.memo import check_numbers, labels_for
from agent.nodes import (route_after_grade, route_after_router,
                         synthesize_node)
from agent.schemas import Memo, RouteDecision
from agent.state import Fact
from analyzer import ai
from tests.agent_fakes import ScriptedClient, make_deps, passage

FACT = Fact("PORTFOLIO calc, 2026-10-17, weight_AAPL", "AAPL is 17.40% of the portfolio (value 3,750.00)")


def route(name, analyses=()):
    return RouteDecision(route=name, tickers=["AAPL"], reason="r", analyses=list(analyses))


def test_check_numbers_matches_supplied_and_flags_unsupplied():
    evidence = "AAPL is 17.40% of the portfolio (value 3,750.00); net sales 416,161"
    assert check_numbers(["Weight is 17.4% and value 3750"], evidence) == []
    assert check_numbers(["Value is 9,999.00"], evidence) == ["9,999.00"]


def test_check_numbers_accepts_unit_scaling():
    assert check_numbers(["Net sales were 416.161 billion"], "net sales 416,161 million") == []


def test_check_numbers_ignores_citations_dates_years_and_small_numbers():
    text = "In 2025 there were 3 risks [AAPL 10-K, 2025-10-31, Item 1A] on 2026-10-17."
    assert check_numbers([text], "nothing numeric here") == []


def test_labels_for_includes_fact_labels():
    assert FACT.label in labels_for({"facts": [FACT]})


def test_synthesizer_uses_facts_as_evidence_and_flags_unmatched_figures():
    memo = Memo(answer=f"AAPL is 17.40% of the portfolio [{FACT.label}] and worth 9,999.00.",
                key_figures=[], sources=[FACT.label], unverified=[])
    client = ScriptedClient(Memo=memo)
    state = {"question": "q", "portfolio_json": "{}", "route": route("live_price"),
             "graded": [], "quotes": {}, "facts": [FACT], "history": []}
    out = synthesize_node(state, make_deps(client))
    assert out["memo"].answer == memo.answer
    assert any("9,999.00" in u for u in out["memo"].unverified)


def test_synthesizer_with_only_facts_does_not_refuse():
    memo = Memo(answer=f"See [{FACT.label}].", key_figures=[], sources=[], unverified=[])
    state = {"question": "q", "portfolio_json": "{}", "route": route("analysis"),
             "graded": [], "quotes": {}, "facts": [FACT], "history": []}
    assert synthesize_node(state, make_deps(ScriptedClient(Memo=memo)))["memo"].answer == memo.answer


def test_analysis_route_without_any_evidence_is_idk():
    state = {"question": "q", "portfolio_json": "{}", "route": route("analysis"),
             "graded": [], "quotes": {}, "facts": [], "history": []}
    assert synthesize_node(state, make_deps(ScriptedClient()))["memo"].answer == ai.IDK_MESSAGE


def test_routing_functions_for_tools():
    assert route_after_router({"route": route("portfolio_only")}) == "tools"
    assert route_after_router({"route": route("analysis")}) == "analysis"
    assert route_after_grade({"route": route("sec_filings"), "graded": [passage(1)]}) == "synthesize"
    assert route_after_grade({"route": route("sec_filings"), "graded": [], "rewrites": 2}) == "synthesize"
    assert route_after_grade({"route": route("sec_filings"), "graded": [], "rewrites": 0}) == "rewrite"


def test_check_numbers_sees_numbers_inside_bracketed_json_evidence():
    evidence = '{"holdings": [{"Ticker": "SCHD", "Gain_Loss": -1526.0}], "total": 2}'
    assert check_numbers(["SCHD lost 1,526.00"], evidence) == []
    assert check_numbers(["loss of -1526.0"], evidence) == []


def test_labels_for_includes_the_portfolio_data_label_when_today_is_known():
    assert "PORTFOLIO data, 2026-10-18, holdings" in labels_for({"today": "2026-10-18"})
    assert not any(label.startswith("PORTFOLIO data") for label in labels_for({}))
