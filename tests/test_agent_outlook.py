from agent.memo import cited_reasons, idk_memo, render_memo
from agent.nodes import synthesize_node
from agent.schemas import Memo, RouteDecision
from agent.state import Fact
from analyzer import ai
from tests.agent_fakes import ScriptedClient, make_deps, passage

LABEL = "AAPL 10-K, 2025-10-31, Item 1A"
PE = "AAPL fundamental, 2026-10-18, VALUATION"
RSI = "AAPL technical, 2026-10-18, RSI"


def memo(answer="Net income was strong.", outlook=(), leaning="", declined=""):
    return Memo(answer=answer, key_figures=[], sources=[], unverified=[],
                outlook=list(outlook), outlook_leaning=leaning, declined=declined)


def state(outlook=True, facts=None, declined="", route="agent", **extra):
    facts = facts if facts is not None else [Fact(PE, "P/E 30.0"), Fact(RSI, "RSI 55.0")]
    decision = RouteDecision(route=route, tickers=["AAPL"], reason="r",
                             wants_outlook=outlook, declined_part=declined)
    return {"question": "q", "portfolio_json": "{}", "route": decision, "graded": [passage(1)],
            "facts": facts, "quotes": {}, "history": [], **extra}


def run(model_memo, **kw):
    return synthesize_node(state(**kw), make_deps(ScriptedClient(Memo=model_memo)))["memo"]


def test_cited_reasons_drops_uncited_and_unknown_labels():
    reasons = [f"Valuation is moderate [{PE}]", "Momentum looks fine", "Odd [AAPL made up, 2026-01-01, X]"]
    assert cited_reasons(reasons, {PE, RSI}) == [f"Valuation is moderate [{PE}]"]


def test_render_memo_shows_outlook_section_with_estimate_label():
    md = render_memo(memo(outlook=[f"Valuation is moderate [{PE}]"], leaning="neutral"))
    assert "### Outlook" in md and "not a filing fact and not financial advice" in md
    assert "neutral" in md.lower() and PE in md


def test_render_memo_shows_declined_sentence():
    md = render_memo(memo(declined="buy advice"))
    assert "Not answered: buy advice" in md
    assert "### Outlook" not in md
    assert render_memo(idk_memo()) == ai.IDK_MESSAGE


def test_synthesize_keeps_cited_outlook_and_drops_uncited():
    out = run(memo(outlook=[f"P/E is 30.0 [{PE}]", "Looks good"], leaning="positive"))
    assert out.outlook == [f"P/E is 30.0 [{PE}]"] and out.outlook_leaning == "positive"


def test_synthesize_numbers_in_outlook_are_checked():
    out = run(memo(outlook=[f"Growth of 47.5% expected [{PE}]"], leaning="positive"))
    assert any("47.5%" in u for u in out.unverified)


def test_synthesize_declined_comes_from_route_not_model():
    out = run(memo(declined="model wrote this"), outlook=False, declined="buy advice")
    assert out.declined == "buy advice"


def test_synthesize_outlook_not_requested_is_empty():
    out = run(memo(outlook=[f"P/E is 30.0 [{PE}]"], leaning="positive"), outlook=False)
    assert out.outlook == [] and out.outlook_leaning == ""


def test_synthesize_outlook_with_all_unavailable_data_has_no_leaning():
    facts = [Fact(PE, "AAPL fundamental analysis is unavailable")]
    out = run(memo(outlook=[f"Cheap [{PE}]"], leaning="positive"), facts=facts)
    assert out.outlook_leaning == "" and out.outlook == []
    assert any("insufficient" in u.lower() for u in out.unverified)


def test_synthesize_without_any_cited_reason_clears_leaning():
    out = run(memo(outlook=["Vibes"], leaning="positive"))
    assert out.outlook_leaning == "" and any("insufficient" in u.lower() for u in out.unverified)


def test_synthesize_step_limit_adds_incomplete_evidence_line():
    out = run(memo(), outlook=False, step_limit_hit=True)
    assert any("step limit" in u for u in out.unverified)


def test_synthesize_agent_route_with_no_evidence_is_idk():
    client = ScriptedClient()
    bare = {"question": "q", "portfolio_json": "{}", "history": [], "graded": [], "facts": [], "quotes": {},
            "route": RouteDecision(route="agent", tickers=["AAPL"], reason="r")}
    out = synthesize_node(bare, make_deps(client))
    assert out["memo"].answer == ai.IDK_MESSAGE and client.calls == []


def test_agent_route_notes_empty_filing_search():
    out = run(memo(), outlook=False, graded=[], searched_filings=True)
    assert any("No relevant SEC filing passages" in u for u in out.unverified)
