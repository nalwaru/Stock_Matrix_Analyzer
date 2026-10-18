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


PORTFOLIO = "PORTFOLIO calc, 2026-10-18, holding_AAPL"


def test_outlook_cannot_rest_on_portfolio_or_filing_labels():
    facts = [Fact(PORTFOLIO, "AAPL is down 12.0%"), Fact(PE, "AAPL fundamental analysis is unavailable")]
    reasons = [f"Your position is down 12.0% [{PORTFOLIO}]", f"Supply pressure [{LABEL}]"]
    out = run(memo(outlook=reasons, leaning="cautious"), facts=facts)
    assert out.outlook == [] and out.outlook_leaning == ""
    assert any("insufficient" in u.lower() for u in out.unverified)


def test_outlook_drops_reason_with_an_unsupplied_citation_in_the_middle():
    bad = f"Services will grow [AAPL 10-K, 2024-01-01, Item 7], supported by [{PE}]"
    out = run(memo(outlook=[bad, f"P/E is 30.0 [{PE}]"], leaning="neutral"))
    assert out.outlook == [f"P/E is 30.0 [{PE}]"]


def test_cited_reasons_requires_every_bracket_to_be_supplied():
    assert cited_reasons([f"a [{PE}] b [{RSI}]", f"a [{PE}] b [nope]"], {PE, RSI}) == [f"a [{PE}] b [{RSI}]"]


def test_idk_memo_names_tickers_without_filings():
    bare = {"question": "q", "portfolio_json": "{}", "history": [], "graded": [], "facts": [], "quotes": {},
            "no_filings": ["ORCL"], "searched_filings": True,
            "route": RouteDecision(route="agent", tickers=["ORCL"], reason="r")}
    out = synthesize_node(bare, make_deps(ScriptedClient()))["memo"]
    assert out.answer == ai.IDK_MESSAGE and out.no_filings == ["ORCL"]
    assert "No SEC filings are available for ORCL" in render_memo(out)


def test_memo_with_evidence_lists_tickers_without_filings():
    out = run(memo(), outlook=False, no_filings=["ORCL"])
    assert any("No SEC filings are available for ORCL" in u for u in out.unverified)


def test_agent_stopped_early_is_flagged():
    out = run(memo(), outlook=False, agent_stopped_early=True)
    assert any("stopped early" in u for u in out.unverified)


def test_outlook_section_carries_a_not_a_financial_advisor_line():
    md = render_memo(memo(outlook=[f"P/E is 30.0 [{PE}]"], leaning="neutral"))
    assert "not a financial advisor" in md.lower()


def test_outlook_answer_skips_filing_availability_notes():
    out = run(memo(outlook=[f"P/E is 30.0 [{PE}]"], leaning="neutral"), no_filings=["ORCL"],
              graded=[], searched_filings=True)
    assert out.outlook_leaning == "neutral"
    assert not any("filing" in u.lower() for u in out.unverified)


def test_filing_note_appears_once_when_the_answer_needs_filings():
    out = run(memo(), outlook=False, no_filings=["ORCL"], graded=[], searched_filings=True)
    notes = [u for u in out.unverified if "filing" in u.lower()]
    assert notes == ["No SEC filings are available for ORCL"]
    assert "No SEC filings are available" not in render_memo(out).split("### What I couldn't verify")[0]
    assert render_memo(out).count("No SEC filings are available for ORCL") == 1
