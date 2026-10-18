import openai

from agent.memo import idk_memo, labels_for, render_memo
from agent.nodes import refuse_node, synthesize_node
from agent.schemas import Memo, RouteDecision
from agent.state import Quote
from analyzer import ai
from tests.agent_fakes import ScriptedClient, make_deps, passage

LABEL = "AAPL 10-K, 2025-10-31, Item 1A"
QUOTE = Quote("AAPL", 250.0, "2026-10-17, 16:45 UTC")


def route(name):
    return RouteDecision(route=name, tickers=["AAPL"], reason="r")


def memo(answer, figures=(), sources=(), unverified=()):
    return Memo(answer=answer, key_figures=list(figures), sources=list(sources), unverified=list(unverified))


def state(route_name="agent", graded=(), quotes=None):
    return {"question": "q", "portfolio_json": "{}", "route": route(route_name),
            "graded": list(graded), "quotes": quotes or {}, "history": []}


def test_grounded_replaces_unsupplied_citation_and_keeps_supplied():
    assert ai.grounded("x [AAPL 10-K, 2020-01-01, Item 7]", {LABEL}) == ai.IDK_MESSAGE
    text = f"ok [{LABEL}]"
    assert ai.grounded(text, {LABEL}) == text
    assert ai.grounded("no citation", set()) == "no citation"


def test_render_memo_has_four_sections_and_idk_is_plain():
    md = render_memo(memo("Sales rose.", ["Sales: $5"], [LABEL], ["Price unavailable"]))
    assert md.startswith("Sales rose.")
    for label in ("Key figures", "Sources", "What I couldn't verify"):
        assert f"**{label}**" in md
    assert render_memo(idk_memo()) == ai.IDK_MESSAGE


def test_labels_cover_graded_passages_and_priced_quotes_only():
    s = state(graded=[passage(1)], quotes={"AAPL": QUOTE, "NVDA": Quote("NVDA", None, "t")})
    assert labels_for(s) == {LABEL, "AAPL live quote, 2026-10-17, 16:45 UTC"}


def test_synthesize_without_evidence_returns_idk_without_model_call():
    client = ScriptedClient()
    out = synthesize_node(state("agent", graded=[]), make_deps(client))
    assert out["memo"].answer == ai.IDK_MESSAGE and client.calls == []
    out = synthesize_node(state("live_price", quotes={"AAPL": Quote("AAPL", None, "t")}), make_deps(client))
    assert out["memo"].answer == ai.IDK_MESSAGE and client.calls == []


def test_synthesize_with_bad_citation_returns_idk():
    bad = memo("Sales fell [AAPL 10-K, 2019-01-01, Item 7].")
    client = ScriptedClient(Memo=bad)
    out = synthesize_node(state(graded=[passage(1)]), make_deps(client))
    assert out["memo"].answer == ai.IDK_MESSAGE


def test_synthesize_keeps_supported_memo_and_adds_missing_price_note():
    good = memo(f"Risk exists [{LABEL}].", [], [LABEL], [])
    client = ScriptedClient(Memo=good)
    quotes = {"AAPL": QUOTE, "NVDA": Quote("NVDA", None, "t")}
    out = synthesize_node(state("agent", graded=[passage(1)], quotes=quotes), make_deps(client))
    assert out["memo"].answer == good.answer
    assert any("NVDA" in u for u in out["memo"].unverified)


def test_synthesizer_failure_returns_idk():
    client = ScriptedClient(Memo=openai.OpenAIError("x"))
    out = synthesize_node(state("portfolio_only"), make_deps(client))
    assert out["memo"].answer == ai.IDK_MESSAGE


def test_portfolio_only_calls_model_and_live_price_citation_is_allowed():
    text = "Price is $250 [AAPL live quote, 2026-10-17, 16:45 UTC]."
    client = ScriptedClient(Memo=memo(text))
    out = synthesize_node(state("live_price", quotes={"AAPL": QUOTE}), make_deps(client))
    assert out["memo"].answer == text


def test_refuse_node_is_idk():
    assert refuse_node(state("out_of_scope"), make_deps())["memo"].answer == ai.IDK_MESSAGE


def test_evidence_tells_the_writer_which_tickers_the_question_resolved_to():
    from agent.nodes import SYNTH_PROMPT, _evidence_block

    s = {"question": "technical analysis orcale", "portfolio_json": "{}", "graded": [], "quotes": {},
         "facts": [], "route": RouteDecision(route="agent", tickers=["ORCL"], reason="r")}
    block = _evidence_block(s)
    assert "ORCL" in block and "resolved" in block.lower()
    assert "misspell" in SYNTH_PROMPT.lower() or "company name" in SYNTH_PROMPT.lower()


def test_no_resolved_tickers_line_when_the_route_has_none():
    from agent.nodes import _evidence_block

    s = {"question": "q", "portfolio_json": "{}", "graded": [], "quotes": {}, "facts": [],
         "route": RouteDecision(route="portfolio_only", tickers=[], reason="r")}
    assert "resolved" not in _evidence_block(s).lower()


def test_refuse_node_states_the_declined_part():
    from agent.memo import render_memo
    declined = RouteDecision(route="out_of_scope", tickers=["AAPL"], reason="r", declined_part="recent news")
    out = refuse_node({**state("out_of_scope"), "route": declined}, make_deps())["memo"]
    assert out.answer == ai.IDK_MESSAGE and out.declined == "recent news"
    assert "Not answered: recent news" in render_memo(out)


def test_router_prompt_declines_news_requests_with_a_clear_reason():
    from agent.nodes import ROUTER_PROMPT
    assert "news" in ROUTER_PROMPT.lower() and "no news source" in ROUTER_PROMPT.lower()
