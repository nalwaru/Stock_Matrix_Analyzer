"""Graph nodes. Each takes (state, deps) and returns a partial state update."""
from agent.deps import Deps
from agent.llm import ask_structured
from agent.schemas import RouteDecision
from agent.state import AgentState
from analyzer import ai

ROUTER_PROMPT = (
    "You route a question about the user's investment portfolio to the right helper. "
    "Choose exactly one route:\n"
    "- sec_filings: needs facts from company SEC filings (10-K, 10-Q, 8-K): results, risks, "
    "business, legal, why something happened.\n"
    "- live_price: needs the current stock price or today's move.\n"
    "- both: needs filings and the current price.\n"
    "- portfolio_only: answerable from the user's own holdings data (values, weights, gains).\n"
    "- out_of_scope: asks for advice, predictions, opinions, market views or anything else "
    "that no data here can answer (for example 'should I buy X?').\n"
    "List the stock tickers the question is about (empty if it is about the whole portfolio). "
    "When unsure between answering and refusing, choose out_of_scope."
)
FILING_ROUTES = {"sec_filings", "both"}
NEXT_NODE = {
    "sec_filings": "retrieve", "both": "retrieve", "live_price": "price",
    "portfolio_only": "synthesize", "out_of_scope": "refuse",
}


def router_node(state: AgentState, deps: Deps) -> dict:
    held = list(state.get("held_tickers", []))
    messages = [
        {"role": "system", "content": ROUTER_PROMPT},
        {"role": "user", "content": (
            f"Held tickers: {', '.join(held) or 'none'}\n"
            f"Tickers with SEC filings available: {', '.join(deps.indexed_tickers) or 'none'}\n"
            f"Question: {state['question']}"
        )},
    ]
    try:
        decision = ask_structured(messages, RouteDecision, deps.model, deps.client)
    except ai.AIError as exc:
        return {
            "route": RouteDecision(route="portfolio_only", tickers=[], reason="routing failed"),
            "trace": [f"Router: routing failed ({exc}); using portfolio data only."],
        }

    route = decision.route
    tickers = list(dict.fromkeys(t.strip().upper() for t in decision.tickers if t.strip()))
    notes = []
    if route in FILING_ROUTES:
        indexed = [t for t in tickers if t in deps.indexed_tickers]
        if not indexed and not tickers:
            indexed = [t for t in held if t in deps.indexed_tickers]
        if not indexed:
            notes.append("no filings are available for these tickers")
            route = "portfolio_only" if route == "sec_filings" else "live_price"
        else:
            tickers = indexed if route == "sec_filings" else list(dict.fromkeys(indexed + tickers))
    if route == "live_price" and not tickers:
        tickers = held
    line = f"Router: {route} ({decision.reason})"
    if notes:
        line += f"; {notes[0]}, so route changed from {decision.route}"
    return {
        "route": RouteDecision(route=route, tickers=tickers, reason=decision.reason),
        "trace": [line],
    }


def route_after_router(state: AgentState) -> str:
    return NEXT_NODE[state["route"].route]
