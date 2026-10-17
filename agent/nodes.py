"""Graph nodes. Each takes (state, deps) and returns a partial state update."""
from agent.deps import Deps
from agent.llm import ask_structured
from agent.schemas import GradeResult, RewriteResult, RouteDecision
from agent.state import AgentState
from analyzer import ai
from rag.retrieve import RetrievalError

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


MAX_REWRITES = 2
PASSAGE_CHARS = 1500
GRADER_PROMPT = (
    "You grade search results for a question about a company. For each passage decide "
    "whether it contains information that helps answer the question. Be strict: a passage "
    "that only shares keywords or is general boilerplate is not relevant. Return one grade "
    "per passage id with a one-line reason."
)
REWRITER_PROMPT = (
    "The search for the question below found nothing relevant. Rewrite it as a short, "
    "specific search query for SEC filing text (use the vocabulary filings use, such as "
    "'net sales', 'risk factors', 'repurchased'). Do not answer the question."
)


def _label(p) -> str:
    return f"{p.ticker} {p.form}, {p.filing_date}, {p.section}"


def retrieve_node(state: AgentState, deps: Deps) -> dict:
    query = state.get("query") or state["question"]
    tickers = state["route"].tickers
    try:
        passages = deps.retrieve(query, tickers)
    except RetrievalError as exc:
        return {"query": query, "passages": [], "trace": [f"Filings search unavailable ({exc})"]}
    return {
        "query": query, "passages": passages,
        "trace": [f"Filings search \"{query}\" -> {len(passages)} passages"],
    }


def grade_node(state: AgentState, deps: Deps) -> dict:
    passages = state.get("passages", [])
    if not passages:
        return {"graded": [], "trace": ["Grader: no passages to grade"]}
    listing = "\n\n".join(f"id: {p.id}\n[{_label(p)}]\n{p.text[:PASSAGE_CHARS]}" for p in passages)
    messages = [
        {"role": "system", "content": GRADER_PROMPT},
        {"role": "user", "content": f"Question: {state['question']}\n\nPassages:\n{listing}"},
    ]
    try:
        result = ask_structured(messages, GradeResult, deps.model, deps.client)
    except ai.AIError as exc:
        return {"graded": [], "trace": [f"Grader failed ({exc}); discarding passages"]}
    relevant = {g.id for g in result.grades if g.relevant}
    graded = [p for p in passages if p.id in relevant]
    return {"graded": graded, "trace": [f"Grader: kept {len(graded)} of {len(passages)} passages"]}


def rewrite_node(state: AgentState, deps: Deps) -> dict:
    rewrites = state.get("rewrites", 0) + 1
    old = state.get("query") or state["question"]
    messages = [
        {"role": "system", "content": REWRITER_PROMPT},
        {"role": "user", "content": f"Question: {state['question']}\nPrevious query: {old}"},
    ]
    try:
        new = ask_structured(messages, RewriteResult, deps.model, deps.client).query.strip() or old
    except ai.AIError as exc:
        return {"rewrites": rewrites, "query": old, "trace": [f"Rewriter failed ({exc}); keeping query"]}
    return {"rewrites": rewrites, "query": new, "trace": [f"Rewriter ({rewrites}/{MAX_REWRITES}): \"{new}\""]}


def route_after_grade(state: AgentState) -> str:
    after = "price" if state["route"].route == "both" else "synthesize"
    if state.get("graded"):
        return after
    return "rewrite" if state.get("rewrites", 0) < MAX_REWRITES else after


def price_node(state: AgentState, deps: Deps) -> dict:
    tickers = state["route"].tickers or list(state.get("held_tickers", []))
    quotes = deps.fetch_quotes(tickers)
    missing = [t for t, q in quotes.items() if q.price is None]
    line = f"Price tool: {len(quotes) - len(missing)} of {len(quotes)} quotes"
    if missing:
        line += f"; unavailable: {', '.join(missing)}"
    return {"quotes": quotes, "trace": [line]}
