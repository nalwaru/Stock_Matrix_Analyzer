"""Tools the agent can call. Each wraps an existing node or calculator and returns a partial
state update plus `observation`, the text the model sees. Nothing here raises."""
from typing import Literal

from pydantic import BaseModel, ValidationError

from agent.deps import Deps
from agent.memo import passage_label, quote_label
from agent.nodes import (MAX_REWRITES, analysis_node, grade_node, price_node, retrieve_node,
                         rewrite_node)
from agent.schemas import CalcCall, RouteDecision
from agent.state import AgentState
from agent.tools.portfolio import run_calc
from analyzer import ai

MAX_FILING_REWRITES = 1


class SearchFilingsArgs(BaseModel):
    query: str
    tickers: list[str] = []


class RunAnalystsArgs(BaseModel):
    ticker: str
    kinds: list[Literal["technical", "fundamental", "risk", "ownership"]]


class GetPriceArgs(BaseModel):
    tickers: list[str]


def _spec(name: str, description: str, model: type[BaseModel]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description, "parameters": model.model_json_schema()}}


TOOL_SPECS = [
    _spec("search_filings", "Search the company's SEC filings (10-K, 10-Q, 8-K) for passages "
          "that answer a query. Use filing vocabulary such as 'net sales' or 'risk factors'.",
          SearchFilingsArgs),
    _spec("portfolio_calc", "Run one exact calculation on the user's holdings: holding, "
          "class_weight, top_n, total or stress. Unused arguments are null.", CalcCall),
    _spec("run_analysts", "Run technical, fundamental, risk and/or ownership checks for one "
          "ticker. Use for valuation, margins, growth, volatility, trend, analyst targets and "
          "outlook questions.", RunAnalystsArgs),
    _spec("get_price", "Get live prices for tickers.", GetPriceArgs),
]


def _error(message: str) -> dict:
    return {"observation": f"Tool error: {message}", "trace": [f"Tool error: {message}"]}


def _facts_text(facts) -> str:
    return "\n".join(f"[{f.label}] {f.text}" for f in facts)


def _search_filings(args: SearchFilingsArgs, state: AgentState, deps: Deps) -> dict:
    wanted = [t.strip().upper() for t in args.tickers if t.strip()] or list(state.get("tickers", []))
    indexed = [t for t in wanted if t in deps.indexed_tickers]
    dropped = [t for t in wanted if t not in deps.indexed_tickers]
    notes = [f"no filings are available for {t}" for t in dropped]
    trace = []
    if not indexed:
        text = "; ".join(notes) or "no filings are available for these tickers"
        return {"graded": [], "searched_filings": True, "trace": [f"search_filings: {text}"],
                "observation": f"No passages: {text}."}
    work = {"question": state["question"], "query": args.query, "rewrites": 0,
            "route": RouteDecision(route="agent", tickers=indexed, reason="tool")}
    graded: list = []
    for _ in range(MAX_FILING_REWRITES + 1):
        work.update(retrieve_node(work, deps))
        trace += work.get("trace", [])
        work.update(grade_node(work, deps))
        trace += work.get("trace", [])
        graded = work.get("graded", [])
        if graded or work["rewrites"] >= MAX_FILING_REWRITES or work["rewrites"] >= MAX_REWRITES:
            break
        work.update(rewrite_node(work, deps))
        trace += work.get("trace", [])
    body = "\n\n".join(f"[{passage_label(p)}]\n{p.text[: ai.MAX_MESSAGE_CHARS]}" for p in graded)
    observation = body or "No relevant passages were found."
    if notes:
        observation += "\n(" + "; ".join(notes) + ")"
    return {"graded": graded, "searched_filings": True, "trace": trace, "observation": observation}


def _portfolio_calc(args: CalcCall, state: AgentState, deps: Deps) -> dict:
    if deps.holdings is None:
        return {"observation": "No portfolio is loaded.", "trace": ["portfolio_calc: no portfolio loaded"]}
    fact = run_calc(deps.holdings, args, state.get("today", ""))
    if fact is None:
        return {"observation": "That calculation matched nothing (unknown ticker or asset class, "
                               "or missing arguments).", "trace": ["portfolio_calc: no result"]}
    return {"facts": [fact], "observation": _facts_text([fact]), "trace": []}


def _run_analysts(args: RunAnalystsArgs, state: AgentState, deps: Deps) -> dict:
    work = {**state, "route": RouteDecision(route="agent", tickers=[args.ticker.strip().upper()],
                                            reason="tool"), "analyses": list(args.kinds)}
    out = analysis_node(work, deps)
    facts = out.get("facts", [])
    return {"facts": facts, "charts": out.get("charts", []), "trace": out.get("trace", []),
            "observation": _facts_text(facts) or "No analyst data was available."}


def _get_price(args: GetPriceArgs, state: AgentState, deps: Deps) -> dict:
    tickers = [t.strip().upper() for t in args.tickers if t.strip()]
    out = price_node({**state, "route": RouteDecision(route="agent", tickers=tickers, reason="tool")}, deps)
    quotes = out.get("quotes", {})
    lines = [f"[{quote_label(q)}] price {q.price:.2f}" if q.price is not None
             else f"{t}: price unavailable" for t, q in quotes.items()]
    observation = "\n".join(lines + [f"[{f.label}] {f.text}" for f in out.get("facts", [])])
    return {"quotes": quotes, "facts": out.get("facts", []), "trace": out.get("trace", []),
            "observation": observation or "No prices were available."}


_TOOLS = {
    "search_filings": (SearchFilingsArgs, _search_filings),
    "portfolio_calc": (CalcCall, _portfolio_calc),
    "run_analysts": (RunAnalystsArgs, _run_analysts),
    "get_price": (GetPriceArgs, _get_price),
}


def run_tool(name: str, args: dict, state: AgentState, deps: Deps) -> dict:
    if name not in _TOOLS:
        return _error(f"unknown tool {name!r}")
    model, handler = _TOOLS[name]
    try:
        parsed = model.model_validate(args)
    except ValidationError:
        return _error(f"invalid arguments for {name}")
    try:
        return handler(parsed, state, deps)
    except Exception as exc:  # noqa: BLE001 - a tool must never break the loop
        return _error(f"{name} failed ({type(exc).__name__})")
