"""LangGraph assembly: router -> (filings loop | price | portfolio | refusal) -> memo."""
from collections.abc import Iterator
from datetime import datetime, timezone
import time

from langgraph.graph import END, START, StateGraph

from agent.deps import Deps
from agent.memo import idk_memo
from agent.nodes import (analysis_node, grade_node, price_node, refuse_node, retrieve_node,
                         rewrite_node, route_after_grade, route_after_router,
                         router_node, synthesize_node, tools_node)
from agent.state import AgentState


def _timed(name: str, node, deps: Deps):
    def run_node(state: AgentState) -> dict:
        start = time.perf_counter()
        update = node(state, deps=deps) or {}
        return {**update, "timings": [(name, round(time.perf_counter() - start, 3))]}
    return run_node


def build_graph(deps: Deps):
    graph = StateGraph(AgentState)
    for name, node in (
        ("router", router_node), ("retrieve", retrieve_node), ("grade", grade_node),
        ("rewrite", rewrite_node), ("price", price_node), ("synthesize", synthesize_node),
        ("refuse", refuse_node), ("tools", tools_node), ("analysis", analysis_node),
    ):
        graph.add_node(name, _timed(name, node, deps))
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router, {
        "retrieve": "retrieve", "price": "price", "tools": "tools", "analysis": "analysis",
        "refuse": "refuse"})
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {
        "price": "price", "synthesize": "synthesize", "rewrite": "rewrite"})
    graph.add_edge("tools", "synthesize")
    graph.add_edge("analysis", "synthesize")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("price", "synthesize")
    graph.add_edge("synthesize", END)
    graph.add_edge("refuse", END)
    return graph.compile()


def run(question: str, history: list[dict], portfolio_json: str,
        held_tickers: list[str], deps: Deps) -> AgentState:
    """Run the graph; never raises. Any unexpected failure yields the polite refusal memo."""
    initial: AgentState = {
        "question": question, "history": history, "portfolio_json": portfolio_json,
        "held_tickers": held_tickers, "rewrites": 0, "trace": [], "timings": [],
        "today": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    try:
        return build_graph(deps).invoke(initial)
    except Exception as exc:  # noqa: BLE001 - the tab must never crash
        return {**initial, "memo": idk_memo(),
                "trace": initial["trace"] + [f"Agent failed unexpectedly ({type(exc).__name__})"]}


NODE_LABELS = {
    "router": "Working out what your question needs",
    "retrieve": "Searching the SEC filings",
    "grade": "Checking which passages really answer it",
    "rewrite": "Rewording the search",
    "tools": "Running calculations",
    "price": "Fetching live prices",
    "analysis": "Running the analyst checks",
    "synthesize": "Writing the memo",
    "refuse": "Checking whether this is something I can answer",
}


def run_stream(question: str, history: list[dict], portfolio_json: str,
               held_tickers: list[str], deps: Deps) -> Iterator[tuple[str, dict]]:
    """Yield (node, update) as each node finishes, then ("done", final_state). Never raises."""
    state: AgentState = {
        "question": question, "history": history, "portfolio_json": portfolio_json,
        "held_tickers": held_tickers, "rewrites": 0, "trace": [], "timings": [],
        "today": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    try:
        for chunk in build_graph(deps).stream(state, stream_mode="updates"):
            for node, update in chunk.items():
                update = update or {}
                for key, value in update.items():
                    state[key] = state.get(key, []) + value if key in ("trace", "facts", "charts", "timings") else value
                yield node, update
    except Exception as exc:  # noqa: BLE001 - the tab must never crash
        state["memo"] = idk_memo()
        state["trace"] = state.get("trace", []) + [f"Agent failed unexpectedly ({type(exc).__name__})"]
    yield "done", state
