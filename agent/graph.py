"""LangGraph assembly: router -> (filings loop | price | portfolio | refusal) -> memo."""
from functools import partial

from langgraph.graph import END, START, StateGraph

from agent.deps import Deps
from agent.memo import idk_memo
from agent.nodes import (grade_node, price_node, refuse_node, retrieve_node, rewrite_node,
                         route_after_grade, route_after_router, router_node, synthesize_node)
from agent.state import AgentState


def build_graph(deps: Deps):
    graph = StateGraph(AgentState)
    for name, node in (
        ("router", router_node), ("retrieve", retrieve_node), ("grade", grade_node),
        ("rewrite", rewrite_node), ("price", price_node), ("synthesize", synthesize_node),
        ("refuse", refuse_node),
    ):
        graph.add_node(name, partial(node, deps=deps))
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router, {
        "retrieve": "retrieve", "price": "price", "synthesize": "synthesize", "refuse": "refuse"})
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {
        "price": "price", "synthesize": "synthesize", "rewrite": "rewrite"})
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
        "held_tickers": held_tickers, "rewrites": 0, "trace": [],
    }
    try:
        return build_graph(deps).invoke(initial)
    except Exception as exc:  # noqa: BLE001 - the tab must never crash
        return {**initial, "memo": idk_memo(),
                "trace": initial["trace"] + [f"Agent failed unexpectedly ({type(exc).__name__})"]}
