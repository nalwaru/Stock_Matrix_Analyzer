"""Shared graph state."""
import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict


@dataclass(frozen=True)
class Quote:
    ticker: str
    price: float | None
    as_of: str  # "YYYY-MM-DD, HH:MM UTC"


@dataclass(frozen=True)
class Fact:
    """A tool result computed in Python. `label` is citable: 'TICKER form, YYYY-MM-DD, section'."""
    label: str
    text: str


class AgentState(TypedDict, total=False):
    question: str
    today: str  # UTC date used in citation labels
    history: list[dict]
    portfolio_json: str
    held_tickers: list[str]
    route: object  # schemas.RouteDecision
    query: str
    passages: list
    graded: list
    rewrites: int
    tickers: list[str]  # resolved by the gate; the agent's default scope
    step_limit_hit: bool
    searched_filings: bool
    no_filings: list[str]  # tickers the agent asked about that have no indexed filings
    agent_stopped_early: bool
    quotes: dict[str, Quote]
    memo: object  # schemas.Memo
    facts: Annotated[list[Fact], operator.add]  # tool results; nodes append
    charts: Annotated[list, operator.add]  # plotly figures built by analysts
    timings: Annotated[list[tuple[str, float]], operator.add]  # (node, seconds) per node run
    trace: Annotated[list[str], operator.add]  # nodes append; the graph concatenates
