"""Shared graph state."""
from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True)
class Quote:
    ticker: str
    price: float | None
    as_of: str  # "YYYY-MM-DD, HH:MM UTC"


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]
    portfolio_json: str
    held_tickers: list[str]
    route: object  # schemas.RouteDecision
    query: str
    passages: list
    graded: list
    rewrites: int
    quotes: dict[str, Quote]
    memo: object  # schemas.Memo
    trace: list[str]
