"""Injected collaborators so every node can be tested with fakes."""
from collections.abc import Callable
from dataclasses import dataclass

from agent.state import Quote


@dataclass(frozen=True)
class Deps:
    client: object  # OpenAI client, or None to build one per call
    model: str
    retrieve: Callable[[str, list[str]], list]  # (query, tickers) -> list[Passage]
    fetch_quotes: Callable[[list[str]], dict[str, Quote]]
    indexed_tickers: list[str]
    holdings: object = None  # enriched holdings DataFrame for the portfolio calculator
    analyst_data: object = None  # YahooData-like adapter for the analysts
    filings_dir: object = None  # Path to downloaded filings for the filing tools
