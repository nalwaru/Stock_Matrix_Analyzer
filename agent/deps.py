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
