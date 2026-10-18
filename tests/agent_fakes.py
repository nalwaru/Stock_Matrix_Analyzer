"""Shared fakes for agent tests: a client that answers each structured schema from a script."""
from types import SimpleNamespace as NS

from agent.deps import Deps
from rag.retrieve import Passage


def passage(i, text="text", ticker="AAPL"):
    return Passage(f"id{i}", text, ticker, "10-K", "2025-10-31", "Item 1A", "http://x", 0.9)


class ScriptedClient:
    """parse() returns the next scripted value for response_format.__name__ (a list = a queue)."""

    def __init__(self, **script):
        self.script = {k: (list(v) if isinstance(v, list) else v) for k, v in script.items()}
        self.calls = []
        self.chat = NS(completions=NS(parse=self._parse))

    def _parse(self, **kwargs):
        name = kwargs["response_format"].__name__
        self.calls.append(name)
        value = self.script[name]
        if isinstance(value, list):
            value = value.pop(0)
        if isinstance(value, Exception):
            raise value
        return NS(choices=[NS(message=NS(parsed=value, refusal=None))])


def make_deps(client=None, retrieve=None, quotes=None, indexed=("AAPL",), holdings=None,
              analyst_data=None, filings_dir=None):
    return Deps(
        client=client, model="m",
        retrieve=retrieve or (lambda q, t: []),
        fetch_quotes=quotes or (lambda t: {}),
        indexed_tickers=list(indexed),
        holdings=holdings, analyst_data=analyst_data, filings_dir=filings_dir,
    )
