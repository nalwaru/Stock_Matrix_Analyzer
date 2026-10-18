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
              analyst_data=None):
    return Deps(
        client=client, model="m",
        retrieve=retrieve or (lambda q, t: []),
        fetch_quotes=quotes or (lambda t: {}),
        indexed_tickers=list(indexed),
        holdings=holdings, analyst_data=analyst_data,
    )


def tool_call(call_id: str, name: str, args: dict):
    import json
    return NS(id=call_id, type="function", function=NS(name=name, arguments=json.dumps(args)))


def assistant(content=None, tool_calls=()):
    return NS(role="assistant", content=content, tool_calls=list(tool_calls) or None)


class ScriptedToolClient:
    """create() returns the next scripted assistant message (an Exception is raised instead)."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return NS(choices=[NS(message=turn)])


def with_tools(client, turns):
    """Give a ScriptedClient a create() that serves scripted tool-calling turns."""
    tool_client = ScriptedToolClient(turns)
    client.chat.completions.create = tool_client._create
    client.tool_client = tool_client
    return client
