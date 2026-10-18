import numpy as np
import pandas as pd

from agent.analysts.data import YahooData
from agent.graph import run
from agent.nodes import analysis_node
from agent.schemas import Memo, RouteDecision
from tests.agent_fakes import ScriptedClient, make_deps


def hist(n=300):
    idx = pd.bdate_range(end="2026-10-16", periods=n)
    close = pd.Series(100 + np.arange(n, dtype=float), index=idx)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1000.0})


class FakeData:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def history(self, ticker, period="2y"):
        self.calls.append(("history", ticker, period))
        if self.fail:
            raise RuntimeError("boom")
        return hist()

    def info(self, ticker):
        return {"trailingPE": 30.0, "recommendationKey": "buy", "recommendationMean": 2.0,
                "numberOfAnalystOpinions": 10, "targetMeanPrice": 300.0, "currentPrice": 250.0}

    def upgrades(self, ticker): return []
    def holders(self, ticker): return []
    def insiders(self, ticker): return []
    def calendar(self, ticker): return {}


def state(analyses=(), tickers=("NVDA",), held=("AAPL", "NVDA")):
    route = RouteDecision(route="agent", tickers=list(tickers), reason="r")
    return {"question": "q", "route": route, "held_tickers": list(held), "analyses": list(analyses)}


def kinds(out):
    return {f.label.split(", ")[0].split(" ")[1] for f in out["facts"]}


def test_selected_analyses_only():
    out = analysis_node(state(["technical"]), make_deps(analyst_data=FakeData()))
    assert kinds(out) == {"technical"}


def test_default_runs_all_four_analyses():
    out = analysis_node(state([]), make_deps(analyst_data=FakeData()))
    assert kinds(out) == {"technical", "risk", "fundamental", "ownership"}


def test_held_tickers_used_and_capped_at_three():
    out = analysis_node(state(["fundamental"], tickers=[], held=["A", "B", "C", "D"]),
                        make_deps(analyst_data=FakeData()))
    assert {f.label.split(" ")[0] for f in out["facts"]} == {"A", "B", "C"}


def test_risk_fetches_benchmark_and_other_holdings():
    data = FakeData()
    analysis_node(state(["risk"]), make_deps(analyst_data=data))
    fetched = {c[1] for c in data.calls}
    assert {"NVDA", "SPY", "AAPL"} <= fetched


def test_analyst_failure_is_traced_not_raised():
    out = analysis_node(state(["technical"]), make_deps(analyst_data=FakeData(fail=True)))
    assert any("failed" in line.lower() for line in out["trace"])
    assert all("unavailable" in f.text for f in out["facts"])


def test_no_data_adapter_returns_trace_only():
    out = analysis_node(state(), make_deps())
    assert out.get("facts", []) == [] and "unavailable" in out["trace"][0].lower()


def test_graph_analysis_route_reaches_synthesizer_with_citable_facts():
    label = "NVDA technical, 2026-10-17, RSI14"
    deps = make_deps(analyst_data=FakeData())
    # Scripted router + synthesizer; the synthesizer cites a real fact label produced today.
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    label = f"NVDA technical, {today}, RSI14"
    from tests.agent_fakes import assistant, tool_call, with_tools
    analysts = tool_call("c1", "run_analysts", {"ticker": "NVDA", "kinds": ["technical"]})
    client = with_tools(ScriptedClient(
        RouteDecision=RouteDecision(route="agent", tickers=["NVDA"], reason="r"),
        Memo=Memo(answer=f"RSI is at an extreme [{label}].", key_figures=[], sources=[label], unverified=[])),
        [assistant(tool_calls=[analysts]), assistant("done")])
    state_out = run("Is NVDA overbought?", [], "{}", ["NVDA"], make_deps(client, analyst_data=FakeData()))
    assert state_out["memo"].answer.startswith("RSI is at an extreme")


def test_yahoo_adapter_never_raises_when_yfinance_fails():
    class Broken:
        def Ticker(self, t):
            raise RuntimeError("offline")

    data = YahooData(yf=Broken())
    assert data.history("AAPL").empty and data.info("AAPL") == {}
    assert data.upgrades("AAPL") == [] and data.holders("AAPL") == []
    assert data.insiders("AAPL") == [] and data.calendar("AAPL") == {}
    assert data.closes("AAPL").empty


def test_technical_analysis_returns_a_chart_other_analyses_do_not():
    out = analysis_node(state(["technical"]), make_deps(analyst_data=FakeData()))
    assert len(out["charts"]) == 1 and out["charts"][0].layout.title.text.startswith("NVDA")
    other = analysis_node(state(["fundamental", "ownership"]), make_deps(analyst_data=FakeData()))
    assert other.get("charts", []) == []


def test_no_chart_when_history_is_too_short():
    class Short(FakeData):
        def history(self, ticker, period="2y"):
            return hist(10)

    out = analysis_node(state(["technical"]), make_deps(analyst_data=Short()))
    assert out.get("charts", []) == [] and out["facts"]


def test_stream_collects_charts_from_the_analysis_node():
    from agent.graph import run_stream
    from agent.schemas import Memo, RouteDecision
    from tests.agent_fakes import ScriptedClient

    from tests.agent_fakes import assistant, tool_call, with_tools
    d = RouteDecision(route="agent", tickers=["NVDA"], reason="r")
    analysts = tool_call("c1", "run_analysts", {"ticker": "NVDA", "kinds": ["technical"]})
    client = with_tools(
        ScriptedClient(RouteDecision=d, Memo=Memo(answer="ok", key_figures=[], sources=[], unverified=[])),
        [assistant(tool_calls=[analysts]), assistant("done")])
    final = list(run_stream("Show the technical chart for NVDA", [], "{}", ["NVDA"], make_deps(client, analyst_data=FakeData())))[-1][1]
    assert len(final["charts"]) == 1
