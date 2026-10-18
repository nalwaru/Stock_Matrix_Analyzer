import openai
import pandas as pd

from agent.nodes import tools_node
from agent.schemas import CalcCall, RouteDecision, ToolPlan
from agent.tools.portfolio import run_calc
from tests.agent_fakes import ScriptedClient, make_deps

TODAY = "2026-10-17"


def frame():
    rows = [
        ("AAPL", "Stocks", 10, 100.0, 120.0, 1200.0, 200.0, 20.0, 10.0),
        ("NVDA", "Stocks", 5, 200.0, 180.0, 900.0, -100.0, -10.0, -5.0),
        ("BTC-USD", "Crypto", 1, 800.0, 900.0, 900.0, 100.0, 12.5, 0.0),
    ]
    cols = ["Ticker", "Asset_Class", "Shares", "Purchase_Price", "Price", "Current_Value",
            "Gain_Loss", "Gain_Loss_Pct", "Day_Change"]
    df = pd.DataFrame(rows, columns=cols)
    df["Cost_Basis"] = df["Shares"] * df["Purchase_Price"]
    df["Prev_Close"] = df["Price"]
    return df


def call(tool, **kw):
    base = dict(ticker=None, asset_class=None, metric=None, n=None, shock_pct=None, order=None)
    return CalcCall(tool=tool, **{**base, **kw})


def test_holding_reports_value_gain_and_weight():
    fact = run_calc(frame(), call("holding", ticker="aapl"), TODAY)
    assert fact.label == "PORTFOLIO calc, 2026-10-17, holding_AAPL"
    assert "1,200.00" in fact.text and "200.00" in fact.text and "40.00%" in fact.text


def test_class_weight():
    fact = run_calc(frame(), call("class_weight", asset_class="Crypto"), TODAY)
    assert "30.00%" in fact.text and "900.00" in fact.text and fact.label.endswith("weight_Crypto")


def test_top_n_orders_by_metric():
    fact = run_calc(frame(), call("top_n", metric="Gain_Loss", n=2), TODAY)
    assert fact.text.index("AAPL") < fact.text.index("BTC-USD") and "NVDA" not in fact.text


def test_top_n_smallest_returns_the_biggest_losers_first():
    fact = run_calc(frame(), call("top_n", metric="Gain_Loss", n=2, order="smallest"), TODAY)
    assert fact.label.endswith("bottom2_Gain_Loss")
    assert fact.text.index("NVDA") < fact.text.index("BTC-USD") and "AAPL" not in fact.text
    assert "-100.00" in fact.text


def test_total_uses_portfolio_summary():
    fact = run_calc(frame(), call("total"), TODAY)
    assert "3,000.00" in fact.text and fact.label.endswith("total")


def test_stress_applies_shock_to_asset_class():
    fact = run_calc(frame(), call("stress", asset_class="Crypto", shock_pct=-50.0), TODAY)
    assert "2,550.00" in fact.text and "-450.00" in fact.text and "-15.00%" in fact.text


def test_unknown_ticker_class_or_missing_args_return_none():
    assert run_calc(frame(), call("holding", ticker="ZZZZ"), TODAY) is None
    assert run_calc(frame(), call("class_weight", asset_class="Bonds"), TODAY) is None
    assert run_calc(frame(), call("holding"), TODAY) is None
    assert run_calc(frame(), call("stress", asset_class="Crypto"), TODAY) is None


def state(route="portfolio_only"):
    return {"question": "q", "route": RouteDecision(route=route, tickers=[], reason="r")}


def plan(calc=()):
    return ToolPlan(calc_calls=list(calc))


def test_tools_node_runs_calc_calls_and_adds_facts():
    client = ScriptedClient(ToolPlan=plan(calc=[call("total"), call("holding", ticker="NVDA")]))
    out = tools_node(state(), make_deps(client, holdings=frame()))
    assert [f.label.split(", ")[-1] for f in out["facts"]] == ["total", "holding_NVDA"]


def test_tools_node_caps_at_three_and_skips_bad_calls():
    calls = [call("holding", ticker="ZZZZ"), call("total"), call("total"), call("total"), call("total")]
    out = tools_node(state(), make_deps(ScriptedClient(ToolPlan=plan(calc=calls)), holdings=frame()))
    assert len(out["facts"]) == 2  # first 3 considered, the bad one skipped
    assert any("skipped" in line.lower() for line in out["trace"])


def test_planner_failure_yields_no_facts():
    client = ScriptedClient(ToolPlan=openai.OpenAIError("x"))
    out = tools_node(state(), make_deps(client, holdings=frame()))
    assert out.get("facts", []) == [] and "planner failed" in out["trace"][0].lower()


def test_tools_node_skips_planner_without_holdings_or_for_other_routes():
    client = ScriptedClient()
    assert tools_node(state(), make_deps(client)).get("facts", []) == []
    assert tools_node(state("live_price"), make_deps(client, holdings=frame())).get("facts", []) == []
    assert client.calls == []
