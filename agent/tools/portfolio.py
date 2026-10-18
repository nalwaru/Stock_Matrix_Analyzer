"""Deterministic portfolio calculator. Every figure is computed here, never by the model."""
import pandas as pd

from agent.schemas import CalcCall
from agent.state import Fact
from analyzer import portfolio, scenarios

_METRICS = {"Current_Value", "Gain_Loss", "Gain_Loss_Pct", "Day_Change"}


def _label(today: str, name: str) -> str:
    return f"PORTFOLIO calc, {today}, {name.replace(',', '').replace(']', '')}"


def _money(x: float) -> str:
    return f"{x:,.2f}"


def _total(df: pd.DataFrame) -> float:
    return float(df["Current_Value"].dropna().sum())


def run_calc(df: pd.DataFrame, call: CalcCall, today: str) -> Fact | None:
    """Run one calculation; None when arguments are missing or match nothing."""
    total = _total(df)
    if call.tool == "holding" and call.ticker:
        rows = df[df["Ticker"].str.upper() == call.ticker.strip().upper()]
        if rows.empty or pd.isna(rows.iloc[0]["Current_Value"]):
            return None
        r = rows.iloc[0]
        weight = r["Current_Value"] / total * 100 if total > 0 else 0.0
        text = (f"{r['Ticker']}: {r['Shares']:g} shares, price {_money(r['Price'])}, "
                f"value {_money(r['Current_Value'])}, gain/loss {_money(r['Gain_Loss'])} "
                f"({r['Gain_Loss_Pct']:.2f}%), {weight:.2f}% of the portfolio")
        return Fact(_label(today, f"holding_{r['Ticker']}"), text)
    if call.tool == "class_weight" and call.asset_class:
        rows = df[df["Asset_Class"].str.lower() == call.asset_class.strip().lower()]
        if rows.empty:
            return None
        name, value = rows.iloc[0]["Asset_Class"], _total(rows)
        weight = value / total * 100 if total > 0 else 0.0
        return Fact(_label(today, f"weight_{name}"),
                    f"{name}: value {_money(value)} = {weight:.2f}% of the portfolio ({len(rows)} holdings)")
    if call.tool == "top_n":
        metric = call.metric if call.metric in _METRICS else "Current_Value"
        n = max(1, min(call.n or 3, 10))
        top = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(n)
        if top.empty:
            return None
        listing = "; ".join(f"{t} {_money(v)}" for t, v in zip(top["Ticker"], top[metric]))
        return Fact(_label(today, f"top{n}_{metric}"), f"Top {n} by {metric}: {listing}")
    if call.tool == "total":
        s = portfolio.summary(df)
        return Fact(_label(today, "total"),
                    f"Total value {_money(s['total_value'])}, cost {_money(s['total_cost'])}, "
                    f"gain/loss {_money(s['total_gain'])} ({s['total_gain_pct']:.2f}%), "
                    f"day P&L {_money(s['day_pnl'])}, {s['positions']} positions")
    if call.tool == "stress" and call.asset_class and call.shock_pct is not None:
        rows = df[df["Asset_Class"].str.lower() == call.asset_class.strip().lower()]
        if rows.empty:
            return None
        name = rows.iloc[0]["Asset_Class"]
        cmp = scenarios.compare(scenarios.apply_shocks(df, {name: call.shock_pct / 100}))
        return Fact(_label(today, f"stress_{name}"),
                    f"If {name} moves {call.shock_pct:+.1f}%: portfolio value {_money(cmp['original'])} -> "
                    f"{_money(cmp['simulated'])}, change {_money(cmp['delta'])} ({cmp['delta_pct']:.2f}%)")
    return None
