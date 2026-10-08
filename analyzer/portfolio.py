"""Pure portfolio math: no Streamlit imports."""
import numpy as np
import pandas as pd

from analyzer.market import QUOTE_COLUMNS


def _pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Percent (x100); NaN where the denominator is not positive."""
    return numerator / denominator.where(denominator > 0) * 100


def _safe_pct(numerator: float, denominator: float) -> float:
    return float(numerator / denominator * 100) if denominator > 0 else 0.0


def enrich(
    holdings: pd.DataFrame, quotes: pd.DataFrame, fallback_to_cost: bool = False
) -> pd.DataFrame:
    df = holdings.merge(
        quotes.reindex(columns=QUOTE_COLUMNS),
        left_on="Ticker",
        right_index=True,
        how="left",
    )
    live = df["Price"].notna()
    df["Price_Status"] = np.where(live, "live", "stale" if fallback_to_cost else "unavailable")
    if fallback_to_cost:
        df["Price"] = df["Price"].fillna(df["Purchase_Price"])
        df["Prev_Close"] = df["Prev_Close"].fillna(df["Price"])

    df["Cost_Basis"] = df["Shares"] * df["Purchase_Price"]
    df["Current_Value"] = df["Shares"] * df["Price"]
    df["Gain_Loss"] = df["Current_Value"] - df["Cost_Basis"]
    df["Gain_Loss_Pct"] = _pct(df["Gain_Loss"], df["Cost_Basis"])
    previous_value = df["Shares"] * df["Prev_Close"]
    df["Day_Change"] = df["Current_Value"] - previous_value
    df["Day_Change_Pct"] = _pct(df["Day_Change"], previous_value)
    return df.reset_index(drop=True)


def summary(df: pd.DataFrame) -> dict:
    priced = df[df["Current_Value"].notna()]
    total_value = float(priced["Current_Value"].sum())
    total_cost = float(priced["Cost_Basis"].sum())
    total_gain = total_value - total_cost
    day_pnl = float(priced["Day_Change"].sum())
    return {
        "total_value": total_value,
        "total_cost": total_cost,
        "total_gain": total_gain,
        "total_gain_pct": _safe_pct(total_gain, total_cost),
        "day_pnl": day_pnl,
        "day_pnl_pct": _safe_pct(day_pnl, total_value - day_pnl),
        "positions": int(len(priced)),
        "unpriced": int(len(df) - len(priced)),
    }


def allocation(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.dropna(subset=["Current_Value"])
        .groupby("Asset_Class", as_index=False)["Current_Value"]
        .sum()
        .sort_values("Current_Value", ascending=False)
        .reset_index(drop=True)
    )


def top_holdings(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        df.dropna(subset=["Current_Value"])
        .groupby("Ticker", as_index=False)["Current_Value"]
        .sum()
        .nlargest(n, "Current_Value")
        .reset_index(drop=True)
    )


def ai_summary(df: pd.DataFrame) -> dict:
    """Aggregated, rounded payload: the only portfolio data sent to the LLM."""
    s = summary(df)
    total = s["total_value"]
    alloc = allocation(df)
    top = top_holdings(df, 3)
    return {
        "total_value": round(total, 2),
        "total_roi_pct": round(s["total_gain_pct"], 2),
        "position_count": s["positions"],
        "allocation_pct": {
            cls: round(value / total * 100, 2)
            for cls, value in zip(alloc["Asset_Class"], alloc["Current_Value"])
        }
        if total > 0
        else {},
        "top_positions": [
            {
                "ticker": ticker,
                "value": round(value, 2),
                "weight_pct": round(value / total * 100, 2),
            }
            for ticker, value in zip(top["Ticker"], top["Current_Value"])
        ]
        if total > 0
        else [],
    }
