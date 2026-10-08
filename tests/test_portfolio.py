import math

import pandas as pd
import pytest

from analyzer import portfolio
from analyzer.market import QUOTE_COLUMNS, empty_quotes


def holdings(rows=None) -> pd.DataFrame:
    rows = rows or [
        ("AAPL", 10, 100.0, "Stocks"),
        ("BTC-USD", 0.5, 20000.0, "Crypto"),
    ]
    return pd.DataFrame(
        {
            "Ticker": [r[0] for r in rows],
            "Shares": [float(r[1]) for r in rows],
            "Purchase_Price": [r[2] for r in rows],
            "Purchase_Date": pd.to_datetime(["2023-01-01"] * len(rows)),
            "Asset_Class": [r[3] for r in rows],
        }
    )


def quotes(rows: dict) -> pd.DataFrame:
    q = pd.DataFrame.from_dict(rows, orient="index", columns=QUOTE_COLUMNS)
    q.index.name = "Ticker"
    return q


def test_enrich_computes_live_metrics():
    df = portfolio.enrich(holdings(), quotes({"AAPL": [110.0, 108.0, 150.0, 90.0]}))
    aapl = df[df["Ticker"] == "AAPL"].iloc[0]
    assert aapl["Price_Status"] == "live"
    assert aapl["Cost_Basis"] == 1000
    assert aapl["Current_Value"] == 1100
    assert aapl["Gain_Loss"] == 100
    assert aapl["Gain_Loss_Pct"] == pytest.approx(10.0)
    assert aapl["Day_Change"] == 20
    assert aapl["Day_Change_Pct"] == pytest.approx(20 / 1080 * 100)
    assert aapl["Week52_High"] == 150.0


def test_ticker_without_quote_is_unavailable_and_excluded_from_totals():
    df = portfolio.enrich(holdings(), quotes({"AAPL": [110.0, 108.0, 150.0, 90.0]}))
    btc = df[df["Ticker"] == "BTC-USD"].iloc[0]
    assert btc["Price_Status"] == "unavailable"
    assert math.isnan(btc["Current_Value"])
    s = portfolio.summary(df)
    assert s["total_value"] == 1100
    assert s["total_cost"] == 1000
    assert s["total_gain"] == 100
    assert s["total_gain_pct"] == pytest.approx(10.0)
    assert s["day_pnl"] == 20
    assert s["day_pnl_pct"] == pytest.approx(20 / 1080 * 100)
    assert s["positions"] == 1 and s["unpriced"] == 1


def test_fallback_to_cost_marks_everything_stale_with_zero_gain():
    df = portfolio.enrich(holdings(), empty_quotes(), fallback_to_cost=True)
    assert (df["Price_Status"] == "stale").all()
    assert df["Current_Value"].tolist() == [1000.0, 10000.0]
    s = portfolio.summary(df)
    assert s["total_gain"] == 0 and s["day_pnl"] == 0


def test_zero_cost_basis_gives_nan_percent_not_inf():
    h = holdings([("GIFT", 10, 0.0, "Stocks")])
    df = portfolio.enrich(h, quotes({"GIFT": [5.0, 5.0, 6.0, 4.0]}))
    assert math.isnan(df.loc[0, "Gain_Loss_Pct"])
    s = portfolio.summary(df)
    assert s["total_gain"] == 50 and s["total_gain_pct"] == 0.0


def test_empty_frame_summary_and_ai_summary_are_zero_not_errors():
    df = portfolio.enrich(holdings().iloc[0:0], empty_quotes())
    s = portfolio.summary(df)
    assert s["total_value"] == 0 and s["total_gain_pct"] == 0 and s["day_pnl_pct"] == 0
    a = portfolio.ai_summary(df)
    assert a["allocation_pct"] == {} and a["top_positions"] == []
    assert portfolio.allocation(df).empty and portfolio.top_holdings(df).empty


def test_all_prices_unavailable_summary_is_zero():
    df = portfolio.enrich(holdings(), empty_quotes())
    s = portfolio.summary(df)
    assert s["total_value"] == 0 and s["unpriced"] == 2


def test_allocation_and_top_holdings_sum_lots_of_same_ticker():
    h = holdings([("AAPL", 10, 100.0, "Stocks"), ("AAPL", 5, 120.0, "Stocks"), ("VOO", 2, 400.0, "ETFs")])
    df = portfolio.enrich(h, quotes({"AAPL": [100.0, 100.0, 1.0, 1.0], "VOO": [500.0, 500.0, 1.0, 1.0]}))
    alloc = portfolio.allocation(df)
    assert alloc["Asset_Class"].tolist() == ["Stocks", "ETFs"]
    assert alloc.set_index("Asset_Class")["Current_Value"].to_dict() == {"ETFs": 1000.0, "Stocks": 1500.0}
    top = portfolio.top_holdings(df, 1)
    assert top["Ticker"].tolist() == ["AAPL"] and top["Current_Value"].tolist() == [1500.0]


def test_ai_summary_shape_and_rounding():
    h = holdings([("AAPL", 10, 100.0, "Stocks"), ("VOO", 2, 400.0, "ETFs")])
    df = portfolio.enrich(h, quotes({"AAPL": [150.0, 150.0, 1.0, 1.0], "VOO": [500.0, 500.0, 1.0, 1.0]}))
    a = portfolio.ai_summary(df)
    assert a["total_value"] == 2500.0
    assert a["position_count"] == 2
    assert a["allocation_pct"] == {"Stocks": 60.0, "ETFs": 40.0}
    assert a["top_positions"][0] == {"ticker": "AAPL", "value": 1500.0, "weight_pct": 60.0}
    assert a["total_roi_pct"] == pytest.approx((2500 - 1800) / 1800 * 100, abs=0.01)
