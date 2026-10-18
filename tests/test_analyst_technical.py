import pandas as pd

from agent.analysts.technical import analyze_technical

TODAY = "2026-10-17"


def history(n=300, start=100.0, step=1.0, volume=1000.0, band=1.0):
    idx = pd.bdate_range(end="2026-10-16", periods=n)
    close = pd.Series([start + step * i for i in range(n)], index=idx)
    return pd.DataFrame({"Open": close, "High": close + band, "Low": close - band,
                         "Close": close, "Volume": volume})


def by_metric(facts):
    return {f.label.rsplit(", ", 1)[1]: f.text for f in facts}


def test_labels_follow_citation_shape():
    for fact in analyze_technical("AAPL", history(), TODAY):
        assert fact.label.startswith("AAPL technical, 2026-10-17, ")


def test_uptrend_gives_rsi_100_and_exact_smas():
    m = by_metric(analyze_technical("AAPL", history(), TODAY))
    assert "100.00" in m["RSI14"]
    # closes 100..399; last 20 closes are 380..399 -> mean 389.50
    assert "389.50" in m["SMA20"] and "399.00" in m["SMA20"]
    assert "above" in m["SMA50_vs_SMA200"]


def test_atr_volume_ratio_and_range_position():
    h = history()
    h.loc[h.index[-1], "Volume"] = 3000.0
    m = by_metric(analyze_technical("AAPL", h, TODAY))
    assert "2.00" in m["ATR14"]
    assert "3.00x" in m["VOLUME"]
    assert "100.00%" in m["RANGE52W"]
    assert "line +" in m["MACD"] and "histogram" in m["MACD"]  # steady uptrend: EMA12 above EMA26


def test_support_and_resistance_use_recent_60_sessions():
    m = by_metric(analyze_technical("AAPL", history(), TODAY))
    # last 60 lows: closes 340..399 minus band 1 -> 339.00; highs max 400.00
    assert "339.00" in m["SUPPORT_RESISTANCE"] and "400.00" in m["SUPPORT_RESISTANCE"]


def test_flat_series_marks_rsi_and_bollinger_unavailable_not_zero():
    m = by_metric(analyze_technical("AAPL", history(step=0.0, band=0.0), TODAY))
    assert "unavailable" in m["RSI14"] and "unavailable" in m["BOLLINGER"]
    assert "100.00" in m["SMA50"]


def test_short_history_marks_long_windows_unavailable_without_crashing():
    m = by_metric(analyze_technical("AAPL", history(n=10), TODAY))
    for name in ("SMA20", "SMA50", "SMA200", "RSI14", "MACD", "RANGE52W"):
        assert "unavailable" in m[name], name


def test_missing_ohlv_columns_only_affect_dependent_metrics():
    h = history()[["Close"]]
    m = by_metric(analyze_technical("AAPL", h, TODAY))
    assert "unavailable" in m["ATR14"] and "unavailable" in m["VOLUME"]
    assert "100.00" in m["RSI14"]


def test_empty_history_returns_single_unavailable_fact():
    facts = analyze_technical("AAPL", pd.DataFrame(), TODAY)
    assert len(facts) == 1 and "unavailable" in facts[0].text
