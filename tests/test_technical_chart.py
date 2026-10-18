import numpy as np
import pandas as pd

from agent.analysts.technical import analyze_technical, indicator_frame
from agent.charts import technical_figure
from agent.memo import markdown_safe


def history(n=300, step=1.0, band=1.0, volume=1000.0):
    idx = pd.bdate_range(end="2026-10-16", periods=n)
    close = pd.Series(100 + step * np.arange(n, dtype=float), index=idx)
    return pd.DataFrame({"Open": close, "High": close + band, "Low": close - band,
                         "Close": close, "Volume": volume})


def test_indicator_frame_last_values_match_the_analyst_facts():
    h = history()
    f = indicator_frame(h)
    facts = {x.label.rsplit(", ", 1)[1]: x.text for x in analyze_technical("AAPL", h, "2026-10-17")}
    assert f"{f['SMA20'].iloc[-1]:.2f}" in facts["SMA20"]
    assert f"{f['SMA50'].iloc[-1]:.2f}" in facts["SMA50"]
    assert f"{f['SMA200'].iloc[-1]:.2f}" in facts["SMA200"]
    assert f"{f['RSI14'].iloc[-1]:.2f}" in facts["RSI14"]
    assert f"{f['BB_UP'].iloc[-1]:.2f}" in facts["BOLLINGER"]
    assert f"{f['MACD'].iloc[-1]:+.2f}" in facts["MACD"]


def test_indicator_frame_is_blank_where_history_is_too_short():
    f = indicator_frame(history(n=30))
    assert f["SMA200"].isna().all() and f["SMA20"].notna().iloc[-1] and f["RSI14"].notna().iloc[-1]


def trace_names(fig):
    return [t.name for t in fig.data]


def test_figure_has_price_averages_bands_volume_rsi_and_macd_panels():
    fig = technical_figure("AAPL", history())
    names = trace_names(fig)
    for expected in ("Close", "SMA20", "SMA50", "SMA200", "Bollinger band", "Volume", "RSI14",
                     "MACD", "Signal", "Histogram"):
        assert expected in names, expected
    assert fig.layout.title.text.startswith("AAPL")


def test_every_panel_has_its_own_single_axis_no_dual_axis():
    fig = technical_figure("AAPL", history())
    assert not any(getattr(fig.layout[k], "overlaying", None) for k in fig.layout if k.startswith("yaxis"))
    axes = {t.yaxis for t in fig.data if t.name in ("Close", "Volume", "RSI14", "MACD")}
    assert len(axes) == 4


def test_series_colors_are_fixed_per_series_and_distinct():
    fig = technical_figure("AAPL", history())
    colors = {t.name: t.line.color for t in fig.data if t.name in ("SMA20", "SMA50", "SMA200", "RSI14", "MACD", "Signal")}
    assert len(set(colors.values())) == 6


def test_rsi_reference_lines_and_limited_history():
    fig = technical_figure("AAPL", history())
    ys = {s.y0 for s in fig.layout.shapes if s.type == "line"}
    assert {30, 70} <= ys
    short = technical_figure("AAPL", history(n=30))
    assert "SMA200" not in trace_names(short)  # no empty series drawn


def test_too_little_or_missing_history_gives_no_figure():
    assert technical_figure("AAPL", history(n=10)) is None
    assert technical_figure("AAPL", pd.DataFrame()) is None


def test_markdown_safe_escapes_dollar_signs_once():
    assert markdown_safe("Price $343.46 and $5") == r"Price \$343.46 and \$5"
    assert markdown_safe(r"already \$ escaped") == r"already \$ escaped"
