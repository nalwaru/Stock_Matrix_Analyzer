from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from analyzer import market

ROOT = Path(__file__).parent.parent


def fake_quotes(tickers):
    q = pd.DataFrame(
        {"Price": 100.0, "Prev_Close": 99.0, "Week52_High": 120.0, "Week52_Low": 80.0},
        index=pd.Index(tickers, name="Ticker"),
    )
    return q


def test_app_renders_sample_dataset(monkeypatch):
    monkeypatch.setattr(market, "fetch_quotes", fake_quotes)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.metric) == 0  # nothing until a dataset is chosen

    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert labels[:3] == ["Total Portfolio Value", "Total Unrealized P/L", "Daily P&L"]
    # 215.75 total shares in the sample x $100 fake price
    assert at.metric[0].value == "$21,575.00"


def test_app_falls_back_to_stale_prices_when_quotes_are_empty(monkeypatch):
    monkeypatch.setattr(market, "fetch_quotes", lambda tickers: market.empty_quotes())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception
    assert any("stale" in e.value.lower() for e in at.error)
    assert at.metric[0].label == "Total Portfolio Value"
