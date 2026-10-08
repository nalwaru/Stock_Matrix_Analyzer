import math
from types import SimpleNamespace as NS

import pytest

from analyzer import market


class FakeYF:
    def __init__(self, infos, boom=False):
        self.infos = infos
        self.boom = boom

    def Tickers(self, symbols):
        if self.boom:
            raise RuntimeError("network down")
        return NS(tickers={k: NS(fast_info=v) for k, v in self.infos.items()})


def info(price=110.0, prev=108.0, high=150.0, low=90.0):
    return NS(last_price=price, previous_close=prev, year_high=high, year_low=low)


def test_returns_quotes_indexed_by_ticker():
    q = market.fetch_quotes_uncached(("AAPL",), FakeYF({"AAPL": info()}))
    assert q.index.name == "Ticker"
    assert list(q.columns) == market.QUOTE_COLUMNS
    assert q.loc["AAPL"].tolist() == [110.0, 108.0, 150.0, 90.0]


def test_unknown_ticker_becomes_nan_row():
    q = market.fetch_quotes_uncached(("AAPL", "ZZZZ"), FakeYF({"AAPL": info()}))
    assert math.isnan(q.loc["ZZZZ", "Price"])
    assert q.loc["AAPL", "Price"] == 110.0


def test_attribute_that_raises_or_is_none_becomes_nan():
    class Bad:
        @property
        def last_price(self):
            raise KeyError("boom")

        previous_close = None
        year_high = "n/a"
        year_low = float("inf")

    q = market.fetch_quotes_uncached(("AAPL", "MSFT"), FakeYF({"AAPL": info(), "MSFT": Bad()}))
    assert q.loc["MSFT"].isna().all()


def test_total_failure_raises_market_data_error():
    with pytest.raises(market.MarketDataError):
        market.fetch_quotes_uncached(("AAPL",), FakeYF({}, boom=True))


def test_no_prices_at_all_raises():
    with pytest.raises(market.MarketDataError):
        market.fetch_quotes_uncached(("ZZZZ",), FakeYF({}))


def test_empty_ticker_tuple_returns_empty_frame():
    q = market.fetch_quotes_uncached((), FakeYF({}))
    assert q.empty and list(q.columns) == market.QUOTE_COLUMNS


def test_fetch_quotes_is_streamlit_cached():
    assert hasattr(market.fetch_quotes, "clear")
