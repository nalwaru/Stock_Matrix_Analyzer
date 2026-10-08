"""yfinance adapter. Returns quotes as a DataFrame; never invents values."""
import math

import pandas as pd
import streamlit as st
import yfinance

QUOTE_COLUMNS = ["Price", "Prev_Close", "Week52_High", "Week52_Low"]
_FAST_INFO_ATTRS = {
    "Price": "last_price",
    "Prev_Close": "previous_close",
    "Week52_High": "year_high",
    "Week52_Low": "year_low",
}


class MarketDataError(RuntimeError):
    """Raised when quotes cannot be retrieved at all."""


def empty_quotes() -> pd.DataFrame:
    return pd.DataFrame(
        columns=QUOTE_COLUMNS, index=pd.Index([], name="Ticker"), dtype=float
    )


def _read(info, attr: str) -> float:
    try:
        value = float(getattr(info, attr))
    except (AttributeError, KeyError, TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) and value > 0 else math.nan


def fetch_quotes_uncached(tickers: tuple[str, ...], yf_module=yfinance) -> pd.DataFrame:
    if not tickers:
        return empty_quotes()
    try:
        handle = yf_module.Tickers(" ".join(tickers))
    except Exception as exc:  # network/library failure
        raise MarketDataError(f"yfinance request failed: {exc}") from exc

    rows = {}
    for symbol in tickers:
        try:
            info = handle.tickers[symbol].fast_info
        except Exception:
            rows[symbol] = [math.nan] * len(QUOTE_COLUMNS)
            continue
        rows[symbol] = [_read(info, _FAST_INFO_ATTRS[c]) for c in QUOTE_COLUMNS]

    quotes = pd.DataFrame.from_dict(rows, orient="index", columns=QUOTE_COLUMNS)
    quotes.index.name = "Ticker"
    if quotes["Price"].isna().all():
        raise MarketDataError("No prices were returned for any ticker.")
    return quotes


@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_quotes(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_quotes_uncached(tickers)
