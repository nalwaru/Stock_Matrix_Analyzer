"""Recent price changes from daily closes. Percentages are computed here, in Python."""
from collections.abc import Callable
from datetime import datetime, timezone

import pandas as pd

from agent.state import Fact

WINDOWS = (("1d", 1), ("1w", 5), ("1m", 21))  # trading sessions back from the latest close


def _yahoo_closes(ticker: str) -> pd.Series:
    import yfinance

    return yfinance.Ticker(ticker).history(period="3mo")["Close"]


def price_changes(
    tickers: list[str], now: datetime | None = None, fetch: Callable[[str], pd.Series] | None = None
) -> list[Fact]:
    day = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    fetch = fetch or _yahoo_closes
    facts: list[Fact] = []
    for ticker in tickers:
        try:
            closes = fetch(ticker).dropna()
        except Exception:  # noqa: BLE001 - offline or library failure
            closes = pd.Series(dtype=float)
        if closes.empty:
            facts.append(Fact(f"{ticker} price history, {day}, history",
                              f"{ticker} price history is unavailable"))
            continue
        last = float(closes.iloc[-1])
        for name, back in WINDOWS:
            if len(closes) > back:
                prev = float(closes.iloc[-1 - back])
                text = (f"{ticker} {name} change: {(last / prev - 1) * 100:+.2f}% "
                        f"(close {prev:.2f} on {closes.index[-1 - back]:%Y-%m-%d} to "
                        f"{last:.2f} on {closes.index[-1]:%Y-%m-%d})")
            else:
                text = f"{ticker} {name} change is unavailable (only {len(closes)} closes)"
            facts.append(Fact(f"{ticker} price history, {day}, {name}", text))
    return facts
