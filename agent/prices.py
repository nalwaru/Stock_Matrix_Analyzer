"""Live quotes via the existing yfinance adapter, stamped with the time of the lookup."""
import math
from collections.abc import Callable
from datetime import datetime, timezone

from agent.state import Quote
from analyzer import market


def quotes_from_market(
    tickers: list[str], now: datetime | None = None, fetch: Callable | None = None
) -> dict[str, Quote]:
    """Never guesses: a ticker without a finite positive price gets price=None.

    Note: market.fetch_quotes is cached for 5 minutes, so the time is the lookup time and
    the underlying quote can be up to 5 minutes older."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d, %H:%M UTC")
    try:
        frame = (fetch or (lambda t: market.fetch_quotes(tuple(t))))(list(tickers))
    except Exception:  # noqa: BLE001 - offline or library failure
        return {t: Quote(t, None, stamp) for t in tickers}
    quotes = {}
    for ticker in tickers:
        try:
            price = float(frame.loc[ticker, "Price"])
        except (KeyError, TypeError, ValueError):
            price = math.nan
        quotes[ticker] = Quote(ticker, price if math.isfinite(price) and price > 0 else None, stamp)
    return quotes
