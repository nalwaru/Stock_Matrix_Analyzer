"""Yahoo Finance adapter for the analysts. Never raises: failures give empty data."""
import time

import pandas as pd

TTL_SECONDS = 300
BENCHMARK = "SPY"


def _clean(value):
    return None if value is None or (not isinstance(value, str) and pd.isna(value)) else value


class YahooData:
    def __init__(self, yf=None):
        self._yf = yf
        self._cache: dict = {}

    def _ticker(self, symbol: str):
        if self._yf is None:
            import yfinance

            self._yf = yfinance
        return self._yf.Ticker(symbol)

    def _cached(self, key, fetch, default):
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < TTL_SECONDS:
            return hit[1]
        try:
            value = fetch()
        except Exception:  # noqa: BLE001 - offline or unofficial-API change
            return default
        if value is None:
            return default
        self._cache[key] = (time.monotonic(), value)
        return value

    def history(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        return self._cached(("history", ticker, period),
                            lambda: self._ticker(ticker).history(period=period), pd.DataFrame())

    def closes(self, ticker: str) -> pd.Series:
        frame = self.history(ticker, "3mo")
        return frame["Close"] if "Close" in frame else pd.Series(dtype=float)

    def info(self, ticker: str) -> dict:
        return self._cached(("info", ticker), lambda: dict(self._ticker(ticker).info), {})

    def upgrades(self, ticker: str) -> list[dict]:
        def fetch():
            df = self._ticker(ticker).upgrades_downgrades.sort_index(ascending=False).head(10)
            return [{"date": f"{idx:%Y-%m-%d}", "firm": _clean(r.get("Firm")), "from": _clean(r.get("FromGrade")),
                     "to": _clean(r.get("ToGrade")), "action": _clean(r.get("Action"))} for idx, r in df.iterrows()]
        return self._cached(("upgrades", ticker), fetch, [])

    def holders(self, ticker: str) -> list[dict]:
        def fetch():
            df = self._ticker(ticker).institutional_holders.head(10)
            return [{"holder": _clean(r.get("Holder")),
                     "pct": f"{r['pctHeld'] * 100:.2f}%" if _clean(r.get("pctHeld")) is not None else None,
                     "shares": _clean(r.get("Shares"))} for _, r in df.iterrows()]
        return self._cached(("holders", ticker), fetch, [])

    def insiders(self, ticker: str) -> list[dict]:
        def fetch():
            df = self._ticker(ticker).insider_transactions.head(10)
            return [{"date": _clean(r.get("Start Date")), "insider": _clean(r.get("Insider")),
                     "transaction": _clean(r.get("Transaction")) or _clean(r.get("Text")),
                     "shares": _clean(r.get("Shares")), "value": _clean(r.get("Value"))} for _, r in df.iterrows()]
        return self._cached(("insiders", ticker), fetch, [])

    def calendar(self, ticker: str) -> dict:
        return self._cached(("calendar", ticker), lambda: dict(self._ticker(ticker).calendar), {})
