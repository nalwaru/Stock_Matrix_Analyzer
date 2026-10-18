"""Fundamental ratios from Yahoo's `info` dict. Missing values are 'unavailable', never zero."""
import math

from agent.state import Fact


def _num(info: dict, key: str) -> float | None:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


# (label, info key, description, formatter)
_RATIO = lambda v: f"{v:.2f}"  # noqa: E731
_PCT = lambda v: f"{v * 100:.2f}%"  # noqa: E731
_MONEY = lambda v: f"{v:,.0f}"  # noqa: E731
METRICS = [
    ("PE", "trailingPE", "trailing P/E", _RATIO),
    ("FORWARD_PE", "forwardPE", "forward P/E", _RATIO),
    ("PS", "priceToSalesTrailing12Months", "price/sales (trailing 12 months)", _RATIO),
    ("EV_EBITDA", "enterpriseToEbitda", "EV/EBITDA", _RATIO),
    ("PEG", "trailingPegRatio", "PEG ratio", _RATIO),
    ("GROSS_MARGIN", "grossMargins", "gross margin", _PCT),
    ("OPERATING_MARGIN", "operatingMargins", "operating margin", _PCT),
    ("NET_MARGIN", "profitMargins", "net profit margin", _PCT),
    ("REVENUE_GROWTH", "revenueGrowth", "revenue growth (year over year)", _PCT),
    ("EARNINGS_GROWTH", "earningsGrowth", "earnings growth (year over year)", _PCT),
    ("DEBT_TO_EQUITY", "debtToEquity", "debt-to-equity", lambda v: f"{v:.2f} (as reported by Yahoo, in percent)"),
    ("CASH", "totalCash", "total cash (USD)", _MONEY),
    ("FREE_CASH_FLOW", "freeCashflow", "free cash flow (USD)", _MONEY),
    ("DIVIDEND_YIELD", "trailingAnnualDividendYield", "trailing dividend yield", _PCT),
]


def analyze_fundamental(ticker: str, info: dict, today: str) -> list[Fact]:
    ticker = ticker.upper()
    facts = []
    for name, key, description, fmt in METRICS:
        value = _num(info or {}, key)
        text = (f"{ticker} {description}: {fmt(value)}" if value is not None
                else f"{ticker} {description} is unavailable (not reported by Yahoo)")
        facts.append(Fact(f"{ticker} fundamental, {today}, {name}", text))
    return facts
