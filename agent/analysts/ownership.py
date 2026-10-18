"""Analyst consensus and ownership. Inputs are plain dicts/lists from the Yahoo adapter."""
from agent.analysts.fundamental import _num
from agent.state import Fact

LIST_LIMIT = 5


def _line(entry: dict, keys: list[str]) -> str:
    return ", ".join(f"{k}: {entry[k]}" for k in keys if entry.get(k) not in (None, ""))


def analyze_ownership(
    ticker: str, info: dict, upgrades: list[dict], holders: list[dict],
    insiders: list[dict], calendar: dict, today: str,
) -> list[Fact]:
    ticker = ticker.upper()
    info = info or {}
    facts: list[Fact] = []

    def add(name: str, text: str | None, why: str) -> None:
        facts.append(Fact(f"{ticker} ownership, {today}, {name}",
                          text or f"{ticker} {name} is unavailable ({why})"))

    mean, count, key = _num(info, "recommendationMean"), _num(info, "numberOfAnalystOpinions"), info.get("recommendationKey")
    add("CONSENSUS", key and mean is not None and
        f"{ticker} analyst consensus '{key}', mean rating {mean:.2f} (1 = strong buy, 5 = sell) "
        f"from {int(count) if count is not None else 'an unknown number of'} analysts", "no consensus from Yahoo")

    target, high, low = (_num(info, k) for k in ("targetMeanPrice", "targetHighPrice", "targetLowPrice"))
    price = _num(info, "currentPrice") or _num(info, "regularMarketPrice")
    gap = target is not None and price and (target / price - 1) * 100
    add("PRICE_TARGET", target is not None and price and
        f"{ticker} analyst price target mean {target:.2f} (low {low if low is None else f'{low:.2f}'}, "
        f"high {high if high is None else f'{high:.2f}'}); the mean target is {abs(gap):.2f}% "
        f"{'above' if gap >= 0 else 'below'} the current price {price:.2f}", "no price targets from Yahoo")

    add("RATING_CHANGES", upgrades and f"{ticker} latest rating changes: " + " | ".join(
        _line(u, ["date", "firm", "from", "to", "action"]) for u in upgrades[:LIST_LIMIT]), "no rating changes from Yahoo")
    add("INSTITUTIONS", holders and f"{ticker} top institutional holders: " + " | ".join(
        _line(h, ["holder", "pct", "shares"]) for h in holders[:LIST_LIMIT]), "no holder data from Yahoo")
    add("INSIDERS", insiders and f"{ticker} recent insider transactions: " + " | ".join(
        _line(i, ["date", "insider", "transaction", "shares", "value"]) for i in insiders[:LIST_LIMIT]),
        "no insider data from Yahoo")
    dates = (calendar or {}).get("Earnings Date")
    dates = [dates] if dates and not isinstance(dates, (list, tuple)) else (list(dates) if dates else [])
    add("NEXT_EARNINGS", dates and f"{ticker} next earnings date: " + " to ".join(str(d) for d in dates),
        "no earnings date from Yahoo")
    return facts
