"""Technical indicators from daily OHLCV. All values are computed with pandas, not the model."""
import pandas as pd

from agent.state import Fact


def _fact(ticker: str, today: str, name: str, text: str) -> Fact:
    return Fact(f"{ticker} technical, {today}, {name}", text)


def _wilder(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def analyze_technical(ticker: str, history: pd.DataFrame, today: str) -> list[Fact]:
    ticker = ticker.upper()
    if history is None or history.empty or "Close" not in history:
        return [_fact(ticker, today, "DATA", f"{ticker} technical analysis is unavailable (no price history)")]
    close = history["Close"].dropna()
    n = len(close)
    last = float(close.iloc[-1])
    out: list[Fact] = []

    def add(name: str, text: str | None, why: str) -> None:
        out.append(_fact(ticker, today, name, text or f"{ticker} {name} is unavailable ({why})"))

    smas = {}
    for w in (20, 50, 200):
        if n >= w:
            smas[w] = float(close.tail(w).mean())
            gap = (last / smas[w] - 1) * 100
            add(f"SMA{w}", f"{ticker} {w}-day SMA {smas[w]:.2f}; close {last:.2f} is "
                f"{abs(gap):.2f}% {'above' if gap >= 0 else 'below'} it", "")
        else:
            add(f"SMA{w}", None, f"needs {w} sessions, have {n}")
    for w in (20, 50):
        if n >= w:
            ema = float(close.ewm(span=w, adjust=False).mean().iloc[-1])
            add(f"EMA{w}", f"{ticker} {w}-day EMA {ema:.2f}", "")
        else:
            add(f"EMA{w}", None, f"needs {w} sessions, have {n}")
    if 50 in smas and 200 in smas:
        rel = "above" if smas[50] > smas[200] else "below"
        add("SMA50_vs_SMA200", f"{ticker} 50-day SMA {smas[50]:.2f} is {rel} the 200-day SMA {smas[200]:.2f}", "")
    else:
        add("SMA50_vs_SMA200", None, "needs 200 sessions")

    if n >= 15:
        delta = close.diff()
        avg_gain, avg_loss = _wilder(delta.clip(lower=0), 14).iloc[-1], _wilder(-delta.clip(upper=0), 14).iloc[-1]
        if avg_gain == 0 and avg_loss == 0:
            add("RSI14", None, "no price movement")
        else:
            rsi = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
            add("RSI14", f"{ticker} RSI(14) is {rsi:.2f}; by convention above 70 is overbought and below 30 oversold", "")
    else:
        add("RSI14", None, f"needs 15 sessions, have {n}")

    if n >= 35:
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        signal = macd.ewm(span=9, adjust=False).mean()
        add("MACD", f"{ticker} MACD(12,26,9) line {macd.iloc[-1]:+.2f}, signal {signal.iloc[-1]:+.2f}, "
            f"histogram {macd.iloc[-1] - signal.iloc[-1]:+.2f}", "")
    else:
        add("MACD", None, f"needs 35 sessions, have {n}")

    if n >= 20:
        mid, std = close.tail(20).mean(), close.tail(20).std(ddof=0)
        if std > 0:
            lower, upper = mid - 2 * std, mid + 2 * std
            add("BOLLINGER", f"{ticker} Bollinger(20,2) lower {lower:.2f}, middle {mid:.2f}, upper {upper:.2f}; "
                f"%b {(last - lower) / (upper - lower) * 100:.2f}%", "")
        else:
            add("BOLLINGER", None, "no price variation")
    else:
        add("BOLLINGER", None, f"needs 20 sessions, have {n}")

    if {"High", "Low"} <= set(history.columns) and n >= 15:
        prev = close.shift(1)
        tr = pd.concat([history["High"] - history["Low"], (history["High"] - prev).abs(),
                        (history["Low"] - prev).abs()], axis=1).max(axis=1)
        atr = float(_wilder(tr, 14).iloc[-1])
        add("ATR14", f"{ticker} ATR(14) is {atr:.2f} ({atr / last * 100:.2f}% of close)", "")
    else:
        add("ATR14", None, "needs High/Low and 15 sessions")

    if "Volume" in history and n >= 21:
        vol = history["Volume"].dropna()
        base = float(vol.iloc[-21:-1].mean())
        add("VOLUME", f"{ticker} latest volume {float(vol.iloc[-1]):,.0f} is {float(vol.iloc[-1]) / base:.2f}x "
            f"the prior 20-session average {base:,.0f}", "") if base > 0 else add("VOLUME", None, "no volume")
    else:
        add("VOLUME", None, "needs Volume and 21 sessions")

    if n >= 252:
        hi, lo = float(close.tail(252).max()), float(close.tail(252).min())
        add("RANGE52W", f"{ticker} 52-week range {lo:.2f} to {hi:.2f}; close is at "
            f"{(last - lo) / (hi - lo) * 100:.2f}% of the range", "") if hi > lo else add("RANGE52W", None, "no range")
    else:
        add("RANGE52W", None, f"needs 252 sessions, have {n}")

    if {"High", "Low"} <= set(history.columns) and n >= 60:
        add("SUPPORT_RESISTANCE", f"{ticker} 60-session low {float(history['Low'].tail(60).min()):.2f} (support) "
            f"and high {float(history['High'].tail(60).max()):.2f} (resistance)", "")
    else:
        add("SUPPORT_RESISTANCE", None, "needs High/Low and 60 sessions")
    return out
