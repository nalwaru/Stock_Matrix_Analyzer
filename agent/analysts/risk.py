"""Risk statistics from daily closes. All values are computed with pandas/numpy, not the model."""
import numpy as np
import pandas as pd

from agent.state import Fact

RISK_FREE = 0.04  # annual, an explicit assumption stated in the Sharpe fact
TRADING_DAYS = 252
MIN_RETURNS = 30
MIN_OVERLAP = 60
MAX_CORRELATIONS = 5
MIN_VOL = 1e-6  # below this a series has no meaningful variation (float noise)


def _returns(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty or "Close" not in frame:
        return pd.Series(dtype=float)
    return frame["Close"].dropna().pct_change().dropna()


def _drawdown(close: pd.Series) -> tuple[float, str, str]:
    peak = close.cummax()
    dd = close / peak - 1
    trough = dd.idxmin()
    return float(dd.min()), f"{close[:trough].idxmax():%Y-%m-%d}", f"{trough:%Y-%m-%d}"


def analyze_risk(
    ticker: str, history: pd.DataFrame, benchmark: pd.DataFrame,
    others: dict[str, pd.DataFrame], today: str,
) -> list[Fact]:
    ticker = ticker.upper()
    r = _returns(history)
    if r.empty:
        return [Fact(f"{ticker} risk, {today}, DATA", f"{ticker} risk analysis is unavailable (no price history)")]
    out: list[Fact] = []

    def add(name: str, text: str | None, why: str = "") -> None:
        out.append(Fact(f"{ticker} risk, {today}, {name}", text or f"{ticker} {name} is unavailable ({why})"))

    year = r.tail(TRADING_DAYS)
    vol = float(year.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(year) >= MIN_RETURNS else None
    if vol is None:
        add("VOLATILITY", None, f"needs {MIN_RETURNS} daily returns, have {len(year)}")
    else:
        add("VOLATILITY", f"{ticker} annualized volatility {vol * 100:.2f}% (last {len(year)} daily returns)")

    b = _returns(benchmark)
    joined = pd.concat([year, b], axis=1, join="inner").dropna().tail(TRADING_DAYS)
    if len(joined) >= MIN_OVERLAP and joined.iloc[:, 1].var() > 0:
        beta = float(joined.iloc[:, 0].cov(joined.iloc[:, 1]) / joined.iloc[:, 1].var())
        add("BETA", f"{ticker} beta vs benchmark {beta:.2f} (over {len(joined)} overlapping daily returns)")
    else:
        add("BETA", None, "needs benchmark history with 60 overlapping days")

    close = history["Close"].dropna()
    for name, sessions, need in (("MAXDD1Y", TRADING_DAYS, MIN_OVERLAP), ("MAXDD3Y", 3 * TRADING_DAYS, 3 * TRADING_DAYS - 20)):
        window = close.tail(sessions)
        if len(window) >= need:
            depth, peak, trough = _drawdown(window)
            add(name, f"{ticker} max drawdown {depth * 100:.2f}% (peak {peak}, trough {trough})")
        else:
            add(name, None, f"needs {need} sessions, have {len(window)}")

    if vol and vol > MIN_VOL:
        sharpe = (float(year.mean()) * TRADING_DAYS - RISK_FREE) / vol
        add("SHARPE", f"{ticker} Sharpe ratio {sharpe:.2f} (annualized mean return minus an assumed "
            f"{RISK_FREE * 100:.1f}% risk-free rate, divided by volatility)")
    else:
        add("SHARPE", None, "needs return variation")

    for other, frame in list((k, v) for k, v in others.items() if k.upper() != ticker)[:MAX_CORRELATIONS]:
        pair = pd.concat([year, _returns(frame)], axis=1, join="inner").dropna().tail(TRADING_DAYS)
        if len(pair) >= MIN_OVERLAP and pair.iloc[:, 0].std() > 0 and pair.iloc[:, 1].std() > 0:
            corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
            add(f"CORR_{other.upper()}", f"{ticker} daily-return correlation with {other.upper()} {corr:.2f}")
        else:
            add(f"CORR_{other.upper()}", None, "needs 60 overlapping days with variation")
    return out
