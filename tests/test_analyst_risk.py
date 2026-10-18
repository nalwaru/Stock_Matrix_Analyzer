import numpy as np
import pandas as pd

from agent.analysts.risk import RISK_FREE, analyze_risk

TODAY = "2026-10-17"


def frame(returns, start=100.0):
    idx = pd.bdate_range(end="2026-10-16", periods=len(returns))
    return pd.DataFrame({"Close": start * np.cumprod(1 + np.asarray(returns, dtype=float))}, index=idx)


def by_metric(facts):
    return {f.label.rsplit(", ", 1)[1]: f.text for f in facts}


def wave(n=120, amp=0.01):
    return [amp * (1 if i % 2 == 0 else -1) for i in range(n)]


def test_labels_follow_citation_shape():
    for fact in analyze_risk("AAPL", frame(wave()), frame(wave()), {}, TODAY):
        assert fact.label.startswith("AAPL risk, 2026-10-17, ")


def test_volatility_matches_independent_numpy_calculation():
    r = wave()
    expected = np.std(pd.Series(frame(r)["Close"]).pct_change().dropna(), ddof=1) * np.sqrt(252) * 100
    m = by_metric(analyze_risk("AAPL", frame(r), frame(r), {}, TODAY))
    assert f"{expected:.2f}%" in m["VOLATILITY"]


def test_beta_is_two_when_stock_moves_twice_the_benchmark():
    bench = [0.01 * np.sin(i / 3) for i in range(150)]
    stock = [2 * x for x in bench]
    m = by_metric(analyze_risk("AAPL", frame(stock), frame(bench), {}, TODAY))
    assert "2.00" in m["BETA"]


def test_identical_series_have_beta_and_correlation_one():
    r = [0.01 * np.sin(i / 3) for i in range(150)]
    m = by_metric(analyze_risk("AAPL", frame(r), frame(r), {"NVDA": frame(r)}, TODAY))
    assert "1.00" in m["BETA"] and "1.00" in m["CORR_NVDA"]


def test_max_drawdown_of_known_path():
    rise = (200 / 100) ** (1 / 50) - 1
    fall = (100 / 200) ** (1 / 49) - 1
    r = [rise] * 50 + [fall] * 49
    m = by_metric(analyze_risk("AAPL", frame(r), frame(wave(len(r))), {}, TODAY))
    assert "-50.00%" in m["MAXDD1Y"]
    assert "unavailable" in m["MAXDD3Y"]


def test_monotonic_rise_has_zero_drawdown_and_sharpe_unavailable_when_no_variation():
    r = [0.001] * 120
    m = by_metric(analyze_risk("AAPL", frame(r), frame(wave(120)), {}, TODAY))
    assert "0.00%" in m["MAXDD1Y"] and "unavailable" in m["SHARPE"]


def test_sharpe_states_risk_free_assumption():
    r = [0.001 + x for x in wave(200, 0.005)]
    m = by_metric(analyze_risk("AAPL", frame(r), frame(wave(200)), {}, TODAY))
    assert f"{RISK_FREE * 100:.1f}%" in m["SHARPE"] and "Sharpe" in m["SHARPE"]


def test_short_history_and_missing_benchmark_are_unavailable_not_errors():
    m = by_metric(analyze_risk("AAPL", frame(wave(20)), pd.DataFrame(), {"NVDA": frame(wave(20))}, TODAY))
    for name in ("VOLATILITY", "BETA", "SHARPE", "CORR_NVDA"):
        assert "unavailable" in m[name], name


def test_correlation_limited_to_five_others_and_skips_self():
    others = {t: frame(wave(120)) for t in ("AAPL", "A", "B", "C", "D", "E", "F")}
    m = by_metric(analyze_risk("AAPL", frame(wave(120)), frame(wave(120)), others, TODAY))
    assert sorted(k for k in m if k.startswith("CORR_")) == ["CORR_A", "CORR_B", "CORR_C", "CORR_D", "CORR_E"]


def test_empty_history_returns_single_unavailable_fact():
    facts = analyze_risk("AAPL", pd.DataFrame(), pd.DataFrame(), {}, TODAY)
    assert len(facts) == 1 and "unavailable" in facts[0].text
