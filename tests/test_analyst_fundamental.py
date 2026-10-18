from agent.analysts.fundamental import analyze_fundamental

TODAY = "2026-10-17"
INFO = {
    "trailingPE": 31.256, "forwardPE": 27.9, "priceToSalesTrailing12Months": 8.1,
    "enterpriseToEbitda": 22.4, "trailingPegRatio": 2.3, "grossMargins": 0.4623,
    "operatingMargins": 0.3105, "profitMargins": 0.2631, "revenueGrowth": 0.061,
    "earningsGrowth": 0.12, "debtToEquity": 154.5, "totalCash": 55_000_000_000,
    "freeCashflow": 98_500_000_000, "trailingAnnualDividendYield": 0.0041,
}


def by_metric(facts):
    return {f.label.rsplit(", ", 1)[1]: f.text for f in facts}


def test_labels_follow_citation_shape():
    for fact in analyze_fundamental("aapl", INFO, TODAY):
        assert fact.label.startswith("AAPL fundamental, 2026-10-17, ")


def test_ratios_and_percentages_are_formatted_exactly():
    m = by_metric(analyze_fundamental("AAPL", INFO, TODAY))
    assert "31.26" in m["PE"] and "27.90" in m["FORWARD_PE"] and "22.40" in m["EV_EBITDA"]
    assert "46.23%" in m["GROSS_MARGIN"] and "31.05%" in m["OPERATING_MARGIN"] and "26.31%" in m["NET_MARGIN"]
    assert "6.10%" in m["REVENUE_GROWTH"] and "12.00%" in m["EARNINGS_GROWTH"]
    assert "0.41%" in m["DIVIDEND_YIELD"]
    assert "55,000,000,000" in m["CASH"] and "98,500,000,000" in m["FREE_CASH_FLOW"]
    assert "154.50" in m["DEBT_TO_EQUITY"] and "as reported by Yahoo" in m["DEBT_TO_EQUITY"]


def test_missing_none_or_nan_values_are_unavailable_not_zero():
    m = by_metric(analyze_fundamental("AAPL", {"trailingPE": None, "forwardPE": float("nan")}, TODAY))
    assert "unavailable" in m["PE"] and "unavailable" in m["FORWARD_PE"] and "unavailable" in m["PS"]


def test_empty_info_returns_all_metrics_unavailable():
    facts = analyze_fundamental("AAPL", {}, TODAY)
    assert len(facts) == 14 and all("unavailable" in f.text for f in facts)
