from agent.analysts.ownership import analyze_ownership

TODAY = "2026-10-17"
INFO = {"recommendationKey": "buy", "recommendationMean": 2.0, "numberOfAnalystOpinions": 40,
        "targetMeanPrice": 300.0, "targetHighPrice": 350.0, "targetLowPrice": 220.0, "currentPrice": 250.0}
UPGRADES = [{"date": f"2026-10-{d:02d}", "firm": f"Firm{d}", "from": "Hold", "to": "Buy", "action": "up"}
            for d in range(1, 8)]
HOLDERS = [{"holder": f"Fund{i}", "pct": 0.05 - i / 1000, "shares": 1_000_000 * (6 - i)} for i in range(7)]
INSIDERS = [{"insider": f"Exec{i}", "transaction": "Sale", "date": f"2026-09-{i + 1:02d}", "shares": 1000 * i}
            for i in range(7)]
CALENDAR = {"Earnings Date": ["2026-10-29"]}


def by_metric(facts):
    return {f.label.rsplit(", ", 1)[1]: f.text for f in facts}


def run(**kw):
    args = dict(info=INFO, upgrades=UPGRADES, holders=HOLDERS, insiders=INSIDERS, calendar=CALENDAR)
    return by_metric(analyze_ownership("AAPL", today=TODAY, **{**args, **kw}))


def test_labels_follow_citation_shape():
    facts = analyze_ownership("AAPL", INFO, UPGRADES, HOLDERS, INSIDERS, CALENDAR, TODAY)
    assert all(f.label.startswith("AAPL ownership, 2026-10-17, ") for f in facts)


def test_consensus_and_price_target_upside_are_computed():
    m = run()
    assert "buy" in m["CONSENSUS"] and "2.00" in m["CONSENSUS"] and "40" in m["CONSENSUS"]
    assert "300.00" in m["PRICE_TARGET"] and "220.00" in m["PRICE_TARGET"]
    assert "20.00% above the current price 250.00" in m["PRICE_TARGET"]


def test_price_target_below_price_says_below():
    m = run(info={**INFO, "targetMeanPrice": 200.0})
    assert "20.00% below the current price 250.00" in m["PRICE_TARGET"]


def test_lists_are_limited_to_five_entries():
    m = run()
    assert m["RATING_CHANGES"].count("Firm") == 5 and m["INSTITUTIONS"].count("Fund") == 5
    assert m["INSIDERS"].count("Exec") == 5


def test_next_earnings_date():
    assert "2026-10-29" in run()["NEXT_EARNINGS"]


def test_missing_inputs_are_unavailable():
    m = run(info={}, upgrades=[], holders=[], insiders=[], calendar={})
    for name in ("CONSENSUS", "PRICE_TARGET", "RATING_CHANGES", "INSTITUTIONS", "INSIDERS", "NEXT_EARNINGS"):
        assert "unavailable" in m[name], name
