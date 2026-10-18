from agent.eval.run_router_eval import CASES_PATH, load_cases, route_accuracy

ROUTES = {"live_price", "portfolio_only", "agent", "out_of_scope"}


def test_route_accuracy_counts_hits_and_lists_misses():
    cases = [{"question": "a", "expected_route": "agent"}, {"question": "b", "expected_route": "live_price"}]
    out = route_accuracy(cases, lambda q: "agent")
    assert out["accuracy"] == 0.5 and out["misses"] == [("b", "live_price", "agent")]


def test_empty_cases_do_not_divide_by_zero():
    assert route_accuracy([], lambda q: "agent")["accuracy"] == 0.0


def test_every_route_has_at_least_two_cases():
    cases = load_cases(CASES_PATH)
    counts = {r: sum(c["expected_route"] == r for c in cases) for r in ROUTES}
    assert all(n >= 2 for n in counts.values()), counts
    assert {c["expected_route"] for c in cases} <= ROUTES


def test_ticker_misses_lists_wrong_or_missing_tickers():
    from agent.eval.run_router_eval import ticker_misses

    cases = [{"question": "a", "expected_route": "analysis", "expected_tickers": ["ORCL"]},
             {"question": "b", "expected_route": "analysis", "expected_tickers": ["AAPL"]},
             {"question": "c", "expected_route": "portfolio_only"}]
    out = ticker_misses(cases, lambda q: ["O"] if q == "a" else ["AAPL"])
    assert out == [("a", ["ORCL"], ["O"])]


def test_some_cases_pin_the_resolved_ticker_for_company_names_and_typos():
    cases = [c for c in load_cases(CASES_PATH) if c.get("expected_tickers")]
    assert any(c["expected_tickers"] == ["ORCL"] for c in cases)


def test_each_check_asks_the_gate_once_so_flaky_answers_cannot_contradict_the_report():
    from agent.eval.run_router_eval import flag_misses, ticker_misses
    calls = []

    def flaky_flag(q):
        calls.append(q)
        return len(calls) % 2 == 0  # first call False, second True

    cases = [{"question": "a", "expected_route": "agent", "expected_declined": True}]
    out = flag_misses(cases, "expected_declined", flaky_flag)
    assert len(calls) == 1 and out == [("a", True, False)]

    tcalls = []

    def flaky_tickers(q):
        tcalls.append(q)
        return ["X"] if len(tcalls) == 1 else ["AAPL"]

    out = ticker_misses([{"question": "b", "expected_route": "agent", "expected_tickers": ["AAPL"]}], flaky_tickers)
    assert len(tcalls) == 1 and out == [("b", ["AAPL"], ["X"])]
