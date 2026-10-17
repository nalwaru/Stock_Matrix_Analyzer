from agent.eval.run_router_eval import CASES_PATH, load_cases, route_accuracy

ROUTES = {"sec_filings", "live_price", "both", "portfolio_only", "out_of_scope"}


def test_route_accuracy_counts_hits_and_lists_misses():
    cases = [{"question": "a", "expected_route": "both"}, {"question": "b", "expected_route": "live_price"}]
    out = route_accuracy(cases, lambda q: "both")
    assert out["accuracy"] == 0.5 and out["misses"] == [("b", "live_price", "both")]


def test_empty_cases_do_not_divide_by_zero():
    assert route_accuracy([], lambda q: "both")["accuracy"] == 0.0


def test_every_route_has_at_least_two_cases():
    cases = load_cases(CASES_PATH)
    counts = {r: sum(c["expected_route"] == r for c in cases) for r in ROUTES}
    assert all(n >= 2 for n in counts.values()), counts
    assert {c["expected_route"] for c in cases} <= ROUTES
