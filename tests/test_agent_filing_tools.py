from pathlib import Path

from agent.nodes import tools_node
from agent.schemas import CalcCall, FilingCall, RouteDecision, ToolPlan
from agent.tools.filings import compare_periods, get_section, list_filings
from tests.agent_fakes import ScriptedClient, make_deps

HTML_K = """<html><body>
<p>Item 1A. Risk Factors</p><p>Supply risk is large.</p>
<p>Item 7. Management's Discussion</p><p>Sales grew.</p>
<table><caption>Net sales</caption><tr><th>Category</th><th>2025</th><th>2024</th></tr>
<tr><td>iPhone</td><td>209,586</td><td>201,183</td></tr>
<tr><td>Total net sales</td><td>416,161</td><td>391,035</td></tr></table>
</body></html>"""
HTML_Q = """<html><body><p>Item 2. Management's Discussion</p>
<table><tr><th>Item</th><th>Q3 2026</th><th>Q3 2025</th></tr>
<tr><td>Total net sales</td><td>109,417</td><td>94,036</td></tr></table></body></html>"""


def make_dir(tmp_path: Path) -> Path:
    base = tmp_path / "filings" / "AAPL"
    base.mkdir(parents=True)
    (base / "10-K-2025-10-31-0000320193-25-000079.htm").write_text(HTML_K)
    (base / "10-Q-2026-07-31-0000320193-26-000010.htm").write_text(HTML_Q)
    return tmp_path / "filings"


def test_get_section_returns_section_text_with_filing_label(tmp_path):
    fact = get_section("aapl", "10-K", "item 7", make_dir(tmp_path))
    assert fact.label == "AAPL 10-K, 2025-10-31, Item 7"
    assert "Sales grew." in fact.text and "209,586" in fact.text and "Supply risk" not in fact.text


def test_get_section_caps_length(tmp_path):
    base = tmp_path / "filings" / "AAPL"
    base.mkdir(parents=True)
    (base / "10-K-2025-10-31-x.htm").write_text("<p>Item 7. MD&A</p>" + "<p>" + "word " * 5000 + "</p>")
    fact = get_section("AAPL", "10-K", "Item 7", tmp_path / "filings")
    assert len(fact.text) <= 6100 and fact.text.endswith("[truncated]")


def test_get_section_missing_returns_none(tmp_path):
    d = make_dir(tmp_path)
    assert get_section("AAPL", "10-K", "Item 99", d) is None
    assert get_section("ZZZZ", "10-K", "Item 7", d) is None
    assert get_section("AAPL", "10-K", "Item 7", tmp_path / "missing") is None


def test_compare_periods_returns_header_and_row_from_each_filing_newest_first(tmp_path):
    facts = compare_periods("AAPL", "total net sales", make_dir(tmp_path))
    assert [f.label.split(", ")[0] for f in facts] == ["AAPL 10-Q", "AAPL 10-K"]
    assert "| 109,417 | 94,036 |" in facts[0].text and "Q3 2026" in facts[0].text
    assert "416,161" in facts[1].text


def test_compare_periods_no_match_is_empty(tmp_path):
    assert compare_periods("AAPL", "goodwill impairment", make_dir(tmp_path)) == []


def test_list_filings_lists_forms_and_dates(tmp_path):
    fact = list_filings("AAPL", make_dir(tmp_path), "2026-10-17")
    assert fact.label == "AAPL filings, 2026-10-17, index"
    assert "10-Q 2026-07-31" in fact.text and "10-K 2025-10-31" in fact.text


def call(tool, **kw):
    base = dict(form=None, section=None, row_label=None)
    return FilingCall(tool=tool, ticker="AAPL", **{**base, **kw})


def test_tools_node_runs_filing_calls_for_filings_routes(tmp_path):
    plan = ToolPlan(calc_calls=[], filing_calls=[
        call("compare_periods", row_label="Total net sales"), call("get_section", form="10-K", section="Item 7"),
        call("get_section", form="10-K", section="Item 99")])
    state = {"question": "q", "route": RouteDecision(route="sec_filings", tickers=["AAPL"], reason="r")}
    out = tools_node(state, make_deps(ScriptedClient(ToolPlan=plan), filings_dir=make_dir(tmp_path)))
    assert len(out["facts"]) == 3 and any("skipped 1" in line for line in out["trace"])


def test_tools_node_ignores_calc_calls_on_filings_routes(tmp_path):
    calc = CalcCall(tool="total", ticker=None, asset_class=None, metric=None, n=None, shock_pct=None)
    plan = ToolPlan(calc_calls=[calc], filing_calls=[])
    state = {"question": "q", "route": RouteDecision(route="sec_filings", tickers=["AAPL"], reason="r")}
    out = tools_node(state, make_deps(ScriptedClient(ToolPlan=plan), filings_dir=make_dir(tmp_path)))
    assert out.get("facts", []) == []


def test_tools_node_skips_planner_when_no_local_filings(tmp_path):
    client = ScriptedClient()
    state = {"question": "q", "route": RouteDecision(route="sec_filings", tickers=["AAPL"], reason="r")}
    assert tools_node(state, make_deps(client)).get("facts", []) == []
    assert tools_node(state, make_deps(client, filings_dir=tmp_path / "none")).get("facts", []) == []
    assert client.calls == []
