from agent.eval.run_agent_eval import CASES_PATH, KNOWN_SECTIONS, KNOWN_TOOLS, load_cases, score
from agent.schemas import Memo
from analyzer import ai


def final(memo, trace=()):
    return {"memo": memo, "trace": list(trace)}


def plain(**kw):
    return Memo(answer="Net income was strong.", key_figures=[], sources=[], unverified=[], **kw)


def test_cases_file_is_valid():
    cases = load_cases(CASES_PATH)
    assert len(cases) >= 5
    for case in cases:
        assert case["question"] and set(case["expect_tools"]) <= KNOWN_TOOLS
        assert set(case["expect_sections"]) <= KNOWN_SECTIONS and isinstance(case["forbid"], list)


def test_score_passes_when_tools_sections_and_text_match():
    case = {"question": "q", "expect_tools": ["search_filings"], "expect_sections": ["outlook"], "forbid": []}
    state = final(plain(outlook=["Cheap [AAPL fundamental, 2026-10-18, VALUATION]"], outlook_leaning="neutral"),
                  ['Agent tool: search_filings({"query": "x"}) 1.0s'])
    assert score(case, state) == []


def test_score_reports_missing_tool_missing_section_and_forbidden_text():
    case = {"question": "q", "expect_tools": ["run_analysts"], "expect_sections": ["declined"],
            "forbid": ["you should buy"]}
    memo = Memo(answer="You should buy it.", key_figures=[], sources=[], unverified=[])
    failures = score(case, final(memo, []))
    assert len(failures) == 3
    assert any("run_analysts" in f for f in failures) and any("declined" in f for f in failures)
    assert any("you should buy" in f for f in failures)


def test_score_idk_section():
    case = {"question": "q", "expect_tools": [], "expect_sections": ["idk"], "forbid": []}
    idk = Memo(answer=ai.IDK_MESSAGE, key_figures=[], sources=[], unverified=[])
    assert score(case, final(idk)) == [] and score(case, final(plain())) != []
