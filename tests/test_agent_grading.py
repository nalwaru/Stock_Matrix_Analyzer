import openai

from agent.nodes import (MAX_REWRITES, grade_node, retrieve_node, rewrite_node,
                         route_after_grade)
from agent.schemas import GradeResult, PassageGrade, RewriteResult, RouteDecision
from rag.retrieve import RetrievalError
from tests.agent_fakes import ScriptedClient, make_deps, passage


def route(name="sec_filings", tickers=("AAPL",)):
    return RouteDecision(route=name, tickers=list(tickers), reason="r")


def grades(**flags):
    return GradeResult(grades=[PassageGrade(id=k, relevant=v, reason="r") for k, v in flags.items()])


def test_retrieve_uses_question_then_rewritten_query():
    seen = []
    deps = make_deps(retrieve=lambda q, t: seen.append((q, t)) or [passage(1)])
    out = retrieve_node({"question": "orig", "route": route()}, deps)
    assert seen == [("orig", ["AAPL"])] and out["passages"][0].id == "id1"
    retrieve_node({"question": "orig", "query": "better", "route": route()}, deps)
    assert seen[-1][0] == "better"


def test_retrieval_error_gives_empty_passages():
    def boom(q, t):
        raise RetrievalError("down")

    out = retrieve_node({"question": "q", "route": route()}, make_deps(retrieve=boom))
    assert out["passages"] == [] and "unavailable" in out["trace"][0].lower()


def test_grade_keeps_only_relevant_passages():
    client = ScriptedClient(GradeResult=grades(id1=True, id2=False))
    state = {"question": "q", "passages": [passage(1), passage(2)]}
    assert [p.id for p in grade_node(state, make_deps(client))["graded"]] == ["id1"]


def test_grade_ignores_unknown_ids():
    client = ScriptedClient(GradeResult=grades(id1=False, ghost=True))
    out = grade_node({"question": "q", "passages": [passage(1)]}, make_deps(client))
    assert out["graded"] == []


def test_grader_failure_yields_no_graded_passages():
    client = ScriptedClient(GradeResult=openai.OpenAIError("x"))
    out = grade_node({"question": "q", "passages": [passage(1)]}, make_deps(client))
    assert out["graded"] == [] and "grader failed" in out["trace"][0].lower()


def test_grade_skips_model_call_when_no_passages():
    client = ScriptedClient()
    assert grade_node({"question": "q", "passages": []}, make_deps(client))["graded"] == []
    assert client.calls == []


def test_rewrite_increments_counter_and_sets_query():
    client = ScriptedClient(RewriteResult=RewriteResult(query="  sharper query "))
    out = rewrite_node({"question": "q", "query": "old", "rewrites": 1}, make_deps(client))
    assert out["rewrites"] == 2 and out["query"] == "sharper query"


def test_rewrite_failure_keeps_old_query_but_still_counts():
    client = ScriptedClient(RewriteResult=openai.OpenAIError("x"))
    out = rewrite_node({"question": "q", "query": "old"}, make_deps(client))
    assert out["rewrites"] == 1 and out["query"] == "old"


def test_route_after_grade_all_branches():
    some, none = [passage(1)], []
    assert route_after_grade({"route": route("sec_filings"), "graded": some}) == "synthesize"
    assert route_after_grade({"route": route("both"), "graded": some}) == "price"
    assert route_after_grade({"route": route(), "graded": none, "rewrites": 0}) == "rewrite"
    assert route_after_grade({"route": route(), "graded": none, "rewrites": MAX_REWRITES - 1}) == "rewrite"
    assert route_after_grade({"route": route(), "graded": none, "rewrites": MAX_REWRITES}) == "synthesize"
