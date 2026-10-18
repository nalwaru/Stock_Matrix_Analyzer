from agent.graph import NODE_LABELS, build_graph, run, run_stream
from agent.schemas import GradeResult, Memo, PassageGrade, RouteDecision, ToolPlan
from analyzer import ai
from tests.agent_fakes import ScriptedClient, make_deps, passage

LABEL = "AAPL 10-K, 2025-10-31, Item 1A"


def sec_client():
    return ScriptedClient(
        RouteDecision=RouteDecision(route="sec_filings", tickers=["AAPL"], reason="r"),
        GradeResult=GradeResult(grades=[PassageGrade(id="id1", relevant=True, reason="r")]),
        Memo=Memo(answer=f"Risk exists [{LABEL}].", key_figures=[], sources=[LABEL], unverified=[]))


def test_stream_yields_each_node_in_order_then_done_with_the_final_state():
    client = sec_client()
    deps = make_deps(client, retrieve=lambda q, t: [passage(1)])
    events = list(run_stream("q", [], "{}", ["AAPL"], deps))
    nodes = [name for name, _ in events]
    assert nodes[0] == "router" and nodes[-1] == "done"
    assert nodes.index("retrieve") < nodes.index("grade") < nodes.index("synthesize")
    final = events[-1][1]
    assert final["memo"].answer.startswith("Risk exists") and any("Grader" in t for t in final["trace"])


def test_stream_final_state_matches_run():
    def deps_and_client():
        client = ScriptedClient(RouteDecision=RouteDecision(route="out_of_scope", tickers=[], reason="r"))
        return make_deps(client)

    streamed = list(run_stream("q", [], "{}", [], deps_and_client()))[-1][1]
    direct = run("q", [], "{}", [], deps_and_client())
    assert streamed["memo"] == direct["memo"] and streamed["trace"] == direct["trace"]


def test_stream_never_raises_and_ends_with_idk_memo():
    def boom(q, t):
        raise RuntimeError("unexpected")

    client = ScriptedClient(RouteDecision=RouteDecision(route="sec_filings", tickers=["AAPL"], reason="r"))
    events = list(run_stream("q", [], "{}", ["AAPL"], make_deps(client, retrieve=boom)))
    assert events[-1][0] == "done" and events[-1][1]["memo"].answer == ai.IDK_MESSAGE
    assert any("failed" in line.lower() for line in events[-1][1]["trace"])


def test_every_graph_node_has_a_friendly_label():
    nodes = set(build_graph(make_deps(ScriptedClient())).get_graph().nodes) - {"__start__", "__end__"}
    assert nodes <= set(NODE_LABELS), nodes - set(NODE_LABELS)
    assert all(label and label[0].isupper() for label in NODE_LABELS.values())
