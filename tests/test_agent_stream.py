from agent.graph import NODE_LABELS, build_graph, run, run_stream
from agent.schemas import GradeResult, Memo, PassageGrade, RouteDecision, ToolPlan
from analyzer import ai
from tests.agent_fakes import ScriptedClient, assistant, make_deps, passage, tool_call, with_tools

LABEL = "AAPL 10-K, 2025-10-31, Item 1A"


def sec_client():
    return with_tools(
        ScriptedClient(
            RouteDecision=RouteDecision(route="agent", tickers=["AAPL"], reason="r"),
            GradeResult=GradeResult(grades=[PassageGrade(id="id1", relevant=True, reason="r")]),
            Memo=Memo(answer=f"Risk exists [{LABEL}].", key_figures=[], sources=[LABEL], unverified=[])),
        [assistant(tool_calls=[tool_call("c1", "search_filings", {"query": "risks", "tickers": ["AAPL"]})]),
         assistant("done")])


def test_stream_yields_each_node_in_order_then_done_with_the_final_state():
    client = sec_client()
    deps = make_deps(client, retrieve=lambda q, t: [passage(1)])
    events = list(run_stream("q", [], "{}", ["AAPL"], deps))
    nodes = [name for name, _ in events]
    assert nodes[0] == "router" and nodes[-1] == "done"
    assert nodes.index("agent") < nodes.index("synthesize")
    final = events[-1][1]
    assert final["memo"].answer.startswith("Risk exists") and any("Agent tool" in t for t in final["trace"])


def test_stream_final_state_matches_run():
    def deps_and_client():
        client = ScriptedClient(RouteDecision=RouteDecision(route="out_of_scope", tickers=[], reason="r"))
        return make_deps(client)

    streamed = list(run_stream("q", [], "{}", [], deps_and_client()))[-1][1]
    direct = run("q", [], "{}", [], deps_and_client())
    assert streamed["memo"] == direct["memo"] and streamed["trace"] == direct["trace"]


def test_stream_never_raises_and_ends_with_idk_memo():
    def boom(t):
        raise RuntimeError("unexpected")

    client = ScriptedClient(RouteDecision=RouteDecision(route="live_price", tickers=["AAPL"], reason="r"))
    events = list(run_stream("q", [], "{}", ["AAPL"], make_deps(client, quotes=boom)))
    assert events[-1][0] == "done" and events[-1][1]["memo"].answer == ai.IDK_MESSAGE
    assert any("failed" in line.lower() for line in events[-1][1]["trace"])


def test_every_graph_node_has_a_friendly_label():
    nodes = set(build_graph(make_deps(ScriptedClient())).get_graph().nodes) - {"__start__", "__end__"}
    assert nodes <= set(NODE_LABELS), nodes - set(NODE_LABELS)
    assert all(label and label[0].isupper() for label in NODE_LABELS.values())
