# Deterministic Gate Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tool-calling agent and the tool planner with a plan emitted by the gate; plain Python validates it and runs the steps in parallel; the writer and guardrails are unchanged.

**Architecture:** The graph becomes `router -> execute -> synthesize` and `router -> refuse`. The gate returns a flat `Plan` (price flag, up to 3 calculator calls, a filings query, analyst kinds). A pure `finalize_plan` clips and validates it; an `execute` node runs the steps concurrently in a thread pool and merges their evidence into the same state fields the writer already reads (`graded`, `facts`, `quotes`, `charts`, `trace`, `no_filings`, `searched_filings`). Existing retrieve/grade/rewrite/analysis/price nodes, `run_calc` and all analysts are reused unchanged.

**Tech Stack:** Python 3.14, LangGraph, `openai` SDK, Pydantic, `concurrent.futures`, pytest, Streamlit.

**Spec:** [docs/superpowers/specs/2026-10-18-deterministic-gate-plan-design.md](../specs/2026-10-18-deterministic-gate-plan-design.md) (supersedes the agent-loop parts of [2026-10-18-unified-agent-design.md](../specs/2026-10-18-unified-agent-design.md)).

## Global Constraints

- Keep LangGraph; use the existing `openai` SDK only. Gate model: `typesafe/jev-router` via OpenRouter when `OPENROUTER_API_KEY` and `ROUTER_MODEL` are set (`Deps.router_client`/`router_model`, `extra_body={"provider": {"require_parameters": True}}`), otherwise `Deps.client`/`Deps.model`. Writer, grader, rewriter: `Deps.model`, `temperature=0`.
- Routes are exactly `answer` and `out_of_scope`. Plan fields are flat (no unions): `price: bool`, `portfolio: list[CalcCall]`, `filings_query: str` (empty means no filings search), `analysts: list[Literal["technical","fundamental","risk","ownership"]]`.
- Caps enforced in code: at most 3 calculator calls; analysts for at most 3 tickers (`MAX_ANALYSIS_TICKERS`); one filings search (inside it the grader/rewriter keep today's single rewrite).
- `technical` is dropped unless the question text matches the existing keyword rule (moved to `agent/plan.py`); when `wants_outlook` is true the plan always includes `fundamental`, `risk` and `ownership`.
- Ticker fallback when the plan needs tickers and the gate gave none: price uses all held tickers; analysts use the first 3 held tickers; filings use held tickers that are in `Deps.indexed_tickers`; if that leaves nothing the step is skipped. Deviation from the spec's "3 tickers for price": price keeps today's behaviour (all held when none named, named tickers uncapped) so a portfolio-wide price question still works.
- A failing step never loses the others: it adds an "unavailable" fact and a trace line. The gate failing returns `route="answer"` with an empty plan and trace "routing failed". An empty plan with `declined_part` set yields the "I don't know" reply plus the "Not answered" line; an empty plan with no `declined_part` still calls the writer with the portfolio JSON.
- Trace lines for steps: `Step: <name>(<compact args>) <seconds>s`. `Deps.on_step` is called from the main thread (Streamlit status widgets cannot be updated from worker threads), in completion order.
- Guardrails, memo format, outlook rules, filing-availability notes and the "I'm not a financial advisor" line stay exactly as they are.
- `uv run pytest -q` must be green at the end of every task (currently 308 tests).

## Review Focus

- The gate asks for `technical` without the question asking, or for an outlook with no analysts: code drops/adds as the rules say. (Task 1)
- One step raises (for example Yahoo down) while others succeed: the other evidence reaches the writer. (Task 3)
- The gate emits a calculator call with an unknown ticker or class, or more than 3 calls, or 5 analyst kinds: skipped/clipped, nothing crashes. (Tasks 1 and 3)
- A follow-up with no ticker ("and its risk?"): the gate prompt receives recent history; if tickers still come back empty the fallback rules apply. (Tasks 1 and 4)
- An unindexed ticker (ORCL) with a filings query: one "No SEC filings are available" note and no crash, and nothing at all when an analyst-based outlook was produced. (Task 4)

---

### Task 1: Plan schema and `finalize_plan`

**Files:**
- Modify: `agent/schemas.py`
- Create: `agent/plan.py`
- Test: `tests/test_agent_plan.py`

**Interfaces:**
- Produces: `class Plan(BaseModel)` with `price: bool = False`, `portfolio: list[CalcCall] = []`, `filings_query: str = ""`, `analysts: list[Analysis] = []`; `RouteDecision` gains `plan: Plan = Plan()` (existing fields and the current `Route` literal are unchanged in this task).
- Produces in `agent/plan.py`: `TECHNICAL_ASKED: re.Pattern` (the regex now in `agent/agent_tools.py`, same text), `MAX_CALC_CALLS = 3`, `OUTLOOK_ANALYSTS = ["fundamental", "risk", "ownership"]`, `finalize_plan(decision: RouteDecision, question: str) -> RouteDecision` (returns a copy with the plan cleaned; an `out_of_scope` decision gets an empty plan; tickers uppercased and de-duplicated; calculator calls clipped to 3; analysts de-duplicated in order, `technical` kept only if `TECHNICAL_ASKED` matches the question, and the outlook trio appended when `wants_outlook`), and `step_tickers(decision: RouteDecision, held: list[str], indexed: list[str]) -> dict[str, list[str]]` returning keys `price`, `analysts`, `filings` using the fallback rules in Global Constraints.

- [ ] **Step 1: Write failing tests** in `tests/test_agent_plan.py`: `test_out_of_scope_gets_empty_plan`; `test_calc_calls_are_clipped_to_three`; `test_technical_dropped_without_keyword` and `test_technical_kept_when_asked` (question "What is NVDA's RSI?"); `test_outlook_adds_fundamental_risk_ownership_once` (input analysts `["risk"]` gives `["risk","fundamental","ownership"]`); `test_tickers_uppercased_and_deduplicated`; `test_step_tickers_fall_back_to_held_and_indexed` (no tickers: price = all held, analysts = first 3 held, filings = held ∩ indexed); `test_step_tickers_use_named_tickers_and_cap_analysts_at_three`; `test_step_tickers_empty_when_nothing_resolves`.
- [ ] **Step 2: Run** `uv run pytest tests/test_agent_plan.py -q`; expected: import error (`agent.plan` missing).
- [ ] **Step 3: Implement** `Plan`, the `plan` field and `agent/plan.py` as specified; in `agent/agent_tools.py` import `TECHNICAL_ASKED` from `agent.plan` instead of defining it.
- [ ] **Step 4: Run** `uv run pytest -q`; expected: all pass.
- [ ] **Step 5: Commit** `git add agent tests && git commit -m "feat: plan schema and finalize_plan"`

### Task 2: Step functions

**Files:**
- Create: `agent/steps.py`
- Test: `tests/test_agent_steps.py`

**Interfaces:**
- Consumes: `retrieve_node`, `grade_node`, `rewrite_node`, `analysis_node`, `price_node`, `MAX_REWRITES` from `agent.nodes`; `run_calc`; `CalcCall`; `Fact`; `Deps`.
- Produces: `step_filings(query: str, tickers: list[str], question: str, deps: Deps) -> dict`; `step_portfolio(calls: list[CalcCall], today: str, deps: Deps) -> dict`; `step_analysts(ticker: str, kinds: list[str], held: list[str], deps: Deps) -> dict`; `step_price(tickers: list[str], deps: Deps) -> dict`. Each returns a partial state update using the usual keys (`graded`, `searched_filings`, `no_filings`, `facts`, `charts`, `quotes`, `trace`) and never raises: an exception becomes `{"facts": [Fact("<step> tool, <today>, DATA", "<step> is unavailable")], "trace": ["<step> failed (<ExceptionName>)"]}`. The logic is the body of today's `_search_filings`, `_portfolio_calc`, `_run_analysts` and `_get_price` in `agent/agent_tools.py`, minus pydantic argument parsing and the `observation` text; synthetic node states use `types.SimpleNamespace(tickers=[...])` as `route` so no route literal is needed.

- [ ] **Step 1: Write failing tests** in `tests/test_agent_steps.py` (reuse `make_deps`, `passage`, `frame`, `FakeData` helpers): `test_step_filings_returns_graded_passages`, `test_step_filings_rewrites_once_when_nothing_relevant`, `test_step_filings_unindexed_ticker_sets_no_filings_without_search`, `test_step_portfolio_skips_unknown_ticker_and_returns_known_fact` (calls for AAPL and "ZZZZ"), `test_step_portfolio_without_holdings_returns_trace_only`, `test_step_analysts_returns_facts_and_chart_for_technical`, `test_step_price_returns_quotes_and_history_facts`, `test_step_raising_dependency_becomes_unavailable_fact` (analyst data whose `history` raises inside `step_price` via `closes`, and a retrieve callable raising a plain `RuntimeError` inside `step_filings`).
- [ ] **Step 2: Run** `uv run pytest tests/test_agent_steps.py -q`; expected: import error.
- [ ] **Step 3: Implement** `agent/steps.py`; do not delete `agent_tools.py` yet.
- [ ] **Step 4: Run** `uv run pytest -q`; expected: all pass.
- [ ] **Step 5: Commit** `git add agent tests && git commit -m "feat: deterministic step functions"`

### Task 3: `execute_node`

**Files:**
- Create: `agent/execute.py`
- Test: `tests/test_agent_execute.py`

**Interfaces:**
- Consumes: Task 1 (`step_tickers`, plan fields on `state["route"].plan`) and Task 2 (step functions).
- Produces: `execute_node(state: AgentState, deps: Deps) -> dict` returning one merged update (`graded`, `facts`, `quotes`, `charts`, `trace`, `searched_filings`, `no_filings`). It builds one task per planned step (filings: one task; portfolio: one task for all calls; price: one task; analysts: one task per ticker), runs them in `ThreadPoolExecutor(max_workers=6)`, reports each as `Step: <name>(<json args>) <seconds>s` in the trace in plan order and through `deps.on_step` on the main thread as tasks complete; an empty plan returns `{"trace": ["Plan: no steps needed"]}`. Merge rules match today's: lists concatenate, `graded` de-duplicated by `id`, `quotes` merged, booleans OR-ed, `no_filings` de-duplicated.

- [ ] **Step 1: Write failing tests** in `tests/test_agent_execute.py`: `test_empty_plan_returns_trace_only`; `test_runs_every_planned_step_and_merges_evidence` (plan with price, one calc call, a filings query, analysts `["fundamental"]`; assert facts, quotes, graded all present); `test_steps_run_concurrently` (two fake dependencies that each block on a shared `threading.Barrier(2, timeout=2)`; passing proves overlap); `test_one_failing_step_keeps_other_evidence` (Review Focus 2: price fetch raises, calculator succeeds); `test_unknown_calc_ticker_is_skipped_and_known_one_kept` and `test_more_than_three_calc_calls_in_plan_are_clipped_before_running` (decision passed through `finalize_plan`); `test_on_step_called_on_main_thread_for_each_step` (record `threading.current_thread()`); `test_on_step_exception_is_ignored`; `test_trace_lines_have_step_prefix_and_duration`; `test_unindexed_filings_ticker_reported_in_no_filings`.
- [ ] **Step 2: Run** `uv run pytest tests/test_agent_execute.py -q`; expected: import error.
- [ ] **Step 3: Implement** `agent/execute.py` using `concurrent.futures.as_completed`; guard `on_step` with try/except like `agent_node._step`.
- [ ] **Step 4: Run** `uv run pytest -q`; expected: all pass.
- [ ] **Step 5: Commit** `git add agent tests && git commit -m "feat: parallel plan executor"`

### Task 4: Gate emits the plan; graph switches to `router -> execute -> synthesize`

**Files:**
- Modify: `agent/schemas.py` (`Route = Literal["answer", "out_of_scope"]`)
- Modify: `agent/nodes.py` (`ROUTER_PROMPT`, `NEXT_NODE`, `router_node`, `synthesize_node`)
- Modify: `agent/graph.py` (nodes `router`, `execute`, `synthesize`, `refuse`; `NODE_LABELS` gets `"execute": "Running the planned steps"` and drops `tools`, `price`, `agent`)
- Modify: `agent/eval/router_cases.jsonl` and `agent/eval/run_router_eval.py` (labels and plan checks, see below)
- Test: `tests/test_agent_router.py`, `tests/test_agent_graph.py`, `tests/test_agent_stream.py`, `tests/test_agent_synth.py`, `tests/test_agent_outlook.py`, `tests/test_agent_facts.py`, `tests/test_agent_llm.py`, `tests/test_router_eval.py`, `tests/test_agent_analysis_node.py` (migrate every use of the old routes `live_price`, `portfolio_only`, `agent`)

**Interfaces:**
- Consumes: `finalize_plan`, `step_tickers`, `execute_node`.
- Produces: `router_node(state, deps)` builds the gate messages from held tickers, the asset classes in `deps.holdings` (if any), `deps.indexed_tickers`, the last `ai.MAX_CHAT_TURNS` turns of `state["history"]` and the question; asks for `RouteDecision` (now including `plan`); returns `{"route": finalize_plan(decision, question), "tickers": [...], "trace": ["Router: ..."]}`; on `ai.AIError` returns `route="answer"` with an empty plan and the "routing failed" trace. `NEXT_NODE = {"answer": "execute", "out_of_scope": "refuse"}`. `ROUTER_PROMPT` carries the calculator menu text now in `PLANNER_PROMPT`, the rule "leave the plan empty when the portfolio JSON alone answers it", the outlook and partial-decline rules from today's prompt, and the technical-only-if-asked rule.
- Produces: `synthesize_node` treats a non-empty plan with no evidence (`graded`, `facts`, priced quotes all empty) as "I don't know" without a model call, an empty plan with `declined_part` set the same way, and an empty plan with no `declined_part` as "call the writer with the portfolio JSON"; every other behaviour (outlook, notes, flags) is unchanged. Remove the `route == "agent"` conditions in favour of `route == "answer"`.

- [ ] **Step 1: Write failing tests**:
  - Router: `test_router_returns_plan_through_finalize_plan` (scripted decision with 5 calc calls and `technical` for a non-technical question; assert clipped and dropped), `test_router_prompt_receives_holdings_classes_indexed_and_history`, `test_router_failure_gives_answer_route_with_empty_plan`, `test_out_of_scope_refuses_with_empty_plan`, `test_partial_decline_keeps_answer_route_and_declined_part`, `test_outlook_question_gets_outlook_analysts`, `test_follow_up_without_ticker_uses_history_in_prompt` (Review Focus 4: assert the history text is in the sent messages), `test_route_after_router_mapping`.
  - Graph (scripted `ScriptedClient` only, no tool client): `test_filings_plan_runs_execute_then_synthesize` (timings order `router, execute, synthesize`), `test_price_plan_calls_no_other_llm` (`client.calls == ["RouteDecision","Memo"]`), `test_portfolio_plan_has_two_model_calls` (`["RouteDecision","Memo"]`; the calculator runs in Python), `test_mixed_plan_combines_filings_price_and_holdings`, `test_empty_plan_with_declined_part_gives_idk_and_not_answered_line`, `test_empty_plan_without_decline_calls_writer_with_portfolio_json`, `test_out_of_scope_makes_no_other_call`, `test_unindexed_ticker_end_to_end_says_no_filings_are_available` (Review Focus 5), `test_unexpected_exception_returns_idk_memo` (price fetch raising a plain exception inside a step is caught by the step, so use a graph-level failure instead: a `Memo` call that raises `RuntimeError`).
  - Migrate the other listed test files to the new route names and plan fields; delete assertions that depend on removed nodes (`tools`, `price`, `agent` in timings).
  - Eval files: migrate `router_cases.jsonl` (`live_price` / `portfolio_only` / `agent` become `answer`, keep `out_of_scope`), add `expected_plan` objects for at least eight cases (price only, portfolio only with a `total` call, filings, filings plus fundamental analysts with `wants_outlook`, technical when asked, no technical when not asked, mixed holdings plus filings, partial decline); `run_router_eval.py` gets `plan_misses(cases, decide_plan)` which checks required fields by subset matching (required booleans/lists present, `filings_query` non-empty when expected, extra `price` allowed, `technical` without keyword is a miss) and prints them; `tests/test_router_eval.py` gets `test_plan_misses_uses_subset_matching`.
- [ ] **Step 2: Run** `uv run pytest -q`; expected: the new and migrated tests fail.
- [ ] **Step 3: Implement** the schema, prompt, router, graph and synthesize changes above. Leave `agent_node.py`, `agent_tools.py`, `tools_node`, `PLANNER_PROMPT`, `ToolPlan` in place (removed in Task 5).
- [ ] **Step 4: Run** `uv run pytest -q` then `uv run python -m agent.eval.run_router_eval` (needs keys); expected: tests pass; the eval prints route accuracy and any plan misses. If Jev misses, adjust the prompt; if it still misses, report it (per the spec, switching the gate model is the user's env choice).
- [ ] **Step 5: Commit** `git add agent tests && git commit -m "feat: gate emits the plan; router -> execute -> synthesize"`

### Task 5: Remove the agent loop and the planner

**Files:**
- Delete: `agent/agent_node.py`, `agent/agent_tools.py`, `tests/test_agent_node.py`, `tests/test_agent_tools.py`
- Modify: `agent/nodes.py` (remove `tools_node`, `PLANNER_PROMPT`, `MAX_TOOL_CALLS`, `ALL_ANALYSES` if unused), `agent/schemas.py` (remove `ToolPlan`), `agent/llm.py` and `tests/agent_fakes.py` (remove `chat_with_tools`, `ScriptedToolClient`, `tool_call`, `assistant`, `with_tools` and their tests in `tests/test_agent_llm.py`), `agent/state.py` (remove `step_limit_hit`, `agent_stopped_early`), `agent/memo.py`/`agent/nodes.py` (remove the step-limit and stopped-early notes and their tests in `tests/test_agent_outlook.py`)
- Test: whole suite

**Interfaces:**
- Consumes: Task 4 (nothing imports the removed modules afterwards).
- Produces: no references to `agent_node`, `agent_tools`, `tools_node`, `ToolPlan`, `chat_with_tools`; `grep -rn "agent_node\|agent_tools\|tools_node\|ToolPlan\|chat_with_tools\|step_limit_hit" agent tests app.py README.md` returns nothing.

- [ ] **Step 1: Run** the grep above; expected: lists the references to remove (it is the failing check).
- [ ] **Step 2: Remove** the files and references listed.
- [ ] **Step 3: Run** the grep again (expect no output) and `uv run pytest -q`; expected: all pass with a smaller total.
- [ ] **Step 4: Commit** `git add -A agent tests && git commit -m "refactor: remove the tool-calling agent and tool planner"`

### Task 6: Evals, latency, docs and live verification

**Files:**
- Modify: `agent/eval/agent_cases.jsonl` (`expect_tools` becomes `expect_steps` with values `filings`, `portfolio`, `analysts`, `price`), `agent/eval/run_agent_eval.py` (read `Step: <name>(` lines from the trace; map step names `step_filings` -> `filings`, etc.; `KNOWN_STEPS`), `agent/eval/profile_latency.py` (print step durations from `Step:` lines), `README.md` (agent graph section: gate plan, execute, no agent loop), `app.py` (skip streaming duplicates for `execute` the way it did for `agent`)
- Test: `tests/test_agent_eval_cases.py`

**Interfaces:**
- Produces: `KNOWN_STEPS = {"filings", "portfolio", "analysts", "price"}`; `score(case, state)` fails a case when an `expect_steps` entry is missing, an `expect_sections` entry is missing, or `forbid` text is present; the five existing cases keep their intent (Apple profit and growth: filings plus analysts and outlook; filings only; holdings plus filings; Tesla partial decline; "will AAPL hit 300" refusal).

- [ ] **Step 1: Write failing tests** in `tests/test_agent_eval_cases.py` for the renamed fields (`test_cases_file_is_valid` using `KNOWN_STEPS`, `test_score_reports_missing_step`).
- [ ] **Step 2: Run** `uv run pytest tests/test_agent_eval_cases.py -q`; expected: FAIL.
- [ ] **Step 3: Implement** the eval, profile, README and app changes.
- [ ] **Step 4: Verify live** (needs keys): `uv run pytest -q` (green); `uv run python -m agent.eval.run_router_eval` (route accuracy at or above 100% on the migrated cases and no plan misses); `uv run python -m agent.eval.run_agent_eval` (5/5); `PYTHONPATH=. uv run python -m agent.eval.profile_latency`. Expected: median total for filings, mixed and outlook questions clearly below the earlier 11-17 s; price and portfolio questions no slower than before (about 4 s and 6 s); Streamlit starts and `curl -s localhost:8501/_stcore/health` prints `ok`.
- [ ] **Step 5: Commit** `git add -A agent tests README.md app.py && git commit -m "feat: plan-based evals, latency profile and docs"`
