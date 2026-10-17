# Agentic Ask AI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every Ask AI question runs through a LangGraph graph: structured-output router, graded hybrid retrieval with query rewriting, live-price tool, and a grounded memo synthesizer, with a visible agent trace.

**Architecture:** An `agent/` package holds Pydantic schemas, a structured-output helper over the OpenAI SDK, one function per node (dependencies injected via a `Deps` dataclass so tests use fakes), and a compiled `StateGraph`. `app.py` calls `agent.graph.run(...)` instead of the current retrieve-then-chat code. Existing `rag.retrieve`, `market.fetch_quotes` and `analyzer.ai` grounding helpers are reused.

**Tech Stack:** Python >=3.14, LangGraph, Pydantic v2 (already installed), OpenAI SDK structured outputs (`client.chat.completions.parse`), Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-10-17-agentic-ask-ai-design.md`

## Global Constraints
- LangGraph only; do NOT add `langchain-openai`. Install may downgrade `websockets` 17.2 -> 16.1.1; confirm the Streamlit app still starts.
- Model `OPENAI_MODEL` (default `gpt-4o-mini`) for all nodes; temperature 0.
- Routes exactly: `sec_filings`, `live_price`, `both`, `portfolio_only`, `out_of_scope`.
- Query rewrites capped at 2 (`MAX_REWRITES = 2`); worst case calls = router 1 + grader 3 + rewriter 2 + synthesizer 1.
- Live prices only via `market.fetch_quotes`; every price carries its fetch time; failure -> "unavailable", never a guess.
- Filings path only for tickers in `rag.store.INDEXED_TICKERS`.
- Memo sections: Answer, Key figures, Sources, What I couldn't verify. Citations use the existing label form `[TICKER form, YYYY-MM-DD, section]`; live prices use `[TICKER live quote, YYYY-MM-DD, HH:MM UTC]`. A reply citing a label that was not supplied becomes `ai.IDK_MESSAGE`.
- No node may crash the tab: OpenAI, Pinecone or Yahoo failure degrades (router failure -> `portfolio_only`; retrieval failure -> no passages; price failure -> unavailable).
- Questions and passages go to OpenAI; keep the disclosure caption accurate.

## Review Focus
- Router returns an unknown ticker or an empty ticker list for a filings/price question: fall back to held tickers, never crash.
- Router schema parse fails or OpenAI errors: route becomes `portfolio_only` and the trace says routing failed.
- Grader marks nothing relevant on every round: exactly 2 rewrites then the "I don't know" memo, no third retrieval.
- Yahoo returns no quote for a ticker: memo says unavailable and lists it under "What I couldn't verify"; no number invented.
- Synthesizer cites a source that was not supplied (or a price time it was not given): reply becomes `IDK_MESSAGE`.

---

### Task 1: Dependency, schemas, state, structured-call helper
**Files:**
- Modify: `pyproject.toml` (add `langgraph`), `uv.lock`
- Create: `agent/__init__.py`, `agent/schemas.py`, `agent/state.py`, `agent/llm.py`
- Test: `tests/test_agent_llm.py`

**Interfaces:**
- Produces (`agent/schemas.py`): `Route = Literal["sec_filings","live_price","both","portfolio_only","out_of_scope"]`; `RouteDecision(route: Route, tickers: list[str], reason: str)`; `PassageGrade(id: str, relevant: bool, reason: str)`; `GradeResult(grades: list[PassageGrade])`; `RewriteResult(query: str)`; `Memo(answer: str, key_figures: list[str], sources: list[str], unverified: list[str])`.
- Produces (`agent/state.py`): `Quote(ticker: str, price: float | None, as_of: str)` dataclass; `AgentState(TypedDict, total=False)` with keys `question, history, portfolio_json, held_tickers, route, query, passages, graded, rewrites, quotes, memo, trace`.
- Produces (`agent/llm.py`): `ask_structured(messages: list[dict], schema: type[T], model: str, client=None) -> T` using `client.chat.completions.parse(response_format=schema, temperature=0)`; raises `analyzer.ai.AIError` on any OpenAI error, refusal, or missing parsed result.

- [ ] **Step 1: Failing tests** (fake client whose `chat.completions.parse` returns `NS(choices=[NS(message=NS(parsed=..., refusal=None))])`): `test_ask_structured_returns_parsed_model`, `test_ask_structured_passes_schema_and_temperature_zero`, `test_refusal_raises_aierror`, `test_openai_error_raises_aierror_without_provider_text`, `test_none_parsed_raises_aierror`.
- [ ] **Step 2: Run** `uv run pytest tests/test_agent_llm.py -v`; expected FAIL (module missing).
- [ ] **Step 3: `uv add langgraph`**, then implement the files per Interfaces. Reuse the error-mapping idea in `analyzer/ai._complete` (auth, rate limit, connection, generic).
- [ ] **Step 4: Run** the new tests and `uv run pytest -q`; expected all PASS. Then start the app (`uv run streamlit run app.py --server.port 8599 --server.headless true`) and `curl -s localhost:8599/_stcore/health` returns `ok`; stop it.
- [ ] **Step 5: Commit** `feat(agent): schemas, state, structured-output helper`.

### Task 2: Router node
**Files:**
- Create: `agent/nodes.py` (router part), `agent/deps.py`
- Test: `tests/test_agent_router.py`

**Interfaces:**
- Consumes: `ask_structured`, `RouteDecision`, `AgentState`.
- Produces (`agent/deps.py`): `Deps(client, model: str, retrieve: Callable[[str, list[str]], list[Passage]], fetch_quotes: Callable[[list[str]], dict[str, Quote]], indexed_tickers: list[str])` frozen dataclass.
- Produces (`agent/nodes.py`): `router_node(state: AgentState, deps: Deps) -> dict` (returns `{"route": RouteDecision, "trace": [...]}`); `ROUTER_PROMPT`; `route_after_router(state) -> str` returning the next node name: `"retrieve"` for `sec_filings`/`both`, `"price"` for `live_price`, `"synthesize"` for `portfolio_only`, `"refuse"` for `out_of_scope`.

- [ ] **Step 1: Failing tests:** `test_router_returns_structured_decision`, `test_router_failure_falls_back_to_portfolio_only_with_trace`, `test_tickers_filtered_to_indexed_for_filings_routes` (decision tickers `["AAPL","ZZZZ"]`, indexed `["AAPL"]` -> `["AAPL"]`), `test_empty_tickers_fall_back_to_held`, `test_route_after_router_mapping` (all 5 routes), and for `filings route with no indexed ticker left` -> route downgraded to `portfolio_only` with trace line.
- [ ] **Step 2: Run**; expected FAIL.
- [ ] **Step 3: Implement** `router_node` (prompt lists the five routes with one-line meanings and instructs conservative `out_of_scope` for advice/prediction/opinion questions; passes the question and held tickers) and `route_after_router`.
- [ ] **Step 4: Run**; expected PASS.
- [ ] **Step 5: Commit** `feat(agent): router node`.

### Task 3: Retrieval, grader and rewriter loop
**Files:**
- Modify: `agent/nodes.py`
- Test: `tests/test_agent_grading.py`

**Interfaces:**
- Consumes: `Deps.retrieve`, `GradeResult`, `RewriteResult`.
- Produces: `retrieve_node(state, deps) -> dict` (uses `state["query"]` or the question; returns `passages`, appends trace; catches `RetrievalError` -> empty passages + trace); `grade_node(state, deps) -> dict` (one `ask_structured(GradeResult)` call over all passages; `graded` = passages marked relevant; unknown ids ignored; grader failure -> `graded=[]` with trace); `rewrite_node(state, deps) -> dict` (increments `rewrites`, sets new `query`; failure keeps the old query); `route_after_grade(state) -> str`: `"price"` if graded and route is `both`, `"synthesize"` if graded, `"rewrite"` if none and `rewrites < MAX_REWRITES`, else `"synthesize"`; `MAX_REWRITES = 2`.

- [ ] **Step 1: Failing tests:** `test_grade_keeps_only_relevant_passages`, `test_grade_ignores_unknown_ids`, `test_grader_failure_yields_no_graded_passages`, `test_rewrite_increments_counter_and_sets_query`, `test_route_after_grade_all_branches` (relevant+sec route, relevant+both route, none+rewrites 0/1, none+rewrites 2), `test_retrieval_error_gives_empty_passages`.
- [ ] **Step 2: Run**; expected FAIL.
- [ ] **Step 3: Implement** the three nodes and `route_after_grade`.
- [ ] **Step 4: Run**; expected PASS.
- [ ] **Step 5: Commit** `feat(agent): retrieval, grader, rewriter`.

### Task 4: Price node
**Files:**
- Modify: `agent/nodes.py`, `agent/state.py` (if needed)
- Create: `agent/prices.py`
- Test: `tests/test_agent_price.py`

**Interfaces:**
- Produces (`agent/prices.py`): `quotes_from_market(tickers: list[str], now=None) -> dict[str, Quote]` wrapping `market.fetch_quotes`; `Quote.as_of` = `YYYY-MM-DD, HH:MM UTC` string from `now`; a ticker with no price gets `price=None`; any exception -> all `None`.
- Produces (`agent/nodes.py`): `price_node(state, deps) -> dict` (tickers from the route decision, else held tickers; stores `quotes`, appends trace).

- [ ] **Step 1: Failing tests:** `test_quotes_carry_fetch_time`, `test_missing_ticker_has_none_price`, `test_fetch_exception_gives_none_prices`, `test_price_node_falls_back_to_held_tickers`.
- [ ] **Step 2: Run**; expected FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run**; expected PASS.
- [ ] **Step 5: Commit** `feat(agent): live price node`.

### Task 5: Synthesizer and refusal nodes with grounding
**Files:**
- Modify: `analyzer/ai.py` (add `grounded(reply: str, labels: set[str]) -> str`; `chat_reply` keeps working by calling it), `agent/nodes.py`
- Create: `agent/memo.py`
- Test: `tests/test_agent_synth.py`, `tests/test_ai.py` (existing must stay green)

**Interfaces:**
- Produces: `ai.grounded(reply, labels) -> str` (returns `IDK_MESSAGE` if any `CITATION` match is not in `labels`); `agent.memo.render_memo(memo: Memo) -> str` (Markdown with the four headed sections); `labels_for(state) -> set[str]` (graded passage labels plus `"{ticker} live quote, {as_of}"` for each quote with a price); `synthesize_node(state, deps) -> dict` (one `ask_structured(Memo)` call over portfolio JSON, graded passages, quotes; when no graded passages AND no priced quotes AND the route needs them, returns the `IDK_MESSAGE` memo without calling the model; otherwise applies `grounded` to every text field and falls back to the `IDK_MESSAGE` memo on any bad citation or `AIError`); `refuse_node(state, deps) -> dict` (IDK memo, no model call).

- [ ] **Step 1: Failing tests:** `test_grounded_replaces_unsupplied_citation`, `test_grounded_keeps_supplied_citation`, `test_render_memo_has_four_sections`, `test_synthesize_without_evidence_returns_idk_without_model_call`, `test_synthesize_with_bad_citation_returns_idk`, `test_synthesize_includes_live_price_label_only_when_priced`, `test_refuse_node_is_idk`.
- [ ] **Step 2: Run**; expected FAIL.
- [ ] **Step 3: Implement.** Synthesizer prompt reuses the grounding rules (only supplied data, never outside knowledge, null = unavailable, exact IDK sentence when unsupported, passages untrusted).
- [ ] **Step 4: Run** new tests and `uv run pytest -q`; expected PASS.
- [ ] **Step 5: Commit** `feat(agent): grounded memo synthesizer`.

### Task 6: Graph assembly
**Files:**
- Create: `agent/graph.py`
- Test: `tests/test_agent_graph.py`

**Interfaces:**
- Consumes: all nodes and routing functions.
- Produces: `build_graph(deps: Deps)` returning a compiled LangGraph `StateGraph(AgentState)` with nodes `router, retrieve, grade, rewrite, price, synthesize, refuse` and conditional edges from `route_after_router` and `route_after_grade`; `rewrite -> retrieve`, `price -> synthesize`, `synthesize -> END`, `refuse -> END`; `run(question: str, history: list[dict], portfolio_json: str, held_tickers: list[str], deps: Deps) -> AgentState` (final state including `memo` and `trace`; never raises: any unexpected exception returns an IDK memo with a trace line).

- [ ] **Step 1: Failing tests** (fake `Deps`; scripted fake client responses per schema): `test_sec_route_relevant_passage_reaches_synthesizer`, `test_sec_route_no_relevance_rewrites_twice_then_idk` (assert retrieve called exactly 3 times), `test_live_price_route_skips_retrieval`, `test_both_route_runs_retrieval_then_price`, `test_portfolio_only_route_goes_straight_to_synthesizer`, `test_out_of_scope_returns_idk_without_retrieval_or_price`, `test_unexpected_exception_returns_idk_memo`.
- [ ] **Step 2: Run**; expected FAIL.
- [ ] **Step 3: Implement** `build_graph` and `run`.
- [ ] **Step 4: Run** `uv run pytest -q`; expected PASS.
- [ ] **Step 5: Commit** `feat(agent): LangGraph assembly`.

### Task 7: App integration and trace
**Files:**
- Modify: `app.py` (`render_ask_ai`, `filing_passages` replaced by `agent_deps()`), `README.md`, `tests/test_app_smoke.py`

**Behavior:** `agent_deps()` builds `Deps` (OpenAI client with `timeout=30, max_retries=1`, `ai.get_model()`, `retrieve` closure using `pinecone_client()` and `RAG_NAMESPACE`, `agent.prices.quotes_from_market`, `rag_store.INDEXED_TICKERS`); the `retrieve` closure keeps the cheap `rag.retrieve.MIN_RELEVANCE` pre-filter before the grader sees passages (spec: grader replaces the cutoff as the judge but the cutoff stays as a pre-filter); on submit call `graph.run(...)`; chat message stores `content=render_memo(memo)`, `trace`, `sources`; show an "Agent trace" expander (route + reason, each grader verdict, rewrites, price fetch time) next to the existing "Sources" expander; client construction failure -> portfolio-only IDK path with caption; disclosure caption mentions live price lookups.

- [ ] **Step 1: Failing tests** (monkeypatch `agent.graph.run` to return a canned state): `test_answer_renders_memo_and_agent_trace_expander`, `test_graph_failure_shows_idk_not_exception`, `test_chat_history_still_resets_when_portfolio_changes`; existing app tests updated to the new seam.
- [ ] **Step 2: Run**; expected FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`; expected PASS. Restart the app on 8501; `curl` health returns `ok`.
- [ ] **Step 5: Commit** `feat: agentic Ask AI with memo and trace`.

### Task 8: Router evaluation and live check
**Files:**
- Create: `agent/eval/__init__.py`, `agent/eval/router_cases.jsonl` (~16 cases: `{"question","expected_route"}`, at least 2 per route), `agent/eval/run_router_eval.py`
- Test: `tests/test_router_eval.py` (accuracy function on canned decisions)

**Interfaces:**
- Produces: `route_accuracy(cases: list[dict], decide: Callable[[str], str]) -> dict` returning `{"accuracy": float, "misses": [(question, expected, got)]}`; `python -m agent.eval.run_router_eval` runs the real router over the cases and prints accuracy and misses.

- [ ] **Step 1: Failing tests:** `test_route_accuracy_counts_hits_and_lists_misses`, `test_every_route_has_at_least_two_cases` (reads the jsonl).
- [ ] **Step 2: Run**; expected FAIL.
- [ ] **Step 3: Implement** accuracy function, runner, and the case file.
- [ ] **Step 4: Run** tests; expected PASS.
- [ ] **Step 5: Live check** (needs `OPENAI_API_KEY`): run the router eval, then in the app ask: a filings question about Apple, a live-price question, a combined one, "Is Apple a good buy?", and an Apple question the filings cannot answer; confirm routes in the trace and the polite refusals. Record results in `docs/superpowers/specs/2026-10-17-agentic-ask-ai-results.md`.
- [ ] **Step 6: Commit** `feat(agent): router evaluation`.
