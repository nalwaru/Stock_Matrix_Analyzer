# Deterministic gate plan (replace the tool-calling agent and the tool planner): Design

Date: 2026-10-18
Status: Draft for approval; not implemented.
Supersedes the agent-loop parts of: [unified agent design](2026-10-18-unified-agent-design.md). Everything else in that design (outlook section, partial decline, guardrails, Jev as front door) stays.

## Intent and why
Live profiling of the unified agent showed that the model-driven loop costs time and predictability without adding judgement the app needs. The agent path takes 11-17 s; the tool planner on the portfolio path is a separate LLM call; the agent once wasted a call on an invalid argument (`kinds: ["growth"]`). The tool set is small and the question types are predictable. The user chose a deterministic design: the gate (Jev) already reads every question, so it also emits a small plan saying which steps are needed; plain Python runs exactly those steps (in parallel where independent), then the writer produces the memo. No LLM chooses tools at run time.

## Goals
- The gate returns, in one structured output: `route` (`answer` or `out_of_scope`), `tickers`, `wants_outlook`, `declined_part`, and a `plan`:
  - `price: bool` (live quotes and price changes for the tickers);
  - `portfolio: list[CalcCall]` (exact calculator calls against the user's holdings; at most 3);
  - `filings: {query: str} | null` (search the SEC filings with this query for the tickers);
  - `analysts: list[fundamental|risk|ownership|technical]` (run for each ticker, at most 3 tickers).
- One `execute` node runs the plan in parallel (thread pool) and returns the same evidence the writer reads today: `graded` passages, `facts`, `quotes`, `charts`, `trace`, `no_filings`.
- Python validates and clips the plan before running it: calculator calls capped at 3 and validated against the holdings (unknown tickers/classes are skipped as today); filings only for indexed tickers (others recorded in `no_filings`); `technical` kept only if the question text asks for it (the existing keyword rule, enforced in code); when `wants_outlook` is true and `analysts` has no fundamental/risk/ownership entry, those three are added for each ticker.
- The old fixed paths become plans: price question = `{price: true}`; portfolio question = `{portfolio: [...]}`; filings question = `{filings: {...}}`; a mixed question = several fields set. No separate route per path.
- The writer, guardrails and memo format are unchanged: Answer, Key figures, Sources, What I couldn't verify, optional Outlook (leaning, analyst-fact citations, "I'm not a financial advisor" line), "Not answered" line, single filing-availability note, charts only when technical was asked.
- Faster: no agent turns and no planner call. Expected LLM calls per question: gate, writer, plus grader (and one rewriter) when filings are searched.
- The Agent trace shows `Step: <name>(<args>) <seconds>s` lines for every executed step, and `on_step` still streams them to the UI.

## Non-goals
- A tool-calling agent or any fallback loop (removed; revisit only if evals show the plan misses things).
- Adaptive follow-up steps that depend on earlier results.
- Changes to retrieval, grading thresholds, analyst formulas, the calculator, or the number checker.

## Anti-goals
- The gate emitting a plan that runs arbitrary or unvalidated work (unknown tickers, more than the caps, technical without being asked).
- Router accuracy dropping below today's (100% on the labelled cases) because the gate now has a bigger output.
- One failing step (Pinecone, Yahoo) losing the other steps' evidence or crashing the tab.
- Any factual figure or outlook claim in the reply without a supplied source.

## Constraints
- Keep LangGraph: `router -> execute -> synthesize` and `router -> refuse`. Remove `agent`, `tools`, and `price` as separate nodes; `refuse` stays.
- Gate model: `typesafe/jev-router` through OpenRouter when `OPENROUTER_API_KEY` and `ROUTER_MODEL` are set, otherwise the OpenAI model (existing). Writer and grader/rewriter stay on `OPENAI_MODEL`.
- The gate prompt receives the held tickers, the asset classes in the holdings, and the indexed tickers, so it can fill valid calculator arguments.
- If the gate call fails: `route=answer` with an empty plan (writer sees the portfolio JSON only), trace "routing failed", matching today's fallback intent.
- Parallel execution: independent steps run concurrently with a thread pool; each step catches its own exceptions and, on failure, adds an "unavailable" fact and a trace line; total steps bounded by the caps above.
- The per-node timing (`timings`) keeps working for `router`, `execute` and `synthesize`; each step also reports its own duration in the trace.

## Decisions made with the user
- Deterministic plan emitted by the gate; planner and tool-calling agent both go away (user: "proceed with this workflow").
- Jev stays the front door; partial decline and outlook behaviour from the unified design stay.

## Calls made without the user (change if wrong)
- Routes shrink to `answer` and `out_of_scope`; `live_price` and `portfolio_only` become plans.
- No agent fallback.
- The calculator calls are emitted by the gate (not a second model call); the gate prompt carries the calculator menu text now in `PLANNER_PROMPT`.
- The filings query is written by the gate; the grader/rewriter inside the filings step still handle weak queries (one rewrite).
- Caps: 3 calculator calls, 3 tickers for analysts and price, one filings search.
- Existing eval labels are migrated: `live_price`/`portfolio_only`/`agent` cases become `answer` with expected plan fields; new `expect_steps` replaces `expect_tools` in the agent eval.

## What is left to the builder
- Pydantic schema names and field layout for the plan; JSON-schema friendliness for the OpenRouter model (flat fields, no unions the model may reject).
- How `execute` is split into step functions (reuse the existing retrieve/grade/rewrite/analysis/price code and `run_calc`).
- Migrating tests and removing dead code (`agent_node.py`, `agent_tools.py`, planner node and prompt, `ToolPlan`).
- Extending the router eval to check plan fields, and re-running router eval, agent eval and the latency profile.

## Success looks like
- Router eval at or above 100% on the migrated cases including plan-field checks.
- Agent eval cases still pass (Apple profit and growth with outlook; filings only; holdings plus filings; Tesla partial decline; refusal).
- Median latency on filings, mixed and outlook questions clearly below the agent path's 11-17 s; portfolio and price questions no slower than before.
- Fewer moving parts: no agent loop, no planner prompt, no tool-call caps.

## Resolved after builder review

Answered by the project:
- Price step: carry over the existing `price_node` and `quotes_from_market` behaviour unchanged (a missing or non-finite price becomes `price=None`, shown as "unavailable"; Yahoo failures never raise).

Calls made without the user (change if wrong):
- Gate capability: Jev is built as the gate and measured against the migrated router eval (including plan fields). It already scored 100% on the current structured output, including `wants_outlook` and `declined_part`. If it misses the bar, no new design is needed: set `ROUTER_MODEL` to an OpenAI-compatible model (the existing env switch) or drop the OpenRouter variables to use the OpenAI model. The builder reports the result, it does not build a second fallback path.
- Empty tickers: the gate resolves tickers from the question and the conversation `history` (the gate now receives the last turns). If the plan needs tickers and none were resolved, Python fills them as today: price uses all held tickers; analysts use the first 3 held tickers; filings use held tickers that are indexed. If that leaves nothing, the step is skipped.
- Gate failure: `route=answer` with an empty plan, so the writer sees the portfolio JSON only (today's behaviour on routing failure); the trace says "routing failed". No keyword fallback.
- Empty plan with `declined_part` set (for example "Should I buy Tesla?" when nothing else is in scope): the writer finds no evidence and returns the "I don't know" reply plus the "Not answered" line, as the unified design already does. `wants_outlook` with no usable analyst data yields the existing "analyst data was insufficient" note and no Outlook.
- Eval ground truth: the builder labels it, using subset matching. Each case lists the plan fields that must be present (for example filings, analysts including fundamental, wants_outlook); the filings query text is not checked; an extra `price` is allowed; `technical` appearing without a technical keyword in the question, steps on an `out_of_scope` route, or a missing required field is a failure.
