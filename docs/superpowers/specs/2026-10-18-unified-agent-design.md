# Unified Ask AI agent (Jev gate, direct paths, tool-calling agent, outlook section): Design

Date: 2026-10-18
Status: Draft for approval; not implemented.
Builds on: [agentic Ask AI design](2026-10-17-agentic-ask-ai-design.md) (LangGraph, router, graded retrieval, memo) and the [filings RAG design](2026-10-17-filings-rag-design.md) (Pinecone dense and sparse indexes).

## Intent and why
Today every question takes exactly one fixed path chosen by the router: filings, live price, filings plus price, portfolio only, analysis, or refusal. A question that needs more than one kind of evidence cannot be answered well. The motivating question: "What is Apple's profit? Do you expect growth next year?" The profit half is a filing fact; the growth half asks for a view. The router must pick one route, so the user gets either a refusal for both halves or an answer that ignores one.

The user wants one agent that chooses its own tools per question and can combine filing facts, portfolio numbers and the analyst checks (risk, technical, fundamental, ownership) in one reply. The reply has separate sections: facts from filings (cited), and, when the question asks about the future, a clearly labelled outlook commentary built on the analyst data. Simple questions keep their fast direct paths. The Jev router stays as the front door that declines out-of-scope requests.

## Goals
- A front door (the existing router, on `typesafe/jev-router` via OpenRouter) that decides: decline, direct price path, direct portfolio path, or hand to the agent. It still extracts tickers (company names resolved to symbols).
- One tool-calling agent (OpenAI, same model as other nodes) that, for handed-over questions, calls any combination of these tools until it has enough evidence or hits the call cap:
  - `search_filings(query, tickers)`: the existing hybrid retrieval, reranker threshold and passage grading (one rewrite retry allowed inside the tool).
  - `portfolio_calc(...)`: the existing calculator (holding, class_weight, top_n, total, stress).
  - `run_analysts(ticker, kinds)`: the existing four analysts; technical also produces its chart.
  - `get_price(tickers)`: the existing quote and price-change lookup.
- Every tool returns citable facts or passages exactly as now; the agent never computes figures itself.
- A final writer step produces the memo with separate sections: Answer (from filings and holdings), Key figures, Sources, What I couldn't verify, and an optional Outlook section.
- The Outlook section appears only when the question asks about the future of a company's business (for example growth next year). It is built only from supplied analyst facts (valuation, growth and margin data, analyst targets and ratings, risk, trend), gives a plain leaning (positive, neutral or cautious) with three to four cited reasons, and carries the label "Estimate based on the data above, not a filing fact and not financial advice".
- Existing guardrails stay on every factual part: citations must come from supplied evidence, the number checker lists unmatched figures under "What I couldn't verify", and failure produces the "I don't know" reply.
- The Agent trace expander keeps showing every step, now including each tool the agent called and with what arguments.

## Non-goals
- Buy/sell/hold advice, price-target predictions of the user's own, or any specific growth percentage invented by the model.
- Earnings call transcripts or news (not in the index; the outlook uses analyst data only).
- Moving the portfolio CSV into Pinecone, or any local copy of filings at runtime (removed on 2026-10-18; the offline index build in `rag/` is unchanged).
- Streaming the memo text token by token.
- Replacing LangGraph.

## Anti-goals (a failure even if it technically works)
- An outlook that states a number or claim not in a supplied fact, or that reads as a recommendation.
- An outlook presented as if it came from the filings.
- An agent loop that runs away: unbounded tool calls, repeated identical calls, or unbounded cost.
- Direct paths getting slower or less accurate than today.
- The tab crashing when OpenAI, OpenRouter, Pinecone or Yahoo is down.

## Constraints
- Keep LangGraph. New top-level shape: `gate (Jev) -> { refuse | price | tools (portfolio) | agent } -> synthesize`. `agent` is one node that runs the tool-calling loop internally. The old top-level `retrieve`, `grade`, `rewrite` and `analysis` nodes become internals of the tools above (their prompts, thresholds and tests move with them).
- Gate model: `typesafe/jev-router` through OpenRouter when `OPENROUTER_API_KEY` and `ROUTER_MODEL` are set, otherwise the OpenAI model (existing behaviour). Agent and writer: `OPENAI_MODEL` (default `gpt-4o-mini`), temperature 0, existing OpenAI SDK (no new LLM library).
- Agent loop caps: at most 6 tool calls and 4 model turns per question; a repeated identical call is skipped; hitting a cap proceeds to the writer with what was gathered.
- Tool errors never raise: a failing tool yields an "unavailable" fact and a trace line (existing analysts already do this).
- Direct paths: `live_price` and `portfolio_only` keep today's behaviour and model-call counts. Everything else in scope (filings questions, analysis, mixed, outlook) goes to the agent.
- Gate rule changes: questions about a company's business outlook or growth are in scope and go to the agent; advice ("should I buy"), predictions of a stock's price, and off-topic requests stay out of scope.
- Filings only for indexed tickers; live prices only through `market.fetch_quotes`, each with its fetch time; Yahoo data is unofficial and fields may be missing (shown as "unavailable").
- Questions, passages and facts go to OpenAI and OpenRouter; the disclosure caption in the app must say so.

## Decisions made with the user
- One agent with tools, not a fixed multi-path graph, so one question can use several data sources.
- Jev remains the front door.
- A question with an in-scope part and an out-of-scope part is answered for the in-scope part, and the out-of-scope part is politely declined in the same reply (for example "Should I buy Tesla, and what's their revenue?" gives the revenue and a short note that buy advice is not something this tool gives). Only a question with nothing in scope gets the plain "I don't know" reply. The gate therefore returns a `declined_part` note (empty when nothing is declined) alongside the route; the writer states the decline in one sentence and never answers the declined part.
- Simple questions keep direct paths.
- The reply has a filings section and a separately labelled outlook section.
- The outlook is based on the risk, technical, fundamental and ownership analysts, as a plain leaning with reasons, not a model-invented growth number. If the data is thin or missing it says so and does not force a view.

## Calls made without the user (change if wrong)
- "Direct paths" means exactly `live_price` and `portfolio_only`. Single-source filings questions go through the agent, which costs about the same model calls as today's loop but loses the fixed order.
- The gate's route set becomes: `live_price`, `portfolio_only`, `agent`, `out_of_scope`.
- Business-outlook questions are in scope; explicit price predictions ("will AAPL hit 300?") stay out of scope.
- Tool and loop caps above; the agent uses OpenAI tool calling.
- Memo gets an optional outlook field; the number checker also runs over the outlook text.

## What is left to the builder
- Tool schemas and argument validation, and the agent's system prompt.
- How trace lines are streamed while the agent loop is running.
- How the memo schema and renderer carry the optional outlook section.
- Evals: update the router eval cases for the new route set, add agent eval cases (including the Apple profit-and-growth question, a purely filings question, a mixed holdings-and-filings question, and an advice question that must be declined), and keep all existing tests green.
- Migration order and test strategy using the scripted fake clients already in `tests/agent_fakes.py`.
- Per-node timing (already in `agent/graph.py`) should cover the agent node and report each tool call duration.

## Success looks like
- "What is Apple's profit? Do you expect growth next year?" returns cited profit facts, plus an outlook section with a leaning and reasons from analyst data, in one reply.
- "Should I buy Tesla?" is still declined at the gate.
- Price and portfolio questions are as fast as today.
- No factual figure appears in the answer without a supplied source; the outlook's figures are all from supplied facts.

## Resolved after builder review

Answered by the project:
- Follow-up questions: the existing `history` (last turns) is passed to the gate, agent and writer exactly as today. Portfolio context: `portfolio_calc` reads the loaded holdings from `Deps.holdings`; with no portfolio loaded it returns nothing and the trace says so.
- Working tree: the planner/filing-tool removal and the per-node timing are uncommitted at the time of writing. The plan builds on them and they are committed before implementation starts.

Calls made without the user (change if wrong):
- Facts and the checker: every analyst result is a list of `Fact(label, text)` as today, with labels like `TICKER technical, DATE, RSI`. Each outlook reason must end with one of those labels. The number checker runs over the outlook; a figure that is not in a supplied fact is listed under "What I couldn't verify". Derived values ("margin up 2 points") are not allowed unless a fact states them, same as the existing "never compute" rule. A reason without a number must still cite a label; a reason with no label is dropped by the writer check.
- Outlook trigger: the gate sets a boolean `wants_outlook` (with the route). When true, the agent must call `run_analysts` for the named tickers; the writer then produces the Outlook section. An outlook-only question still gets an Answer section built from whatever facts were gathered.
- Caps: a skipped repeated call does not count toward the 6-tool-call cap but does count as a model turn toward the 4-turn cap. If a cap is hit, the writer adds "The agent stopped at its step limit, so evidence may be incomplete" under "What I couldn't verify".
- Performance: no latency limit. The test asserts model-call counts for the direct paths; `agent/eval/profile_latency.py` is the manual baseline.
