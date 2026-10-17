# Agentic Ask AI (LangGraph router, graded retrieval, memo): Design

Date: 2026-10-17
Status: Approved in conversation; not yet implemented.
Builds on: [filings RAG design](2026-10-17-filings-rag-design.md) (retrieval, chunking, Pinecone indexes already built).

## Intent and why
Ask AI should behave like a small agent system, not a fixed pipeline. A classifier agent decides what a question needs (SEC filings, a live stock price, both, portfolio data only, or nothing it can answer), then the matching agents run. A grader checks that retrieved passages really answer the question, and a rewriter retries when they do not. The final answer is a short financial memo built only from verified material. The user's hard requirement: answers are grounded, and when the data does not support an answer the assistant politely says it does not know.

## Goals
- A LangGraph graph with these nodes: Router, Retrieval, Document Grader, Query Rewriter, Price Tool, Synthesizer.
- Router, Grader and Rewriter return structured output (Pydantic schemas via the OpenAI structured-output API), never free text that is parsed.
- Every Ask AI question runs through the graph; the reply is a memo with four sections: Answer, Key figures, Sources, What I couldn't verify.
- An "Agent trace" expander in the app shows the route chosen, each grader verdict, and each rewrite, so the user can verify behaviour.
- Existing grounding rules stay: temperature 0, citations validated against supplied sources, "I don't know" fallback sentence (`ai.IDK_MESSAGE`).

## Non-goals
Free-form tool use by the model beyond these nodes; news or analyst data; trading advice; companies other than those already indexed (Apple today); streaming output; multi-turn memory inside the graph (the chat history keeps its current handling).

## Anti-goals
- A memo that states a number or cause no supplied source supports.
- A live price presented without its timestamp, or a stale/failed price presented as current.
- An endless retry loop or runaway model cost.
- The graph crashing the Ask AI tab when OpenAI, Pinecone or Yahoo is unavailable.

## Constraints
- LangGraph (user's choice). Dry run resolves on Python 3.14 (langgraph 1.2.x, 18 packages, downgrades `websockets` 17.2 to 16.1.1); the builder must confirm Streamlit still works after install. Do not add `langchain-openai`; use the existing `openai` SDK.
- Model from `OPENAI_MODEL` (default `gpt-4o-mini`) for all nodes; temperature 0.
- Query rewrites capped at 2. Maximum model calls per question: router 1 + grader 1 per retrieval round + rewriter 1 per retry + synthesizer 1.
- Live prices only through the existing `market.fetch_quotes`; each price carries the fetch time; failure yields "unavailable", never a guess.
- Filings only for indexed tickers (`rag.store.INDEXED_TICKERS`); other tickers cannot take the filings path.
- Questions and passages go to OpenAI; the existing disclosure caption stays accurate.

## Behaviour by route (Router output: `route`, `tickers`, `reason`)
- `sec_filings`: Retrieval -> Grader. If at least one passage is relevant, continue to the Synthesizer with only the relevant passages. If none, Rewriter -> Retrieval again (max 2 rewrites), then the Synthesizer with no passages, which must answer with the "I don't know" sentence.
- `live_price`: Price Tool for the named or held tickers -> Synthesizer.
- `both`: Retrieval/Grader/Rewriter loop, then Price Tool -> Synthesizer.
- `portfolio_only`: straight to the Synthesizer with the portfolio JSON.
- `out_of_scope` (for example "is Apple a good buy?"): the "I don't know" reply without calling retrieval or price.
- If the Router call itself fails or returns an invalid schema: treat as `portfolio_only` and say in the trace that routing failed (Claude's call; builder may refine).

## Data shared between nodes (graph state)
question, chat history, portfolio JSON, route decision, search query (current), passages, graded passages, rewrite count, price quotes with timestamps, memo text, trace entries.

## Claude's calls (flag if wrong)
- Graph code lives in a new `agent/` package (`state.py`, `schemas.py`, `nodes.py`, `graph.py`), importing `rag.retrieve` and `market`; `analyzer/ai.py` keeps prompts shared with the audit.
- The grader judges each retrieved passage individually (relevant yes/no with a one-line reason); it replaces the fixed 0.2 reranker cutoff in the app but the cutoff stays as a cheap pre-filter.
- Price node covers tickers the router names, falling back to held tickers.
- Memo section "What I couldn't verify" is filled by the Synthesizer from the grader and price results (missing quotes, ungraded passages, omitted holdings).

## Left to the builder
Prompt wording for the four LLM nodes, exact Pydantic field names beyond those above, trace formatting, and how the graph is compiled and cached in Streamlit.

## Verification
- Unit tests per node with fake OpenAI and fake retrieval clients; graph tests for each route, the rewrite cap, and node failures.
- Router eval: a small labelled set of questions with expected routes, scored by accuracy.
- Manual: ask a filings question, a live-price question, a combined one, "Is Apple a good buy?" (expect the polite refusal), and a question the filings cannot answer (expect refusal after rewrites, visible in the trace).
