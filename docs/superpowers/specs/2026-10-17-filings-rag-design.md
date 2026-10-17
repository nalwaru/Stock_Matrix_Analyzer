# Filings-grounded Ask AI (table-aware hybrid RAG): Design

Date: 2026-10-17
Status: Approved in conversation; not yet implemented.

## Intent and why
A user looks at a holding that moved (for example NVDA) and asks Ask AI why. The answer should be grounded in company filings and cite them. The same work is the "Financial Document Intelligence" exercise: compare fixed-size and semantic chunking, add reranking, and measure the improvement on one query set.

## Goals
- Ask AI answers move, risk and number questions using the portfolio JSON plus retrieved filing passages. Each claim is cited as [TICKER form, date, section], and the quoted passage is visible under "Sources".
- Retrieval is table-aware and hybrid (dense plus sparse), tables are never split, and results are reranked.
- One gold question set produces a comparison of {fixed, semantic} x {no rerank, rerank} using hit@k, MRR and recall@k.
- When the filings do not explain a move, the answer says so.

## Non-goals
News, analyst commentary, earnings call transcripts (no free official source), market-wide explanations, filings for ETFs or crypto, and insurance or loan documents (the technique transfers later).

## Anti-goals
The failure to design against is a confident but wrong reason.
- Never state a cause that no retrieved passage supports; say "not explained by filings".
- Only filings dated near the move may be offered as a cause. Older filings are labelled background. (Claude's guess; the window is a builder decision.)
- The gold set includes unanswerable questions whose correct result is "cannot tell from filings". Reranking is judged on those as well.

## Constraints
- `rag/` package inside Stock_Matrix_Analyzer, not a separate project.
- Pinecone for dense vectors, sparse vectors and reranking, using the Python SDK (the Pinecone MCP is not required).
- Python >= 3.14 per `pyproject.toml`; new dependencies `pinecone`, `beautifulsoup4`, `lxml`, `requests`. Confirm an lxml wheel exists, otherwise use `html.parser`.
- Secrets in the git-ignored `.env`: `PINECONE_API_KEY`, `OPENAI_API_KEY`, and a new `SEC_USER_AGENT` ("Name email"), which EDGAR requires. The user has not yet supplied this value.
- EDGAR allows at most 10 requests per second.
- Filing text and the user's questions are sent to OpenAI; the UI disclosure caption must say so.

## Decisions made by the user
- Package inside this repo; Pinecone for all three retrieval stages.
- Ask AI always retrieves, filtered to held tickers that have indexed filings.
- Filings only (10-K, 10-Q, 8-K).
- Claude drafts about 30 gold questions including unanswerable ones; the user reviews them.

## Claude's calls (flag if wrong)
- Corpus: for each held stock (AAPL, MSFT, NVDA, AMZN, GOOGL; O later), the latest 10-K, the last 2-3 10-Qs and the last 90 days of 8-Ks. Held ETFs and crypto get portfolio-only answers, stated as such.
- Models: `llama-text-embed-v2` (dense), `pinecone-sparse-english-v0` (sparse), `bge-reranker-v2-m3`. Verify availability on the user's plan first.
- Indexes: one dense and one sparse index, each with namespaces `fixed` and `semantic`. Merge with reciprocal rank fusion, take the top 30, rerank to the top 5.
- Record: `_id` TICKER-form-date-section-n; text field `text`; metadata ticker, form, filing_date, fiscal_year, section, chunk_type (text|table), source_url. Tables are single chunks in Markdown with caption and section heading prepended.
- Chunkers: fixed-size about 500 tokens with 50 overlap; semantic splits on headings, then merges sentences until embedding similarity drops.

## Components
- `rag/ingest.py`: ticker to CIK via `company_tickers.json`, filing list via `data.sec.gov/submissions`, HTML download to git-ignored `data/filings/`, parse to sections and tables.
- `rag/chunking.py`, `rag/store.py` (create indexes, batched `upsert_records`, stats), `rag/retrieve.py` (filter, hybrid query, rerank, return passages with ids and URLs).
- `rag/eval/gold.jsonl` and `rag/eval/run_eval.py` (prints the 2x2 table).
- `analyzer/ai.py`: `build_chat_messages` gains an optional `passages` argument; the prompt requires citations, treats passages as untrusted data, and allows "not explained by filings". Reuses `chat_reply`, `AIError` and the safe error mapping.
- `app.py` `render_ask_ai`: retrieve before `chat_reply`, show a Sources expander, update the disclosure caption, and degrade to portfolio-only if retrieval fails.
- Tests: chunker invariants (tables never split, overlap), metric functions, retrieval with a fake Pinecone client, prompt with passages, app smoke (no key, no index). README documents setup, ingest and eval.

## Left to the builder
Chunk sizes, the exact "near the move" window, the number of 10-Qs and 8-Ks, batch sizes, and how to detect section headings in each filing's HTML.

## Estimates (unmeasured)
One-time ingest 10-25 minutes; eval 1-3 minutes; about 5-10 seconds per Ask AI question.

## Verification
`uv run pytest`; `uv run python -m rag.eval.run_eval` prints the 2x2 table; in the app, "Why is NVDA down?" cites a recent 8-K or 10-Q or says the filings do not explain it, and a VOO question is answered from portfolio data only.
