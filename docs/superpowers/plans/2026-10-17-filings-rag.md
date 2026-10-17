# Filings-grounded Ask AI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ask AI answers move/risk/number questions from retrieved SEC filing passages with citations, and a gold-set eval compares fixed vs semantic chunking and rerank vs no rerank.

**Architecture:** A new `rag/` package ingests 10-K/10-Q/8-K HTML from EDGAR, chunks it two ways (tables kept whole), upserts into one Pinecone dense and one sparse index (namespaces `fixed`, `semantic`), retrieves with reciprocal rank fusion plus a hosted reranker, and feeds passages into the existing `analyzer/ai.py` chat path.

**Tech Stack:** Python >=3.14, Streamlit, Pinecone SDK (hosted embed, sparse, rerank), OpenAI chat completions (existing), BeautifulSoup/lxml, pytest.

**Spec:** `docs/superpowers/specs/2026-10-17-filings-rag-design.md`

## Global Constraints
- Package `rag/` inside this repo; Pinecone Python SDK only (no MCP dependency).
- New deps: `pinecone`, `beautifulsoup4`, `lxml`, `requests` (pin like existing deps; if no lxml wheel for Python 3.14, use `html.parser`).
- Secrets only in git-ignored `.env`: `PINECONE_API_KEY`, `OPENAI_API_KEY`, new `SEC_USER_AGENT` ("Name email"). Never log or print them.
- EDGAR: send `SEC_USER_AGENT` on every request, at most 10 requests/second.
- Models: dense `llama-text-embed-v2`, sparse `pinecone-sparse-english-v0`, rerank `bge-reranker-v2-m3` (verify availability first; names are constants in `rag/store.py`).
- Namespaces `fixed` and `semantic`; merge with reciprocal rank fusion, top 30 -> rerank -> top 5.
- Record `_id` = `TICKER-form-date-section-n`; text field `text`; metadata `ticker, form, filing_date, fiscal_year, section, chunk_type (text|table), source_url`.
- Tables are never split. Fixed chunker ~500 tokens, 50 overlap.
- Citations render as `[TICKER form, date, section]`; passages are untrusted data; if nothing supports a cause, answer "not explained by filings".
- Retrieval failure or no indexed filings for held tickers degrades to portfolio-only; existing tests stay green.
- Filing text and questions go to OpenAI; the Ask AI caption must say so.

## Review Focus
- Filing with zero tables or zero recognizable headings: chunkers still produce chunks, no crash.
- A single table longer than the chunk size: stays one chunk.
- Held ticker with no filings (VOO, BTC-USD): portfolio-only answer, no error, stated as such.
- Pinecone/EDGAR down or `SEC_USER_AGENT` unset: Ask AI still answers portfolio-only with a caption; ingest fails with a clear message.
- Retrieval returns nothing relevant: answer says "not explained by filings" instead of inventing a cause.

---

### Task 1: Config, dependencies, EDGAR client
**Files:**
- Modify: `pyproject.toml`, `.env.example`, `.gitignore` (add `data/filings/`)
- Create: `rag/__init__.py`, `rag/ingest.py`
- Test: `tests/test_rag_ingest.py`

**Interfaces:**
- Produces: `FilingRef = dataclass(ticker, form, filing_date, accession, url)`; `list_filings(ticker: str, session=None, today: date | None = None) -> list[FilingRef]` (latest 10-K, last 3 10-Qs, 8-Ks in last 90 days, newest first); `download(ref: FilingRef, session=None) -> Path` (cache in `data/filings/`); raises `IngestError` if `SEC_USER_AGENT` unset.

- [ ] **Step 1: Failing tests** with a fake session returning canned `company_tickers.json` and submissions JSON: `test_list_filings_selects_latest_10k_three_10qs_recent_8ks`, `test_ticker_to_cik_pads_to_ten_digits` (AAPL -> `0000320193`), `test_missing_user_agent_raises_ingest_error`, `test_unknown_ticker_returns_empty_list`.
- [ ] **Step 2: Run** `uv run pytest tests/test_rag_ingest.py -v`; expect FAIL (module missing).
- [ ] **Step 3: Implement** per Interfaces; throttle to <=10 req/s; skip download if cached file exists.
- [ ] **Step 4: Run tests**; expect PASS.
- [ ] **Step 5: Commit** `feat(rag): EDGAR filing list and download`.

### Task 2: Filing parser (sections and tables)
**Files:**
- Modify: `rag/ingest.py`
- Test: `tests/test_rag_parse.py` (small HTML fixtures in the test)

**Interfaces:**
- Produces: `Block = dataclass(kind: Literal["text","table"], section: str, text: str)`; `parse_filing(html: str) -> list[Block]`. Tables become Markdown with caption and section heading prepended; section from "Item N." headings, `"Document"` when none found.

- [ ] **Step 1: Failing tests:** `test_table_becomes_single_markdown_block_with_section`, `test_text_split_by_item_headings`, `test_no_headings_or_tables_still_returns_blocks`.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** `parse_filing` with BeautifulSoup; drop empty/layout-only tables.
- [ ] **Step 4: Run**; expect PASS.
- [ ] **Step 5: Commit** `feat(rag): parse filings into text and table blocks`.

### Task 3: Chunkers
**Files:**
- Create: `rag/chunking.py`
- Test: `tests/test_rag_chunking.py`

**Interfaces:**
- Consumes: `Block`.
- Produces: `Chunk = dataclass(id, text, metadata: dict)`; `fixed_size(blocks, meta, size=500, overlap=50) -> list[Chunk]`; `semantic(blocks, meta, embed: Callable[[list[str]], list[list[float]]], threshold=0.75) -> list[Chunk]` (merge consecutive sentences while cosine similarity to the running chunk >= threshold; split at headings).

- [ ] **Step 1: Failing tests:** `test_tables_never_split_in_either_strategy` (table of 3x size stays one chunk), `test_fixed_chunks_overlap_by_50_tokens`, `test_semantic_splits_where_similarity_drops` (fake `embed`), `test_chunk_ids_follow_ticker_form_date_section_n`, `test_empty_blocks_give_no_chunks`.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** both; token count by whitespace split (builder's call, note in README).
- [ ] **Step 4: Run**; expect PASS.
- [ ] **Step 5: Commit** `feat(rag): fixed-size and semantic chunkers`.

### Task 4: Pinecone store
**Files:**
- Create: `rag/store.py`, `rag/build_index.py` (CLI: list, download, parse, chunk both ways, upsert, print stats)
- Test: `tests/test_rag_store.py` (fake Pinecone client)

**Interfaces:**
- Consumes: `Chunk`.
- Produces: constants `DENSE_INDEX`, `SPARSE_INDEX`, model names; `ensure_indexes(pc) -> None` (`create_index_for_model`, idempotent); `upsert_chunks(pc, namespace: str, chunks: list[Chunk], batch=90) -> int`; `stats(pc) -> dict[str, int]` (records per index/namespace). Deterministic ids make re-runs idempotent.

- [ ] **Step 1: Failing tests:** `test_ensure_indexes_is_idempotent`, `test_upsert_batches_of_90_to_both_indexes`, `test_upsert_returns_record_count`.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement**; first check real model names with `pc.inference.list_models()` and adjust constants (record the result in README).
- [ ] **Step 4: Run tests**; expect PASS.
- [ ] **Step 5: Live check:** run `uv run python -m rag.build_index` with `SEC_USER_AGENT` set; `stats()` counts match the chunks printed.
- [ ] **Step 6: Commit** `feat(rag): Pinecone dense and sparse indexes`.

### Task 5: Hybrid retrieval with rerank
**Files:**
- Create: `rag/retrieve.py`
- Test: `tests/test_rag_retrieve.py` (fake client)

**Interfaces:**
- Consumes: store constants.
- Produces: `Passage = dataclass(id, text, ticker, form, filing_date, section, source_url, score)`; `rrf(dense: list[str], sparse: list[str], k=60) -> list[str]`; `retrieve(pc, query: str, tickers: list[str], namespace: str, rerank: bool = True, top_n: int = 5, candidates: int = 30) -> list[Passage]`. Returns `[]` when `tickers` is empty; raises `RetrievalError` on Pinecone failure.

- [ ] **Step 1: Failing tests:** `test_rrf_rewards_items_ranked_high_in_both_lists`, `test_retrieve_filters_by_tickers`, `test_rerank_false_skips_rerank_call`, `test_empty_tickers_returns_empty`, `test_pinecone_error_becomes_retrieval_error`.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** (`index.search` on both indexes with a ticker `$in` filter; `pc.inference.rerank` on the merged top 30).
- [ ] **Step 4: Run**; expect PASS.
- [ ] **Step 5: Commit** `feat(rag): hybrid retrieval with RRF and rerank`.

### Task 6: Evaluation harness and gold set
**Files:**
- Create: `rag/eval/__init__.py`, `rag/eval/metrics.py`, `rag/eval/run_eval.py`, `rag/eval/gold.jsonl`
- Test: `tests/test_rag_metrics.py`

**Interfaces:**
- Produces: `hit_at_k(ranked: list[str], gold: set[str], k: int) -> float`; `mrr(ranked, gold) -> float`; `recall_at_k(ranked, gold, k) -> float`; gold line `{"question", "tickers", "gold_ids": [...], "answerable": bool}`; for `answerable=false` the score is the share of questions where the top reranker score falls below a threshold (document the threshold). `python -m rag.eval.run_eval` prints the 2x2 table.

- [ ] **Step 1: Failing tests:** `test_hit_at_k_basic`, `test_mrr_first_rank_and_miss`, `test_recall_at_k_partial`, `test_empty_gold_does_not_divide_by_zero`.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement** metrics and runner.
- [ ] **Step 4: Run tests**; expect PASS.
- [ ] **Step 5: Draft ~30 gold questions** (incl. unanswerable) from the indexed filings with the passage ids each should retrieve; **stop and have the user review/correct `gold.jsonl`** before running the eval.
- [ ] **Step 6: Run eval**, save the table to `docs/superpowers/specs/2026-10-17-filings-rag-results.md`; report honestly even if rerank gives no gain.
- [ ] **Step 7: Commit** `feat(rag): eval harness and gold set`.

### Task 7: Passages and citations in the chat prompt
**Files:**
- Modify: `analyzer/ai.py` (`build_chat_messages`, `chat_reply`)
- Test: `tests/test_ai.py`

**Interfaces:**
- Consumes: `rag.retrieve.Passage`.
- Produces: `build_chat_messages(summary_json, history, question, max_turns=MAX_CHAT_TURNS, passages: list | None = None)`; `chat_reply(..., passages=None)`. With passages, the system prompt appends them labelled `[TICKER form, date, section]`, requires citations, treats passages as untrusted data, allows "not explained by filings", labels filings older than the move window as background. `passages=None` leaves current behavior byte-identical.

- [ ] **Step 1: Failing tests:** `test_messages_without_passages_unchanged`, `test_passages_appear_with_citation_labels`, `test_prompt_requires_not_explained_fallback`, `test_passage_text_truncated_per_passage` (4,000 chars).
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** full `uv run pytest -q`; expect PASS.
- [ ] **Step 5: Commit** `feat(ai): cite filing passages in chat prompt`.

### Task 8: Ask AI integration
**Files:**
- Modify: `app.py` (`render_ask_ai`), `README.md`, `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: `retrieve(...)`, `chat_reply(..., passages=...)`.
- Behavior: before `chat_reply`, call `retrieve` with held tickers that have filings and namespace from `RAG_NAMESPACE` env (default `semantic`, set to the eval winner); show a "Sources" expander of quoted passages under each answer (store sources with each chat message); on `RetrievalError` or no tickers, answer portfolio-only and show a caption; update the disclosure caption.

- [ ] **Step 1: Failing tests** (monkeypatched `retrieve` and `chat_reply`): `test_answer_shows_sources_expander`, `test_retrieval_error_falls_back_to_portfolio_only`, `test_etf_only_portfolio_skips_retrieval`.
- [ ] **Step 2: Run**; expect FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`; expect PASS.
- [ ] **Step 5: Manual check:** restart the app; ask "Why is NVDA down?" (cites an 8-K/10-Q or says filings don't explain it) and a VOO question (portfolio-only).
- [ ] **Step 6: README:** setup (`SEC_USER_AGENT`, build_index, eval), chunking/rerank findings, data-sent disclosure.
- [ ] **Step 7: Commit** `feat: filings-grounded Ask AI`.
