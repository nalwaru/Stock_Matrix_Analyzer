# Stock Portfolio Analyzer

Streamlit app: upload a portfolio CSV, see live-priced KPIs and charts, stress-test
asset classes, get an AI audit, chat about your holdings, and export the results.

## Setup

```bash
uv sync
cp .env.example .env   # then add your OPENAI_API_KEY (optional; Ask AI tab is disabled without it)
uv run streamlit run app.py
```

## CSV format

Required columns: `Ticker, Shares, Purchase_Price, Purchase_Date, Asset_Class`.
Use `data/sample_portfolio.csv` as a template, or switch on **Use Sample Dataset**.
Crypto uses Yahoo symbols (e.g. `BTC-USD`).

## Behaviour notes

- Quotes are cached for 5 minutes. Tickers with no quote are flagged and excluded from totals.
- The AI audit sends only aggregated data (allocation %, top 3 positions, ROI). The Ask AI chat also sends your filtered per-holding rows (ticker, shares, cost, price, value, P/L; up to 100) and your questions to OpenAI.
- The model defaults to `gpt-4o-mini`; override with `OPENAI_MODEL`.
- Print-friendly mode (Export tab) hides the sidebar/controls; then use the browser's Print > Save as PDF.
- AI output is educational, not financial advice.

## Development

```bash
uv run pytest
```

## Filings-grounded Ask AI (optional)

Ask AI can cite SEC filings (latest 10-K, last 3 10-Qs, last 90 days of 8-Ks) for AAPL (add more tickers to `INDEXED_TICKERS` in `rag/store.py`, then re-run the loader).

1. Set `SEC_USER_AGENT="Your Name you@example.com"` (EDGAR requires it) and `PINECONE_API_KEY` in `.env`.
2. Build the indexes once: `uv run python -m rag.build_index` (creates `filings-dense` and `filings-sparse` with namespaces `fixed` and `semantic`; re-running overwrites, it does not duplicate).
3. Compare strategies: put questions in `rag/eval/gold.jsonl` (`{"question", "tickers", "gold_ids", "answerable"}` per line), then `uv run python -m rag.eval.run_eval` prints the {fixed, semantic} x {no rerank, rerank} table. Set `RAG_NAMESPACE` to the winner (default `semantic`).

Answers cite `[TICKER form, date, section]`, show the quoted passages under **Sources**, and say the move is not explained by filings when no passage supports a cause. If filing search is unavailable, Ask AI answers from portfolio data only and says so. Your questions and the retrieved passages are sent to OpenAI. Tokens are counted as whitespace-separated words.
