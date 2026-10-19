# Stock Portfolio Analyzer

Streamlit app: upload a portfolio CSV, see live-priced KPIs and charts, stress-test
asset classes, chat about your holdings and SEC filings, and export the results.

## Setup

```bash
uv sync
cp .env.example .env   # then add your OPENAI_API_KEY (optional; Ask AI tab is disabled without it)
uv run streamlit run app.py
```

For Ask AI (prerequisites, keys, the filing index, what changed and troubleshooting), see [docs/ASK_AI.md](docs/ASK_AI.md).

## CSV format

Required columns: `Ticker, Shares, Purchase_Price, Purchase_Date, Asset_Class`.
Use `data/sample_portfolio.csv` as a template, or switch on **Use Sample Dataset**.
Crypto uses Yahoo symbols (e.g. `BTC-USD`).

## Behaviour notes

- Quotes are cached for 5 minutes. Tickers with no quote are flagged and excluded from totals.
- The Ask AI chat sends your filtered per-holding rows (ticker, shares, cost, price, value, P/L; up to 100), matching filing passages and your questions to OpenAI (and the question to OpenRouter when the router model is configured).
- The model defaults to `gpt-4o-mini`; override with `OPENAI_MODEL`.
- Print-friendly mode (Export tab) hides the sidebar/controls; then use the browser's Print > Save as PDF.
- AI output is educational, not financial advice.

## Development

```bash
uv run pytest
```

## Filings-grounded Ask AI (optional)

Ask AI can cite SEC filings (latest 10-K, last 3 10-Qs, last 90 days of 8-Ks) for AAPL and RIVN (add more tickers to `INDEXED_TICKERS` in `rag/store.py`, then re-run the loader).

1. Set `SEC_USER_AGENT="Your Name you@example.com"` (EDGAR requires it) and `PINECONE_API_KEY` in `.env`.
2. Build the indexes once: `uv run python -m rag.build_index` (creates `filings-dense` and `filings-sparse` with namespaces `fixed` and `semantic`; re-running overwrites, it does not duplicate).
3. Compare strategies: put questions in `rag/eval/gold.jsonl` (`{"question", "tickers", "evidence": ["exact phrase from the filing"], "answerable"}` per line), then `uv run python -m rag.eval.run_eval` prints the {fixed, semantic} x {no rerank, rerank} table. Set `RAG_NAMESPACE` to the winner (default `semantic`).

Answers cite `[TICKER form, date, section]`, show the quoted passages under **Sources**, and say the move is not explained by filings when no passage supports a cause. If filing search is unavailable, Ask AI answers from portfolio data only and says so. Your questions and the retrieved passages are sent to OpenAI. Tokens are counted as whitespace-separated words.

**Grounding.** Ask AI is instructed to use only your portfolio data and the retrieved filing passages, with temperature 0. If they do not contain the answer it replies "I'm sorry, I don't know…". Passages scoring below `MIN_RELEVANCE` (`rag/retrieve.py`, starting value 0.2) are not sent to the model, and a reply that cites a filing passage that was never supplied is replaced by the same "I don't know". This reduces but cannot eliminate wrong answers: the model can still misread a supplied passage, so check the Sources.

## Agent graph (Ask AI)

Every Ask AI question runs through a LangGraph graph (`agent/`). A **gate** (the router, optionally on an OpenRouter model such as `typesafe/jev-router` via `OPENROUTER_API_KEY` and `ROUTER_MODEL`) picks `live_price`, `portfolio_only`, `agent` or `out_of_scope`, resolves tickers, flags questions that ask about a company's future (`wants_outlook`) and notes any part of the question it will not answer (`declined_part`). `live_price` and `portfolio_only` are fast direct paths (price lookup, or a planner choosing portfolio calculator calls). Everything else goes to a **tool-calling agent** that may call `search_filings` (hybrid Pinecone retrieval, graded, one rewrite), `portfolio_calc`, `run_analysts` and `get_price` in any combination (max 6 tool calls, 4 turns). A **synthesizer** writes the memo (Answer, Key figures, Sources, What I couldn't verify, plus an **Outlook** section when asked: a leaning with cited reasons built only from analyst facts, labelled as an estimate); any reply citing a source that was not supplied becomes "I don't know". Each answer has an **Agent trace** expander showing the route and every tool call. Score the gate with `uv run python -m agent.eval.run_router_eval`.

### Tools and analysts

All tool results are computed in Python and returned as cited facts; the model only writes the memo.
- **Portfolio calculator** (`portfolio_only` route): holding, asset-class weight, top-N, totals and what-if stress, chosen by a planner call (max 3 calls).
- **Price history** (price path): 1-day, 1-week and 1-month changes from Yahoo.
- **Analysts** (`run_analysts` tool on the agent path, max 3 tickers; technical only when asked): technical (SMA/EMA, RSI, MACD, Bollinger, ATR, volume, 52-week range, support/resistance), risk (volatility, beta vs SPY, drawdown, Sharpe at an assumed 4% risk-free rate, correlations), fundamental (valuation, margins, growth, debt, cash flow, dividend) and ownership (consensus, price targets, rating changes, holders, insiders, earnings date). Data is Yahoo Finance via `yfinance` (unofficial; missing fields show as "unavailable").
- **Number checker:** figures in a memo that appear in no supplied source are listed under "What I couldn't verify". RSI/ATR use Wilder smoothing seeded from the first value, so they can differ slightly from charting sites.

A technical analysis answer also draws a chart (price with SMA20/50/200 and Bollinger band, volume, RSI with 30/70 lines, MACD with signal and histogram). The chart and the cited facts use the same indicator formulas (`agent/analysts/technical.py`).
