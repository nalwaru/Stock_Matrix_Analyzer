# Stock Portfolio Analyzer — Design

Date: 2026-10-08
Status: Draft, awaiting user review

## 1. Purpose and scope

A single-user Streamlit app (localhost, no auth, no database) that ingests a portfolio CSV,
enriches it with live `yfinance` prices, and presents KPIs, Plotly charts, filters, a what-if
stress simulator, AI-generated insights (OpenAI), and CSV export.

Covers user stories 1.1, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2, 5.1, 5.2, 5.3.

Out of scope: authentication, persistence, multi-currency, historical price charts,
server-side PDF generation.

## 2. Decisions

- **LLM provider:** OpenAI (`openai` SDK), replacing `google-genai` from story 1.1. Model is read
  from `OPENAI_MODEL` (one default in one place). Key from `OPENAI_API_KEY` (env or `.env`).
- **Packaging:** `uv init --no-package stock-portfolio-analyzer`; dependencies `streamlit`,
  `pandas`, `plotly`, `yfinance`, `openai`, `python-dotenv` (+ dev: `pytest`) locked in `uv.lock`.
- **Architecture:** layered modules; `app.py` is UI wiring only. Financial logic is pure
  (no Streamlit imports) and unit tested.
- **Sample data:** `data/sample_portfolio.csv`, derived from the user-supplied
  `stock_sample.numbers` (12 holdings: Stocks, ETFs, Crypto, REITs; includes `BTC-USD`/`ETH-USD`
  and fractional shares). Padding rows and empty columns in the source are dropped.

## 3. Structure

```
app.py                    UI wiring (sidebar, KPIs, charts, tabs)
analyzer/
  schema.py               CSV validation
  market.py               yfinance adapter, @st.cache_data(ttl=300)
  portfolio.py            enrich(), allocation, top holdings
  scenarios.py            apply_shocks()
  ai.py                   OpenAI client: audit (cached) + chat
  export.py               enriched CSV bytes, print CSS
data/sample_portfolio.csv
tests/                    pytest (schema, portfolio, scenarios, AppTest smoke)
.env.example  .gitignore  pyproject.toml  uv.lock
```

## 4. Data flow

1. Upload CSV (`st.file_uploader`, `.csv`) or enable "Use Sample Dataset".
2. `schema.validate` returns a clean DataFrame plus errors/warnings.
3. Sidebar filters (Asset Class multiselect, acquisition date range) narrow the frame.
4. `market.fetch_quotes(tickers)` returns price, previous close, 52-week high/low.
5. `portfolio.enrich` merges quotes: Current Value, Gain/Loss ($, %), Day Change ($, %).
6. Sidebar shock sliders feed `scenarios.apply_shocks` for simulated values.
7. KPIs, charts, table, AI tab, and export read from the enriched frame; all recompute on
   filter or dataset change.

## 5. Components

### 5.1 Schema (`schema.py`) — Story 2.1
- Required columns: `Ticker, Shares, Purchase_Price, Purchase_Date, Asset_Class`.
  Missing columns → `st.error` listing them; pipeline stops.
- Fully empty rows/columns are dropped silently. Tickers trimmed and uppercased.
- Shares ≤ 0, non-numeric price/shares, unparsable dates → row dropped, `st.warning` shows which.
- Fractional shares are valid.

### 5.2 Market data (`market.py`) — Stories 2.2, 5.1
- Uses `yfinance.Tickers`; returns current price, previous close, 52-week high/low.
- `@st.cache_data(ttl=300)` keyed by the sorted tuple of tickers.
- Ticker with no quote → marked "price unavailable", excluded from totals, warned.
- Total `yfinance` failure → `st.error`; rows fall back to purchase price, labeled stale.
  No silent zeros.

### 5.3 Portfolio math (`portfolio.py`) — Stories 2.2, 3.1, 3.2
- Per row: Current Value = shares × price; Gain/Loss $ = value − shares × purchase price;
  Gain/Loss % = gain / cost basis (guarded for zero cost basis); Day Change $ = shares ×
  (price − previous close).
- Aggregates: total value, total gain/loss $ and %, daily P&L $ and %, allocation by asset
  class, top holdings by value.

### 5.4 UI (`app.py`) — Stories 3.1, 3.2
- `st.metric` row: Total Portfolio Value (delta = total gain/loss $), Total P/L %, Daily P&L ($ / %).
- Plotly Express donut (allocation by asset class) and bar (top holdings by current value).
- `st.dataframe` with sortable columns and `$` / `%` column formatting.

### 5.5 Stress simulator (`scenarios.py`) — Story 5.2
- One sidebar slider per asset class present (−50% to +50%).
  `apply_shocks` scales each class's current value by (1 + pct).
- Side-by-side Original vs Simulated value with delta. Pure function; never mutates cached data.

### 5.6 AI (`ai.py`) — Stories 4.1, 4.2
- **Audit:** sends only summarized JSON (allocation % by class, top 3 positions, total ROI,
  position count), never raw rows. System prompt: Certified Financial Analyst persona,
  executive summary + bulleted action items. `@st.cache_data` keyed by the payload hash.
  Rendered in an expanded `st.expander` as Markdown, with a "not financial advice" note.
- **Chat:** `st.chat_input` / `st.chat_message`; history in `st.session_state`, capped to last
  N turns. System prompt is rebuilt on every call with the current filtered portfolio summary.
- Missing `OPENAI_API_KEY` → AI tab disabled with setup instructions; rest of the app works.
  API/rate-limit errors → retryable `st.error`, never a crash.

### 5.7 Export and print (`export.py`) — Story 5.3
- `st.download_button` for the enriched CSV (includes live values).
- "Print mode" toggle injects CSS hiding the sidebar and controls for browser print-to-PDF.

## 6. Security and config
- Secrets only via environment / `.env`; `.env` gitignored, `.env.example` committed.
- Only aggregated data is sent to OpenAI.

## 7. Testing
- pytest for `schema`, `portfolio`, `scenarios` using fixture frames. Cases: missing columns,
  negative shares, empty padding rows, ticker with no quote, zero cost basis, shocks of 0% and −100%.
- `yfinance` and OpenAI are mocked: tests are offline and deterministic.
- Streamlit `AppTest` smoke test runs `app.py` on the sample dataset.

## 8. Acceptance mapping
| Story | Satisfied by |
|---|---|
| 1.1 | `pyproject.toml`/`uv.lock`; `uv run streamlit run app.py` |
| 2.1 | §5.1 + uploader/sample toggle in §5.4 |
| 2.2 | §5.2, §5.3 |
| 3.1 / 3.2 | §5.4 |
| 4.1 / 4.2 | §5.6 |
| 5.1 | §5.2 (quotes), §5.6 (audit) caching |
| 5.2 | §5.5 |
| 5.3 | §5.7 |
