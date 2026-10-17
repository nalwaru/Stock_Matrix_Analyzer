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
