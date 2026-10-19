# Ask AI: quick start and what changed

Ask AI is the chat tab of the Stock Portfolio Analyzer. You ask a question in plain English and it
answers from four sources only: your uploaded holdings, SEC filings stored in Pinecone, analyst
data from Yahoo Finance, and live prices. Every figure carries a citation. When the sources don't
support an answer, it says "I'm sorry, I don't know" instead of guessing.

## 1. Prerequisites

| Need | Why | Required? |
|---|---|---|
| Python 3.14 | The project pins it (`.python-version`) | Yes |
| [uv](https://docs.astral.sh/uv/) | Installs dependencies and runs everything | Yes |
| OpenAI API key | All agents except the gate run on OpenAI (`gpt-4o-mini`) | Yes: Ask AI is disabled without it |
| Pinecone API key | Holds the SEC filing indexes used for filing answers | Yes, for filing questions |
| OpenRouter API key + credits | Runs the gate (router) on `typesafe/jev-router` | Optional: without it the gate uses OpenAI |
| SEC user agent (name + email) | EDGAR requires it to download filings | Only to build or extend the index |

## 2. Run it in five minutes

```bash
git clone https://github.com/nalwaru/Stock_Matrix_Analyzer.git
cd Stock_Matrix_Analyzer
git checkout feature/portfolio-analyzer
uv sync                      # installs the dependencies below
cp .env.example .env         # then fill in the keys (section 3)
uv run streamlit run app.py  # opens http://localhost:8501
```

In the app, switch on **Use Sample Dataset** in the sidebar, open the **Ask AI** tab and try:

- `Which holding lost the most? How concentrated am I in crypto?` (your holdings)
- `What are Apple's main risk factors?` (SEC filings)
- `What is Apple's profit? Do you expect growth next year?` (filings plus an outlook)
- `Is NVDA overbought?` (technical analysis with a chart)
- `Should I buy Tesla?` (politely declined)

## 3. Settings (`.env`)

| Variable | Example | Notes |
|---|---|---|
| `OPENAI_API_KEY` | `sk-...` | Required |
| `OPENAI_MODEL` | `gpt-4o-mini` | Default if unset |
| `PINECONE_API_KEY` | `pcsk_...` | Required for filing answers |
| `RAG_NAMESPACE` | `semantic` | Which chunk set to search (`semantic` or `fixed`) |
| `OPENROUTER_API_KEY` | `sk-or-...` | Optional; needs credits on openrouter.ai |
| `ROUTER_MODEL` | `typesafe/jev-router` | Set together with the OpenRouter key |
| `SEC_USER_AGENT` | `Your Name you@example.com` | Only for `rag.build_index` |

Never commit `.env`; it is already in `.gitignore`.

## 4. The SEC filing index (Pinecone)

Two indexes, `filings-dense` (semantic search) and `filings-sparse` (keyword search), hold the same
chunks of Apple (AAPL) and Rivian (RIVN) filings: the latest 10-K, the last three 10-Qs and recent
8-Ks. If your Pinecone project doesn't have them yet, build them once:

```bash
uv run python -m rag.build_index AAPL RIVN
```

This downloads the filings from SEC EDGAR, chunks them and uploads them. It takes several minutes
and uses Pinecone embedding quota. To add another company, run it with that ticker and add the
ticker to `INDEXED_TICKERS` in `rag/store.py`.

## 5. Dependencies

From `pyproject.toml`: `streamlit` (UI), `langgraph` (agent workflow), `openai` (LLM calls),
`pinecone` (vector database), `yfinance` (prices and analyst data), `pandas`, `plotly` (charts),
`python-dotenv`, `requests`, `beautifulsoup4` and `lxml` (reading filings). Tests use `pytest`.

## 6. How a question is answered

```
Question → Gate (classifier) → one of four paths → Writer → Guardrails → Reply
```

| Path | When | What runs |
|---|---|---|
| Refuse | Advice, price predictions, news, unknown companies | Polite "I don't know", plus what was not answered |
| Live price | "What's Apple's price?" | Price lookup, then the writer |
| Portfolio | Questions about your holdings | Planner picks calculator calls, then the writer |
| Agent | Everything else | Tool-calling agent: filings search (RAG), analysts, prices, filing list, calculator |

Six LLM agents take part: the **gate**, the **planner**, the **tool-calling agent**, the
**grader** and **rewriter** (which, with the non-LLM retriever, make up the RAG agent), and the
**writer**. All numbers are computed in Python; the LLMs only choose tools and write text.

## 7. What changed (summary)

**Routing and agents**
- A gate (classifier) picks the path for each question. It can run on OpenRouter's Jev, retries once on failure and has no fallback to OpenAI.
- One tool-calling agent replaced the fixed filing, analysis and mixed paths, so a single question can use several sources.
- Filing questions always search the filings first. The analysts are used only if that search finds nothing, except for outlook questions.
- Questions that say "my" and name a stock you hold always look up that holding.
- Company names with no real ticker (for example "Vivian") are refused in code, so they can never borrow your portfolio's numbers.
- News requests are declined clearly: the app has no news source.

**Filings (RAG)**
- Hybrid keyword and semantic search, merged and reranked, with a 0.2 relevance cutoff, then an LLM grader and one query rewrite.
- Rivian's filings were added next to Apple's (4,395 chunks in each index).
- A filing list answers "what is the latest 10-K" from the index's metadata.
- Fixed: large chunks used to make the reranker fail the whole search.

**Answers**
- Outlook questions get an Outlook section: a leaning plus cited reasons built only from analyst data, ending with "I'm not a financial advisor".
- A question with an unanswerable part answers the rest and says which part it did not answer.
- Technical analysis and its chart appear only when you ask for them.
- Replies use plain body text with bold labels, and empty sections are hidden.
- More reliable writer: it asks the model for compact JSON (this stopped a frequent runaway), retries once, and the trace says why an answer was rejected.

**Interface**
- New avatars: a dollar-eyes face for you, and a light-yellow robot for the assistant that transmits signal rings while it works.
- One plain "Working… (last step: …)" line while working, with the full steps in the **Agent trace** expander.
- The separate "Generate AI Audit" button was removed.

## 8. Tests and quality checks

```bash
uv run pytest                                          # 364 offline tests, no keys needed
uv run python -m agent.eval.run_router_eval            # gate accuracy on labelled questions (live)
uv run python -m agent.eval.run_agent_eval             # end-to-end answer checks (live)
PYTHONPATH=. uv run python -m agent.eval.profile_latency  # time per step (live)
```

The live checks call OpenAI, OpenRouter and Pinecone, so they cost a little and need the keys.
The labelled questions are in `agent/eval/router_cases.jsonl` and `agent/eval/agent_cases.jsonl`.

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "Ask AI is disabled" | No `OPENAI_API_KEY` | Add it to `.env` and restart |
| Every answer comes from portfolio data, trace says "routing failed" | OpenRouter out of credits (HTTP 402) or down | Add credits, or clear `OPENROUTER_API_KEY` to use OpenAI for the gate |
| "No SEC filings are available for X" | Only AAPL and RIVN are indexed | Index that ticker (section 4) |
| Filing answers are "I don't know" | Pinecone key missing, or the indexes are not built | Check `PINECONE_API_KEY`; run `rag.build_index` |
| Answers feel slow | Agent questions take about 10–17 s | Expected; price and portfolio questions are faster |

Ask AI is educational analysis, not financial advice.
