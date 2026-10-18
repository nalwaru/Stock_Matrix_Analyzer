# Tools and Analyst Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic tools to the agentic Ask AI graph: portfolio calculator, filings tools, price history, four ticker analysts (technical, risk, fundamental, ownership), and a number checker on the memo.

**Architecture:** Every tool returns `Fact(label, text)` objects computed in Python; the Synthesizer cites facts by label exactly as it cites passages and quotes. A planner node (one structured-output LLM call) picks portfolio or filing tool calls; analysts and price history are fixed tool nodes. The router gains an `analysis` route.

**Tech Stack:** Python >=3.14, LangGraph, pandas, yfinance (already installed), OpenAI structured outputs, pytest.

**Specs:** `docs/superpowers/specs/2026-10-17-agentic-ask-ai-design.md`; analysts design in `/Users/ssv/.claude/plans/agile-questing-gem.md` (approved 2026-10-17).

## Global Constraints
- LLM never calculates: every figure in a tool result is computed in Python; tools return formatted text with exact values.
- Fact labels use the existing citation shape `TICKER form, YYYY-MM-DD, section` (portfolio facts use ticker `PORTFOLIO`, e.g. `PORTFOLIO calc, 2026-10-17, weight_AAPL`); the existing `ai.grounded` check accepts only supplied labels.
- yfinance only; missing data becomes an `unavailable` fact, never a guess.
- At most 3 planner tool calls per question; at most 3 tickers per analysis.
- Memo wording: observations, never buy/sell advice; "should I buy" stays `out_of_scope`.
- Number checker flags, does not replace: unmatched figures are appended to the memo's `unverified` list.
- All existing tests stay green (update the ones whose graph shape changes).

## Review Focus
- Planner asks for a tool/ticker that does not exist (hallucinated args): skipped with a trace line, no crash.
- Local filing files missing (fresh checkout): filing tools skipped; filings path still works from retrieval.
- yfinance returns empty history or info: analysts yield `unavailable` facts; RSI with fewer than 15 closes is unavailable, not 0.
- A memo number that is a unit conversion of a supplied number (million vs billion) is not flagged; an unsupported number is.
- `analysis` question with no named ticker uses held tickers (capped at 3).

---

### Task 9: Facts, evidence plumbing, number checker, graph shape
**Files:** Modify `agent/state.py` (`Fact`, `facts` reducer), `agent/schemas.py` (`RouteDecision.analyses`, route `analysis`, `CalcCall`, `FilingCall`, `ToolPlan`), `agent/deps.py` (`holdings`, `analyst_data`, `filings_dir` optional fields), `agent/memo.py` (`labels_for` includes facts, `check_numbers`), `agent/nodes.py` (router prompt/route map, synthesizer evidence + number flag, `route_after_grade`/`route_after_tools`), `agent/graph.py` (nodes `tools`, `analysis`); update existing agent tests.
**Interfaces:** `Fact(label: str, text: str)`; `check_numbers(texts: list[str], evidence: str) -> list[str]` (unmatched figure strings; ignores citation brackets, dates, years, numbers under 3 digits without decimal/percent, and treats ×/÷ 1e3/1e6 variants as matches); `ToolPlan(calc_calls: list[CalcCall], filing_calls: list[FilingCall])`.
- [ ] Failing tests for `check_numbers` (match, mismatch, scaled match, ignored dates/years/citations), `labels_for` with facts, synthesizer evidence/no-evidence rule with facts, router mapping (`portfolio_only`→`tools`, `analysis`→`analysis`), `route_after_grade` → `tools`; run (FAIL); implement; run `uv run pytest -q` (PASS); commit.

### Task 10: Portfolio calculator tools + planner node
**Files:** Create `agent/tools/__init__.py`, `agent/tools/portfolio.py`, `agent/tools/planner.py`; Test `tests/test_agent_portfolio_tools.py`.
**Interfaces:** `run_calc(df: pd.DataFrame, call: CalcCall, today: str) -> Fact | None` with tools `holding`, `class_weight`, `top_n`, `total`, `stress` (reuse `analyzer.portfolio.summary/allocation`, `analyzer.scenarios.apply_shocks/compare`); `tools_node(state, deps) -> dict` (menu by route: `portfolio_only` → calc calls on `deps.holdings`; `sec_filings`/`both` → filing calls; otherwise no-op; planner = `ask_structured(ToolPlan)`; caps at 3; unknown ticker/class → skipped with trace).
- [ ] Failing tests per tool with a 3-row DataFrame (exact figures), unknown ticker skipped, planner failure yields no facts, cap of 3; implement; PASS; commit.

### Task 11: Filings tools
**Files:** Create `agent/tools/filings.py`; extend `agent/tools/planner.py`; Test `tests/test_agent_filing_tools.py`.
**Interfaces:** `get_section(ticker, form, section, filings_dir) -> Fact | None` (latest local filing of that form; section text capped 6000 chars; label `TICKER form, date, Item N`); `compare_periods(ticker, row_label, filings_dir) -> list[Fact]` (table header + first matching row from each of up to 4 local filings, newest first); `list_filings(ticker, filings_dir, today) -> Fact | None`; parses cached by path and mtime.
- [ ] Failing tests with tiny HTML fixtures in `tmp_path`; implement; PASS; commit.

### Task 12: Price history
**Files:** Create `agent/tools/history.py`; modify `agent/nodes.py` `price_node`; Test `tests/test_agent_history.py`.
**Interfaces:** `price_changes(tickers, now=None, fetch=None) -> list[Fact]` (1d, 1w=5 sessions, 1m=21 sessions % change with start/end close and dates; label `TICKER price history, date, 1w`); fetch failure or short history → `unavailable` fact.
- [ ] Failing tests with synthetic closes; implement; wire into `price_node`; PASS; commit.

### Task 13: Technical analyst
**Files:** Create `agent/analysts/__init__.py`, `agent/analysts/data.py` (`YahooData` adapter: `history`, `info`, `upgrades`, `holders`, `insiders`, `calendar`; all failures → empty), `agent/analysts/technical.py`; Test `tests/test_analyst_technical.py`.
**Interfaces:** `analyze_technical(ticker: str, history: pd.DataFrame, today: str) -> list[Fact]` — SMA20/50/200, price vs SMA50/200, cross status, RSI14 (Wilder), MACD(12,26,9), Bollinger(20,2) %b, ATR14, volume vs 20-day average, 52-week position, support/resistance (60-session low/high); metric names as label sections (`RSI14`, `SMA50`, ...).
- [ ] Failing tests with synthetic series (constant uptrend → RSI14 100; flat → SMA = price; short history → unavailable); implement with pandas only; PASS; commit.

### Task 14: Risk analyst
**Files:** Create `agent/analysts/risk.py`; Test `tests/test_analyst_risk.py`.
**Interfaces:** `analyze_risk(ticker, history, benchmark, others: dict[str, pd.DataFrame], today) -> list[Fact]` — annualized volatility, beta vs benchmark, max drawdown (1y), Sharpe (stated risk-free 4.0%), correlation with each other held ticker (max 5).
- [ ] Failing tests with synthetic series (identical series → beta 1, correlation 1; monotonic rise → drawdown 0); implement; PASS; commit.

### Task 15: Fundamental and ownership analysts
**Files:** Create `agent/analysts/fundamental.py`, `agent/analysts/ownership.py`; Test `tests/test_analyst_fundamental.py`, `tests/test_analyst_ownership.py`.
**Interfaces:** `analyze_fundamental(ticker, info: dict, today) -> list[Fact]` (P/E, forward P/E, P/S, EV/EBITDA, PEG, gross/operating/net margin, revenue/earnings growth, debt/equity, cash, free cash flow, dividend yield; missing → `unavailable` text); `analyze_ownership(ticker, info, upgrades: list[dict], holders: list[dict], insiders: list[dict], calendar: dict, today) -> list[Fact]` (consensus, price targets vs current price, last 5 rating changes, top 5 institutions, last 5 insider transactions, next earnings date).
- [ ] Failing tests with dict fixtures including missing keys; implement; PASS; commit.

### Task 16: Analysis node, app wiring, router cases, live check
**Files:** Modify `agent/nodes.py` (`analysis_node`), `agent/graph.py`, `agent/wiring.py` (holdings, `YahooData`, filings dir), `app.py` (pass enriched holdings), `agent/eval/router_cases.jsonl`, `README.md`; Test `tests/test_agent_analysis_node.py`, `tests/test_agent_graph.py`.
**Interfaces:** `analysis_node(state, deps) -> dict` (tickers from route else held, max 3; analyses from `route.analyses` else all four; appends facts and trace lines per analyst); graph path `router → analysis → synthesize`.
- [ ] Failing tests (fake `analyst_data`, each analyst selected correctly, held-ticker fallback, cap of 3, data failure → unavailable facts); implement; update router eval (≥3 `analysis` cases); full `uv run pytest -q`; live check: "Is NVDA overbought?", "What is AAPL's valuation vs analyst targets?", "How risky is RIVN?", "What percent of my portfolio is crypto?", "Compare Apple's net sales this quarter vs last year", "Should I buy NVDA?" (expect refusal); record in `docs/superpowers/specs/2026-10-17-agentic-ask-ai-results.md`; commit.
