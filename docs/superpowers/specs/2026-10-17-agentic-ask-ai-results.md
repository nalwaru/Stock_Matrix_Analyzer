# Agentic Ask AI: results of the first live checks

Date: 2026-10-17. Data: Apple only (5 filings, `semantic` namespace). Models: `gpt-4o-mini` for all nodes.

## Router evaluation (real OpenAI calls)
18 labelled questions written by Claude (at least 2 per route): accuracy 100% on the final run. This is a small, author-written set; it shows the router is sensible, not that it is robust. History: the first prompt scored 94% after I added cases, because "How is Apple exposed to China manufacturing?" was sent to `out_of_scope`; the prompt now reserves `out_of_scope` for advice, predictions, opinions and off-topic requests.

## End-to-end live run (real Pinecone, Yahoo, OpenAI), 5 questions
| Question | Outcome |
|---|---|
| Apple total net sales fiscal 2025 | Correct: $416,161 million, cited to the 10-K (matches the filing table). 1 of 5 passages kept by the grader. |
| How is Apple exposed to China manufacturing? | Answered from one graded 10-K passage with a citation. |
| What is AAPL trading at today? | Price from Yahoo, stamped 2026-10-17 22:25 UTC. |
| Why did Apple's stock fall last week? | Polite "I don't know" after one rewrite; the synthesizer's citation check rejected an unsupported answer. |
| Should I buy more Apple? | Polite "I don't know" without searching (out of scope). |

## Bugs the live run found (and fixed)
1. Real Pinecone hits expose `.id`, not `_id`; fakes had hidden this, so every search silently failed as "unavailable". Fixed with a test using SDK-shaped objects.
2. Router too eager to refuse company-risk questions (see above).

## Not yet verified
Combined `both` route live; portfolio-only questions live; answer correctness beyond these five; the retrieval comparison (fixed vs semantic, rerank vs not) from `rag/eval` has not been run; the reranker model call worked in this run but its effect is unmeasured.

## Tools and analysts: live run (2026-10-18 UTC, real Yahoo, Pinecone, OpenAI)
| Question | Outcome |
|---|---|
| What percent of my portfolio is in crypto? | Calculator fact `weight_Crypto`: 31.52%, cited. |
| Compare Apple's total net sales this quarter with last year | `compare_periods` returned 10-Q rows; memo quoted 111,184 vs 95,359 million with citations. An earlier run computed a difference (15,825) itself; the number checker flagged it, and the prompt now forbids computing new numbers. |
| Is NVDA overbought? | Technical analyst only; RSI(14) 37.56, cited. |
| AAPL valuation vs analyst targets | Fundamental + ownership; target mean 329.38 is 4.10% below the price 343.46, worded correctly after a fix. |
| How risky is RIVN? | Risk analyst: volatility 70.36%, beta 2.13, drawdowns, Sharpe 0.17, all cited. |
| Should I buy NVDA? | Polite refusal without searching. |

Router: 100% on 27 cases (small, author-written).

## Defects the live runs found (all fixed, each with a test)
1. `router_node` dropped the chosen `analyses`, so every analysis question ran all four analysts.
2. The synthesizer prompt's example label was copied into a real answer; the grounding check rejected it. Example removed.
3. Price-target fact used a signed percentage the model misread as the reverse; now worded "above/below".
4. A duplicated block in `agent/nodes.py` (my edit) overrode the real analysis node; found by tests, removed.
5. Float noise gave a Sharpe ratio of 158 trillion for a perfectly steady series; near-zero volatility is now "no variation".

## Still unverified
Analyst numbers were not compared with a charting site (do this for RSI, SMA50/200 and P/E); portfolio calculator was run on the sample portfolio only; the fixed-vs-semantic and rerank evaluation has still not been run; 8-K exhibits (earnings press releases) are not downloaded; yfinance is unofficial and its fields can change.

## Defects found from real use (2026-10-18)
- "technical analysis orcale" returned "I don't know" even though 13 ORCL facts and a chart existed: the router resolved the typo to ORCL, but the writer was never told, and its no-outside-knowledge rule made it refuse. The writer now gets a "tickers the question refers to" line. Fixed with a test.
- "technical analysis for oracle" (correct spelling) resolved to `O` (Realty Income, a held ticker): the router substituted a similar-looking held ticker. Router prompt now says to give the named company's real ticker; the router eval pins expected tickers (Oracle/"orcale" -> ORCL, Rivian -> RIVN, Nvidia -> NVDA).
- A chart was shown under an "I don't know" answer; charts are now hidden when the answer is a refusal.
- My earlier app test for agent charts asserted "at least one chart", which the Dashboard's own two charts always satisfied; it now counts exactly.
- Router eval: 31 cases, 100% on three consecutive runs after sharpening the analysis wording ("how risky", "is X expensive", "what do analysts think" are analysis). The router is not perfectly deterministic: borderline wording flipped between runs before the fix.
