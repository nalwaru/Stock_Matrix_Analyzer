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
