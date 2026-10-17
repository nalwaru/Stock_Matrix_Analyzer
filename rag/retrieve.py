"""Hybrid retrieval: dense + sparse search, reciprocal rank fusion, hosted rerank."""
from dataclasses import dataclass

from rag import store

CANDIDATES = 30
TOP_N = 5
RRF_K = 60
FIELDS = ["text", "ticker", "form", "filing_date", "section", "source_url"]


class RetrievalError(RuntimeError):
    """Retrieval failed; safe to show. Callers fall back to portfolio-only answers."""


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    ticker: str
    form: str
    filing_date: str
    section: str
    source_url: str
    score: float


def rrf(dense: list[str], sparse: list[str], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in (dense, sparse):
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item: (-scores[item], item))


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _hits(pc, index_name: str, namespace: str, query: str, tickers: list[str], top_k: int) -> list:
    response = pc.Index(index_name).search(
        namespace=namespace,
        query={"inputs": {"text": query}, "top_k": top_k, "filter": {"ticker": {"$in": tickers}}},
        fields=FIELDS,
    )
    return _get(_get(response, "result"), "hits")


def _passage(hit, score: float) -> Passage:
    fields = _get(hit, "fields")
    return Passage(
        id=_get(hit, "_id"), score=score,
        **{name: str(_get(fields, name)) for name in FIELDS[:1] + FIELDS[1:]},
    )


def retrieve(
    pc, query: str, tickers: list[str], namespace: str,
    rerank: bool = True, top_n: int = TOP_N, candidates: int = CANDIDATES,
) -> list[Passage]:
    if not tickers:
        return []
    try:
        by_id, rankings = {}, []
        for index_name in (store.DENSE_INDEX, store.SPARSE_INDEX):
            hits = _hits(pc, index_name, namespace, query, tickers, candidates)
            rankings.append([_get(h, "_id") for h in hits])
            for h in hits:
                by_id.setdefault(_get(h, "_id"), h)
        merged = rrf(*rankings)[:candidates]
        if not rerank:
            return [_passage(by_id[i], 1.0 / (RRF_K + n + 1)) for n, i in enumerate(merged[:top_n])]
        docs = [{"id": i, "text": str(_get(_get(by_id[i], "fields"), "text"))} for i in merged]
        result = pc.inference.rerank(
            model=store.RERANK_MODEL, query=query, documents=docs,
            rank_fields=["text"], top_n=top_n, return_documents=False,
        )
        return [_passage(by_id[merged[_get(r, "index")]], float(_get(r, "score"))) for r in _get(result, "data")]
    except Exception as exc:  # noqa: BLE001 - any Pinecone/network failure degrades gracefully
        raise RetrievalError("Filing search is unavailable right now.") from exc
