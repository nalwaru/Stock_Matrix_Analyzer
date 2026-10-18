from types import SimpleNamespace as NS

import pytest

from rag import retrieve, store


def hit(i, ticker="AAPL"):
    return {"_id": f"id{i}", "_score": 1.0 / (i + 1), "fields": {
        "text": f"text {i}", "ticker": ticker, "form": "10-K", "filing_date": "2025-11-01",
        "section": "Item 7", "source_url": "http://x"}}


class FakeIndex:
    def __init__(self, name, pc):
        self.name, self.pc = name, pc

    def search(self, *, namespace, query, fields=None):
        if self.pc.error:
            raise self.pc.error
        self.pc.searches.append((self.name, namespace, query))
        ids = self.pc.order[self.name]
        return {"result": {"hits": [hit(i) for i in ids]}}


class FakeInference:
    def __init__(self, pc):
        self.pc = pc

    def rerank(self, *, model, query, documents, rank_fields, top_n, return_documents, parameters=None):
        self.pc.reranks.append((model, len(documents), top_n))
        self.pc.rerank_parameters = parameters
        order = list(range(len(documents)))[::-1][:top_n]  # reverse = visibly reordered
        return NS(data=[NS(index=i, score=0.9 - 0.1 * n) for n, i in enumerate(order)])


class FakePinecone:
    def __init__(self, error=None):
        self.searches, self.reranks, self.error = [], [], error
        self.order = {store.DENSE_INDEX: [0, 1, 2], store.SPARSE_INDEX: [2, 3, 0]}
        self.inference = FakeInference(self)

    def Index(self, name):
        return FakeIndex(name, self)


def test_rrf_rewards_items_ranked_high_in_both_lists():
    fused = retrieve.rrf(["a", "b", "c"], ["c", "d", "a"])
    assert fused[:2] == ["a", "c"] and set(fused) == {"a", "b", "c", "d"}


def test_retrieve_filters_by_tickers_and_searches_both_indexes():
    pc = FakePinecone()
    retrieve.retrieve(pc, "why down?", ["AAPL", "MSFT"], "fixed", rerank=False)
    assert {name for name, _, _ in pc.searches} == {store.DENSE_INDEX, store.SPARSE_INDEX}
    for _, namespace, query in pc.searches:
        assert namespace == "fixed"
        assert query["filter"] == {"ticker": {"$in": ["AAPL", "MSFT"]}}
        assert query["inputs"] == {"text": "why down?"} and query["top_k"] == 30


def test_rerank_false_skips_rerank_call():
    pc = FakePinecone()
    out = retrieve.retrieve(pc, "q", ["AAPL"], "fixed", rerank=False, top_n=2)
    assert pc.reranks == [] and len(out) == 2 and out[0].id == "id0"


def test_rerank_true_reorders_by_reranker():
    pc = FakePinecone()
    out = retrieve.retrieve(pc, "q", ["AAPL"], "semantic", rerank=True, top_n=2)
    assert pc.reranks == [(store.RERANK_MODEL, 4, 2)]
    assert [p.id for p in out] == ["id3", "id1"] and out[0].score == 0.9
    assert out[0].ticker == "AAPL" and out[0].section == "Item 7"


def test_empty_tickers_returns_empty_without_calls():
    pc = FakePinecone()
    assert retrieve.retrieve(pc, "q", [], "fixed") == [] and pc.searches == []


def test_pinecone_error_becomes_retrieval_error():
    with pytest.raises(retrieve.RetrievalError):
        retrieve.retrieve(FakePinecone(error=RuntimeError("boom")), "q", ["AAPL"], "fixed")


def test_retrieve_reads_sdk_style_hit_objects():
    """The real Pinecone SDK returns objects with .id/.score/.fields, not dicts with _id."""
    def sdk_hit(i):
        fields = {"text": f"t{i}", "ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31",
                  "section": "Item 7", "source_url": "u"}
        return NS(id=f"id{i}", score=0.5, fields=fields)

    class SdkIndex:
        def search(self, *, namespace, query, fields=None):
            return NS(result=NS(hits=[sdk_hit(0), sdk_hit(1)]))

    class SdkPinecone(FakePinecone):
        def Index(self, name):
            return SdkIndex()

    out = retrieve.retrieve(SdkPinecone(), "q", ["AAPL"], "semantic", rerank=False, top_n=2)
    assert [p.id for p in out] == ["id0", "id1"] and out[0].ticker == "AAPL"


def test_rerank_truncates_oversized_passages_instead_of_failing():
    pc = FakePinecone()
    retrieve.retrieve(pc, "q", ["AAPL"], "semantic", rerank=True, top_n=2)
    assert pc.rerank_parameters == {"truncate": "END"}
