"""Builds the production Deps: Pinecone retrieval, yfinance quotes, indexed tickers."""
from collections.abc import Callable

from agent.deps import Deps
from agent.prices import quotes_from_market
from rag import retrieve as rag_retrieve
from rag import store as rag_store


def make_deps(get_pinecone: Callable, namespace: str, pinecone_key: str, model: str = "",
              holdings=None, analyst_data=None, filings_dir=None) -> Deps:
    def retrieve(query: str, tickers: list[str]) -> list:
        if not pinecone_key:
            raise rag_retrieve.RetrievalError("Filing search is off (no PINECONE_API_KEY).")
        try:
            pc = get_pinecone()
        except Exception as exc:  # noqa: BLE001 - client construction must not break the tab
            raise rag_retrieve.RetrievalError("Filing search is unavailable right now.") from exc
        found = rag_retrieve.retrieve(pc, query, tickers, namespace)
        return [p for p in found if p.score >= rag_retrieve.MIN_RELEVANCE]

    from analyzer import ai

    return Deps(
        client=None,  # ask_structured builds a client with timeout and retry per call
        model=model or ai.get_model(),
        retrieve=retrieve,
        fetch_quotes=quotes_from_market,
        indexed_tickers=list(rag_store.INDEXED_TICKERS),
        holdings=holdings, analyst_data=analyst_data, filings_dir=filings_dir,
    )
