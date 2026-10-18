"""Builds the production Deps: Pinecone retrieval, yfinance quotes, indexed tickers."""
import os
from collections.abc import Callable

from openai import OpenAI

from agent.deps import Deps
from agent.prices import quotes_from_market
from rag import retrieve as rag_retrieve
from rag import store as rag_store


def make_deps(get_pinecone: Callable, namespace: str, pinecone_key: str, model: str = "",
              holdings=None, analyst_data=None) -> Deps:
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

    router_client, router_model = None, ""
    if os.getenv("OPENROUTER_API_KEY") and os.getenv("ROUTER_MODEL"):
        router_client = OpenAI(base_url="https://openrouter.ai/api/v1",
                               api_key=os.environ["OPENROUTER_API_KEY"], timeout=30.0, max_retries=1)
        router_model = os.environ["ROUTER_MODEL"]

    return Deps(
        client=None,  # ask_structured builds a client with timeout and retry per call
        model=model or ai.get_model(),
        retrieve=retrieve,
        fetch_quotes=quotes_from_market,
        indexed_tickers=list(rag_store.INDEXED_TICKERS),
        holdings=holdings, analyst_data=analyst_data,
        router_client=router_client, router_model=router_model,
    )
