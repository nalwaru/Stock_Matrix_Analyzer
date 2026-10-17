"""Pinecone dense and sparse indexes (integrated embedding), namespaced per chunking strategy."""
import time

from pinecone.errors.exceptions import RateLimitError

from rag.chunking import Chunk

DENSE_INDEX = "filings-dense"
SPARSE_INDEX = "filings-sparse"
DENSE_MODEL = "llama-text-embed-v2"
SPARSE_MODEL = "pinecone-sparse-english-v0"
RERANK_MODEL = "bge-reranker-v2-m3"
CLOUD, REGION = "aws", "us-east-1"
INDEXED_TICKERS = ["AAPL"]  # holdings with ingested filings
BATCH = 90
EMBED_BATCH = 96
RATE_LIMIT_WAIT = 30  # seconds per attempt; the embedding limit is per minute
MAX_ATTEMPTS = 8


def _retry(call):
    """Run call(); on a 429 wait for the per-minute window to reset, then retry."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return call()
        except RateLimitError:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(RATE_LIMIT_WAIT * attempt)


def ensure_indexes(pc) -> None:
    for name, model in ((DENSE_INDEX, DENSE_MODEL), (SPARSE_INDEX, SPARSE_MODEL)):
        if not pc.has_index(name):
            pc.create_index_for_model(
                name=name, cloud=CLOUD, region=REGION,
                embed={"model": model, "field_map": {"text": "text"}},
            )


def _record(chunk: Chunk) -> dict:
    return {"_id": chunk.id, "text": chunk.text, **chunk.metadata}


def upsert_chunks(pc, namespace: str, chunks: list[Chunk], batch: int = BATCH) -> int:
    """Upsert into both indexes; deterministic ids make re-runs overwrite, not duplicate."""
    records = [_record(c) for c in chunks]
    for name in (DENSE_INDEX, SPARSE_INDEX):
        index = pc.Index(name)
        for i in range(0, len(records), batch):
            part = records[i:i + batch]
            _retry(lambda: index.upsert_records(records=part, namespace=namespace))
    return len(records)


def stats(pc) -> dict[str, int]:
    out = {}
    for name in (DENSE_INDEX, SPARSE_INDEX):
        for namespace, info in pc.Index(name).describe_index_stats().namespaces.items():
            out[f"{name}/{namespace}"] = info.vector_count
    return out


def dense_embed(pc, texts: list[str]) -> list[list[float]]:
    """Embed passages with the dense model; used by the semantic chunker."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        part = texts[i:i + EMBED_BATCH]
        result = _retry(lambda: pc.inference.embed(
            model=DENSE_MODEL, inputs=part, parameters={"input_type": "passage"}
        ))
        vectors.extend(list(item["values"]) for item in result)
    return vectors
