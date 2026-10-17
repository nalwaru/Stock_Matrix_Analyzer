"""Pinecone dense and sparse indexes (integrated embedding), namespaced per chunking strategy."""
from rag.chunking import Chunk

DENSE_INDEX = "filings-dense"
SPARSE_INDEX = "filings-sparse"
DENSE_MODEL = "llama-text-embed-v2"
SPARSE_MODEL = "pinecone-sparse-english-v0"
RERANK_MODEL = "bge-reranker-v2-m3"
CLOUD, REGION = "aws", "us-east-1"
INDEXED_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]  # holdings with ingested filings
BATCH = 90
EMBED_BATCH = 96


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
            index.upsert_records(records=records[i:i + batch], namespace=namespace)
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
        result = pc.inference.embed(
            model=DENSE_MODEL, inputs=texts[i:i + EMBED_BATCH], parameters={"input_type": "passage"}
        )
        vectors.extend(list(item["values"]) for item in result)
    return vectors
