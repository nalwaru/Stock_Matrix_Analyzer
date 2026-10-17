"""Download, parse, chunk (both ways) and upsert filings: `uv run python -m rag.build_index`."""
import os
import sys

from dotenv import load_dotenv
from pinecone import Pinecone

from rag import chunking, ingest, store

DEFAULT_TICKERS = store.INDEXED_TICKERS


def main(tickers: list[str]) -> int:
    load_dotenv()
    if not os.getenv("PINECONE_API_KEY"):
        print("Set PINECONE_API_KEY in .env")
        return 1
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    try:
        store.ensure_indexes(pc)
        embed = lambda texts: store.dense_embed(pc, texts)  # noqa: E731
        totals = {"fixed": 0, "semantic": 0}
        for ticker in tickers:
            refs = ingest.list_filings(ticker)
            print(f"{ticker}: {len(refs)} filings")
            for ref in refs:
                blocks = ingest.parse_filing(ingest.download(ref).read_text(encoding="utf-8"))
                meta = {
                    "ticker": ref.ticker, "form": ref.form, "filing_date": ref.filing_date,
                    "fiscal_year": int(ref.filing_date[:4]), "source_url": ref.url,
                }
                totals["fixed"] += store.upsert_chunks(pc, "fixed", chunking.fixed_size(blocks, meta))
                totals["semantic"] += store.upsert_chunks(pc, "semantic", chunking.semantic(blocks, meta, embed))
    except ingest.IngestError as exc:
        print(exc)
        return 1
    print("chunks upserted:", totals)
    print("index stats:", store.stats(pc))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT_TICKERS))
