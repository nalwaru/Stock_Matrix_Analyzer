"""Which filings are in the index, read from Pinecone itself (record ids and stored metadata)."""
import re
from dataclasses import dataclass

from rag.retrieve import RetrievalError

_ID = re.compile(r"^[A-Z.]+-(?P<form>10-K|10-Q|8-K)-(?P<date>\d{4}-\d{2}-\d{2})-")


@dataclass(frozen=True)
class FilingInfo:
    ticker: str
    form: str
    filing_date: str
    url: str


def parse_id(record_id: str) -> tuple[str, str] | None:
    """(form, filing date) from a chunk id like AAPL-10-K-2025-10-31-Document-0."""
    m = _ID.match(record_id)
    return (m["form"], m["date"]) if m else None


def list_filings(index, ticker: str, namespace: str) -> list[FilingInfo]:
    """Every distinct filing stored for the ticker, newest first, with its SEC link."""
    ticker = ticker.strip().upper()
    try:
        first: dict[tuple[str, str], str] = {}
        for page in index.list(prefix=f"{ticker}-", namespace=namespace):
            for item in page:
                key = parse_id(item.id)
                if key:
                    first.setdefault(key, item.id)
        if not first:
            return []
        records = index.fetch(ids=list(first.values()), namespace=namespace).vectors
        found = [
            FilingInfo(ticker, form, date, str(records[record_id].metadata.get("source_url", "")))
            for (form, date), record_id in first.items() if record_id in records
        ]
    except Exception as exc:  # noqa: BLE001 - any Pinecone/network failure degrades gracefully
        raise RetrievalError("Filing inventory is unavailable right now.") from exc
    return sorted(found, key=lambda f: f.filing_date, reverse=True)
