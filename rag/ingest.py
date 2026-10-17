"""EDGAR access: pick a ticker's filings, download them, parse them into blocks."""
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import requests

FILINGS_DIR = Path(__file__).parent.parent / "data" / "filings"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
RECENT_8K_DAYS = 90
MAX_10Q = 3
MIN_INTERVAL = 0.11  # EDGAR allows 10 requests/second

_CIKS: dict[str, int] = {}
_last_request = 0.0


class IngestError(RuntimeError):
    """Any ingest failure, safe to show to the user."""


@dataclass(frozen=True)
class FilingRef:
    ticker: str
    form: str
    filing_date: str
    accession: str
    url: str


def _session(session=None):
    agent = os.getenv("SEC_USER_AGENT", "").strip()
    if not agent:
        raise IngestError("Set SEC_USER_AGENT in .env (your name and email); EDGAR requires it.")
    if session is None:
        session = requests.Session()
    session.headers.update({"User-Agent": agent})
    return session


def _get(session, url):
    global _last_request
    wait = MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise IngestError(f"EDGAR request failed for {url}: {exc}") from None
    return response


def _cik(ticker: str, session) -> int | None:
    if not _CIKS:
        for row in _get(session, TICKERS_URL).json().values():
            _CIKS[row["ticker"].upper()] = int(row["cik_str"])
    return _CIKS.get(ticker.upper())


def list_filings(ticker: str, session=None, today: date | None = None) -> list[FilingRef]:
    """Latest 10-K, last 3 10-Qs and 8-Ks from the last 90 days, newest first per form."""
    session = _session(session)
    cik = _cik(ticker, session)
    if cik is None:
        return []
    recent = _get(session, SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
    rows = sorted(
        zip(recent["form"], recent["filingDate"], recent["accessionNumber"], recent["primaryDocument"]),
        key=lambda r: r[1],
        reverse=True,
    )
    cutoff = ((today or date.today()) - timedelta(days=RECENT_8K_DAYS)).isoformat()
    limits = {"10-K": 1, "10-Q": MAX_10Q}
    taken: dict[str, int] = {}
    refs = []
    for form, filed, accession, document in rows:
        if form == "8-K":
            if filed < cutoff:
                continue
        elif form in limits:
            if taken.get(form, 0) >= limits[form]:
                continue
        else:
            continue
        taken[form] = taken.get(form, 0) + 1
        url = ARCHIVE_URL.format(cik=cik, accession=accession.replace("-", ""), document=document)
        refs.append(FilingRef(ticker.upper(), form, filed, accession, url))
    return refs


def download(ref: FilingRef, session=None) -> Path:
    """Fetch a filing's HTML into data/filings/, skipping files already cached."""
    path = FILINGS_DIR / ref.ticker / f"{ref.form}-{ref.filing_date}-{ref.accession}.htm"
    if path.exists():
        return path
    html = _get(_session(session), ref.url).text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
