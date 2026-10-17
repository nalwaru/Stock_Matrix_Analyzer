"""EDGAR access: pick a ticker's filings, download them, parse them into blocks."""
import os
import re
import time
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

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


@dataclass(frozen=True)
class Block:
    kind: Literal["text", "table"]
    section: str
    text: str


_ITEM = re.compile(r"^\s*item\s+(\d+[a-c]?)\b", re.IGNORECASE)
_BLOCK_TAGS = ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "table"]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _table_markdown(table, section: str) -> str | None:
    rows = []
    for tr in table.find_all("tr"):
        cells = [_clean(c.get_text(" ")) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c not in ("", "$", ")", "%")]
        if cells:
            rows.append(cells)
    if not rows:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
    caption = table.find("caption")
    title = _clean(caption.get_text(" ")) if caption else "Table"
    return f"[{section}] {title}\n\n" + "\n".join(lines)


def parse_filing(html: str) -> list[Block]:
    """Split filing HTML into text blocks and whole Markdown tables, tagged by Item section."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)  # EDGAR serves XHTML
        soup = BeautifulSoup(html or "", "lxml")
    blocks: list[Block] = []
    section = "Document"
    for el in soup.find_all(_BLOCK_TAGS):
        if el.find_parent("table") is not None:
            continue
        if el.name == "table":
            flat = _clean(el.get_text(" "))
            heading = _ITEM.match(flat)
            if heading and len(flat) < 200:
                section = f"Item {heading.group(1).upper()}"  # heading laid out as a table
                continue
            markdown = _table_markdown(el, section)
            if markdown:
                blocks.append(Block("table", section, markdown))
            continue
        if el.find(_BLOCK_TAGS):  # only leaf blocks carry text
            continue
        text = _clean(el.get_text(" "))
        if not text:
            continue
        heading = _ITEM.match(text)
        if heading and len(text) < 200:
            section = f"Item {heading.group(1).upper()}"
            continue
        blocks.append(Block("text", section, text))
    return blocks
