"""Lookups in locally downloaded SEC filings (data/filings). Exact text, never summarised."""
import re
from functools import lru_cache
from pathlib import Path

from agent.state import Fact
from rag.ingest import Block, parse_filing

MAX_SECTION_CHARS = 6000
MAX_COMPARE_FILINGS = 4
_NAME = re.compile(r"^(?P<form>.+?)-(?P<date>\d{4}-\d{2}-\d{2})-")


@lru_cache(maxsize=32)
def _blocks(path: str, mtime: float) -> tuple[Block, ...]:
    return tuple(parse_filing(Path(path).read_text(encoding="utf-8", errors="replace")))


def _filings(ticker: str, filings_dir) -> list[tuple[str, str, Path]]:
    """(form, date, path) for the ticker's 10-K and 10-Q files, newest first."""
    folder = Path(filings_dir) / ticker.strip().upper()
    out = []
    for path in folder.glob("*.htm") if folder.is_dir() else []:
        m = _NAME.match(path.stem)
        if m and m["form"] in ("10-K", "10-Q", "8-K"):
            out.append((m["form"], m["date"], path))
    return sorted(out, key=lambda f: f[1], reverse=True)


def _load(path: Path) -> tuple[Block, ...]:
    return _blocks(str(path), path.stat().st_mtime)


def _item(section: str) -> str:
    m = re.search(r"(\d+)\s*([a-cA-C])?", section or "")
    return f"Item {m.group(1)}{(m.group(2) or '').upper()}" if m else ""


def get_section(ticker: str, form: str, section: str, filings_dir) -> Fact | None:
    item = _item(section)
    for f_form, date, path in _filings(ticker, filings_dir):
        if f_form != form or not item:
            continue
        body = "\n\n".join(b.text for b in _load(path) if b.section == item)
        if not body:
            return None
        if len(body) > MAX_SECTION_CHARS:
            body = body[:MAX_SECTION_CHARS] + "\n[truncated]"
        return Fact(f"{ticker.strip().upper()} {form}, {date}, {item}", body)
    return None


def _first_row(table: str, label: str) -> tuple[str, str] | None:
    lines = [ln for ln in table.splitlines() if ln.startswith("|")]
    if len(lines) < 3:
        return None
    header = lines[0]
    want = label.strip().lower()
    rows = [(ln, ln.strip("|").split("|")[0].strip().lower()) for ln in lines[2:]]
    for ln, first in rows:
        if first == want:
            return header, ln
    for ln, first in rows:
        if want in first:
            return header, ln
    return None


def compare_periods(ticker: str, row_label: str, filings_dir) -> list[Fact]:
    facts = []
    for form, date, path in _filings(ticker, filings_dir)[:MAX_COMPARE_FILINGS]:
        for block in _load(path):
            hit = _first_row(block.text, row_label) if block.kind == "table" else None
            if hit:
                facts.append(Fact(f"{ticker.strip().upper()} {form}, {date}, {block.section}",
                                  f"{hit[0]}\n{hit[1]}"))
                break
    return facts


def list_filings(ticker: str, filings_dir, today: str) -> Fact | None:
    found = _filings(ticker, filings_dir)
    if not found:
        return None
    return Fact(f"{ticker.strip().upper()} filings, {today}, index",
                "; ".join(f"{form} {date}" for form, date, _ in found))
