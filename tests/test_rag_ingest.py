from datetime import date

import pytest

from rag import ingest

TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}


def submissions(rows):
    cols = ["form", "filingDate", "accessionNumber", "primaryDocument"]
    return {"filings": {"recent": {c: [r[i] for r in rows] for i, c in enumerate(cols)}}}


ROWS = [
    ("8-K", "2026-10-01", "0000320193-26-000010", "a8k1.htm"),
    ("8-K", "2026-01-02", "0000320193-26-000001", "old8k.htm"),
    ("10-Q", "2026-08-01", "0000320193-26-000008", "q3.htm"),
    ("10-Q", "2026-05-01", "0000320193-26-000005", "q2.htm"),
    ("10-Q", "2026-02-01", "0000320193-26-000002", "q1.htm"),
    ("10-Q", "2025-08-01", "0000320193-25-000008", "q3old.htm"),
    ("10-K", "2025-11-01", "0000320193-25-000099", "k.htm"),
    ("10-K", "2024-11-01", "0000320193-24-000099", "kold.htm"),
    ("4", "2026-10-02", "0000320193-26-000011", "form4.htm"),
]


class FakeResponse:
    def __init__(self, payload=None, text=""):
        self._payload, self.text, self.content = payload, text, text.encode()

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, rows=ROWS):
        self.urls, self.headers, self.rows = [], {}, rows

    def get(self, url, **kwargs):
        self.urls.append(url)
        if url.endswith("company_tickers.json"):
            return FakeResponse(TICKERS)
        if "submissions/CIK" in url:
            return FakeResponse(submissions(self.rows))
        return FakeResponse(text="<html>filing</html>")


@pytest.fixture(autouse=True)
def user_agent(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    ingest._CIKS.clear()


def test_list_filings_selects_latest_10k_three_10qs_recent_8ks():
    refs = ingest.list_filings("AAPL", FakeSession(), today=date(2026, 10, 17))
    assert [r.form for r in refs].count("10-K") == 1
    assert [r.url.rsplit("/", 1)[1] for r in refs if r.form == "10-K"] == ["k.htm"]
    assert [r.url.rsplit("/", 1)[1] for r in refs if r.form == "10-Q"] == ["q3.htm", "q2.htm", "q1.htm"]
    assert [r.url.rsplit("/", 1)[1] for r in refs if r.form == "8-K"] == ["a8k1.htm"]
    assert "form4.htm" not in " ".join(r.url for r in refs)


def test_ticker_to_cik_pads_to_ten_digits():
    session = FakeSession()
    ingest.list_filings("AAPL", session, today=date(2026, 10, 17))
    assert any("CIK0000320193.json" in u for u in session.urls)


def test_filing_url_uses_accession_without_dashes():
    refs = ingest.list_filings("AAPL", FakeSession(), today=date(2026, 10, 17))
    k = next(r for r in refs if r.form == "10-K")
    assert k.url == "https://www.sec.gov/Archives/edgar/data/320193/000032019325000099/k.htm"


def test_missing_user_agent_raises_ingest_error(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(ingest.IngestError, match="SEC_USER_AGENT"):
        ingest.list_filings("AAPL", FakeSession())


def test_unknown_ticker_returns_empty_list():
    assert ingest.list_filings("VOO", FakeSession(), today=date(2026, 10, 17)) == []
