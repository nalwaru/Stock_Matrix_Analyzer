from types import SimpleNamespace as NS

import pytest

from agent.memo import filings_fact
from rag import inventory
from rag.retrieve import RetrievalError

URL = "https://www.sec.gov/Archives/edgar/data/320193/{}.htm"
IDS = ["AAPL-10-K-2025-10-31-Document-0", "AAPL-10-K-2025-10-31-Item-7-3", "AAPL-10-Q-2026-07-31-Document-0",
       "AAPL-8-K-2026-07-30-Document-0", "AAPL-weird-id"]


class FakeIndex:
    def __init__(self, ids=IDS, error=None):
        self.ids, self.error, self.fetched = ids, error, []

    def list(self, *, prefix, namespace):
        if self.error:
            raise self.error
        assert prefix == "AAPL-"
        yield [NS(id=i) for i in self.ids[:2]]
        yield [NS(id=i) for i in self.ids[2:]]

    def fetch(self, *, ids, namespace):
        self.fetched.append(list(ids))
        return NS(vectors={i: NS(metadata={"source_url": URL.format(i), "form": i.split("-")[1] + "-" + i.split("-")[2],
                                           "filing_date": "x"}) for i in ids})


def test_parse_id_reads_form_and_date_and_ignores_other_ids():
    assert inventory.parse_id("AAPL-10-K-2025-10-31-Document-0") == ("10-K", "2025-10-31")
    assert inventory.parse_id("AAPL-8-K-2026-07-30-Document-0") == ("8-K", "2026-07-30")
    assert inventory.parse_id("AAPL-weird-id") is None


def test_list_filings_is_unique_newest_first_with_urls():
    index = FakeIndex()
    found = inventory.list_filings(index, "aapl", "semantic")
    assert [(f.form, f.filing_date) for f in found] == [("10-Q", "2026-07-31"), ("8-K", "2026-07-30"), ("10-K", "2025-10-31")]
    assert all(f.ticker == "AAPL" and f.url.startswith("https://www.sec.gov") for f in found)
    assert len(index.fetched) == 1 and len(index.fetched[0]) == 3  # one record fetched per filing, in one call


def test_list_filings_empty_when_the_ticker_has_no_records():
    assert inventory.list_filings(FakeIndex(ids=[]), "AAPL", "semantic") == []


def test_list_filings_wraps_pinecone_errors():
    with pytest.raises(RetrievalError):
        inventory.list_filings(FakeIndex(error=RuntimeError("down")), "AAPL", "semantic")


def test_filing_fact_names_each_filing_newest_first_and_marks_the_latest_of_each_form():
    found = [inventory.FilingInfo("AAPL", "10-Q", "2026-07-31", "u1"), inventory.FilingInfo("AAPL", "10-K", "2025-10-31", "u2"),
             inventory.FilingInfo("AAPL", "10-Q", "2026-05-01", "u3")]
    fact = filings_fact("AAPL", found, "2026-10-18")
    assert fact.label == "AAPL filings, 2026-10-18, INDEX"
    assert "latest 10-K: filed 2025-10-31 (u2)" in fact.text and "latest 10-Q: filed 2026-07-31 (u1)" in fact.text
    assert "10-Q filed 2026-05-01 (u3)" in fact.text
