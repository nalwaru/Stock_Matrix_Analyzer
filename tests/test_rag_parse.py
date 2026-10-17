from rag.ingest import Block, parse_filing

TABLE_HTML = """
<html><body>
<p>Item 7. Management's Discussion</p>
<p>Revenue grew strongly.</p>
<table><caption>Net sales by segment</caption>
<tr><th>Segment</th><th>2025</th></tr>
<tr><td>Americas</td><td>$100</td></tr>
</table>
</body></html>
"""


def test_table_becomes_single_markdown_block_with_section():
    tables = [b for b in parse_filing(TABLE_HTML) if b.kind == "table"]
    assert len(tables) == 1
    assert tables[0].section == "Item 7"
    assert "Net sales by segment" in tables[0].text
    assert "| Segment | 2025 |" in tables[0].text
    assert "| Americas | $100 |" in tables[0].text


def test_text_split_by_item_headings():
    html = "<html><body><p>Item 1A. Risk Factors</p><p>Risk one.</p><p>Item 7. MD&A</p><p>Money.</p></body></html>"
    texts = [(b.section, b.text) for b in parse_filing(html) if b.kind == "text"]
    assert ("Item 1A", "Risk one.") in texts
    assert ("Item 7", "Money.") in texts


def test_no_headings_or_tables_still_returns_blocks():
    blocks = parse_filing("<html><body><p>Plain text only.</p></body></html>")
    assert blocks == [Block("text", "Document", "Plain text only.")]


def test_layout_only_table_is_dropped_and_empty_html_is_empty():
    assert parse_filing("<table><tr><td></td></tr></table>") == []
    assert parse_filing("") == []
