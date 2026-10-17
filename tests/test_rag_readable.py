from rag.export_readable import export_all, render_markdown
from rag.ingest import Block


def test_render_markdown_groups_blocks_under_section_headings():
    blocks = [
        Block("text", "Item 1A", "Risk one."),
        Block("table", "Item 1A", "[Item 1A] Table\n\n| a | b |\n| --- | --- |\n| 1 | 2 |"),
        Block("text", "Item 7", "Money."),
    ]
    md = render_markdown(blocks, "AAPL 10-K, 2025-11-01")
    assert md.startswith("# AAPL 10-K, 2025-11-01")
    assert md.index("## Item 1A") < md.index("Risk one.") < md.index("| a | b |") < md.index("## Item 7")
    assert md.count("## Item 1A") == 1


def test_export_all_writes_one_markdown_per_filing(tmp_path):
    src = tmp_path / "filings" / "AAPL"
    src.mkdir(parents=True)
    (src / "10-K-2025-11-01-0000320193-25-000099.htm").write_text("<html><body><p>Hello filing.</p></body></html>")
    out = tmp_path / "readable"
    paths = export_all(tmp_path / "filings", out)
    assert [p.relative_to(out).as_posix() for p in paths] == ["AAPL/10-K-2025-11-01.md"]
    assert "Hello filing." in paths[0].read_text()
