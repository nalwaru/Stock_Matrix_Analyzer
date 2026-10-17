from rag.chunking import fixed_size, semantic
from rag.ingest import Block

META = {"ticker": "AAPL", "form": "10-K", "filing_date": "2025-11-01", "source_url": "http://x"}
WORDS = " ".join(f"w{i}" for i in range(1000))


def fake_embed(sentences):
    return [[1.0, 0.0] if "cat" in s else [0.0, 1.0] for s in sentences]


def test_tables_never_split_in_either_strategy():
    big_table = Block("table", "Item 8", "| a | b |\n" + "\n".join(f"| r{i} | {i} |" for i in range(1500)))
    for chunks in (fixed_size([big_table], META), semantic([big_table], META, fake_embed)):
        assert len(chunks) == 1
        assert chunks[0].text == big_table.text
        assert chunks[0].metadata["chunk_type"] == "table"


def test_fixed_chunks_overlap_by_50_tokens():
    chunks = fixed_size([Block("text", "Item 7", WORDS)], META)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert len(first) == 500
    assert second[:50] == first[-50:]


def test_semantic_splits_where_similarity_drops():
    text = "The cat sat. The cat purred. Stocks fell today. Stocks fell again."
    chunks = semantic([Block("text", "Item 7", text)], META, fake_embed, min_words=1)
    assert [c.text for c in chunks] == ["The cat sat. The cat purred.", "Stocks fell today. Stocks fell again."]


def test_chunk_ids_and_metadata():
    chunks = fixed_size([Block("text", "Item 1A", "hello world")], META)
    assert chunks[0].id == "AAPL-10-K-2025-11-01-Item-1A-0"
    assert chunks[0].metadata["section"] == "Item 1A"
    assert chunks[0].metadata["ticker"] == "AAPL"


def test_empty_blocks_give_no_chunks():
    assert fixed_size([], META) == []
    assert semantic([], META, fake_embed) == []


def test_text_does_not_merge_across_sections_or_tables():
    blocks = [Block("text", "Item 1", "alpha"), Block("table", "Item 1", "| t |"), Block("text", "Item 2", "beta")]
    assert [c.text for c in fixed_size(blocks, META)] == ["alpha", "| t |", "beta"]


def test_semantic_does_not_split_before_min_words():
    text = "The cat sat. The cat purred. Stocks fell today. Stocks fell again."
    chunks = semantic([Block("text", "Item 7", text)], META, fake_embed, min_words=10)
    assert len(chunks) == 1 and chunks[0].text == text


def test_semantic_splits_on_drop_once_min_words_reached():
    cats = " ".join(f"The cat number {i} sat quietly." for i in range(5))  # 30 words
    text = cats + " Stocks fell today."
    chunks = semantic([Block("text", "Item 7", text)], META, fake_embed, min_words=20)
    assert [len(c.text.split()) for c in chunks] == [30, 3]
