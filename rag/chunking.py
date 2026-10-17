"""Two chunking strategies. Tables are always emitted whole; text never crosses a section."""
import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from rag.ingest import Block

SIZE = 500  # tokens, counted as whitespace-separated words
OVERLAP = 50
THRESHOLD = 0.6
MIN_WORDS = 100  # never split on a similarity drop before this many words
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict


def _make(meta: dict, section: str, kind: str, n: int, text: str) -> Chunk:
    slug = re.sub(r"\W+", "-", section).strip("-") or "Document"
    chunk_id = f"{meta['ticker']}-{meta['form']}-{meta['filing_date']}-{slug}-{n}"
    return Chunk(chunk_id, text, {**meta, "section": section, "chunk_type": kind})


def _runs(blocks: list[Block]) -> Iterator[tuple[str, str, str]]:
    """Yield (kind, section, text); consecutive text blocks of one section are joined."""
    buffer: list[str] = []
    section = ""
    for block in blocks:
        if buffer and (block.kind == "table" or block.section != section):
            yield "text", section, "\n".join(buffer)
            buffer = []
        if block.kind == "table":
            yield "table", block.section, block.text
        else:
            section = block.section
            buffer.append(block.text)
    if buffer:
        yield "text", section, "\n".join(buffer)


def _assemble(blocks: list[Block], meta: dict, split_text: Callable[[str], list[str]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    counts: dict[str, int] = {}
    for kind, section, text in _runs(blocks):
        pieces = [text] if kind == "table" else split_text(text)
        for piece in pieces:
            n = counts.get(section, 0)
            counts[section] = n + 1
            chunks.append(_make(meta, section, kind, n, piece))
    return chunks


def fixed_size(blocks: list[Block], meta: dict, size: int = SIZE, overlap: int = OVERLAP) -> list[Chunk]:
    def split(text: str) -> list[str]:
        words = text.split()
        step = max(1, size - overlap)
        return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)]

    return _assemble(blocks, meta, split)


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.hypot(*a) * math.hypot(*b)
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


def semantic(
    blocks: list[Block],
    meta: dict,
    embed: Callable[[list[str]], list[list[float]]],
    threshold: float = THRESHOLD,
    min_words: int = MIN_WORDS,
    max_words: int = 2 * SIZE,
) -> list[Chunk]:
    """Start a new chunk when a sentence is not similar enough to the chunk's running mean
        (only once the chunk has min_words) or when it would exceed max_words."""

    def split(text: str) -> list[str]:
        sentences = [s for s in _SENTENCE.split(text) if s.strip()]
        if not sentences:
            return []
        vectors = embed(sentences)
        chunks, current, centroid, words = [], [sentences[0]], list(vectors[0]), len(sentences[0].split())
        for sentence, vector in zip(sentences[1:], vectors[1:]):
            length = len(sentence.split())
            drop = _cosine(centroid, vector) < threshold and words >= min_words
            if drop or words + length > max_words:
                chunks.append(" ".join(current))
                current, centroid, words = [sentence], list(vector), length
            else:
                centroid = [c + v for c, v in zip(centroid, vector)]
                current.append(sentence)
                words += length
        chunks.append(" ".join(current))
        return chunks

    return _assemble(blocks, meta, split)
