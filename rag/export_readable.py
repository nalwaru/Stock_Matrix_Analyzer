"""Write each downloaded filing as readable Markdown for manual checking:
`uv run python -m rag.export_readable` -> data/readable/<TICKER>/<form>-<date>.md"""
import re
from itertools import groupby
from pathlib import Path

from rag.ingest import FILINGS_DIR, Block, parse_filing

READABLE_DIR = FILINGS_DIR.parent / "readable"
_NAME = re.compile(r"^(?P<form>.+?)-(?P<date>\d{4}-\d{2}-\d{2})-")


def render_markdown(blocks: list[Block], title: str) -> str:
    lines = [f"# {title}", ""]
    for section, group in groupby(blocks, key=lambda b: b.section):
        lines += [f"## {section}", ""]
        for block in group:
            lines += [block.text, ""]
    return "\n".join(lines)


def export_all(src: Path = FILINGS_DIR, dest: Path = READABLE_DIR) -> list[Path]:
    written = []
    for path in sorted(src.glob("*/*.htm")):
        match = _NAME.match(path.stem)
        if not match:
            continue
        form, date = match["form"], match["date"]
        blocks = parse_filing(path.read_text(encoding="utf-8", errors="replace"))
        out = dest / path.parent.name / f"{form}-{date}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(blocks, f"{path.parent.name} {form}, {date}"), encoding="utf-8")
        written.append(out)
    return written


if __name__ == "__main__":
    files = export_all()
    print(f"wrote {len(files)} files to {READABLE_DIR}")
