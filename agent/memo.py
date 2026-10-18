"""Memo helpers: the polite refusal memo, citable labels, Markdown rendering."""
import re

from agent.schemas import Memo
from analyzer import ai


def idk_memo() -> Memo:
    return Memo(answer=ai.IDK_MESSAGE, key_figures=[], sources=[], unverified=[])


def passage_label(p) -> str:
    return f"{p.ticker} {p.form}, {p.filing_date}, {p.section}"


def quote_label(q) -> str:
    return f"{q.ticker} live quote, {q.as_of}"


def labels_for(state) -> set[str]:
    labels = {passage_label(p) for p in state.get("graded", [])}
    labels |= {f.label for f in state.get("facts", [])}
    labels |= {quote_label(q) for q in state.get("quotes", {}).values() if q.price is not None}
    return labels


def render_memo(memo: Memo) -> str:
    if memo.answer == ai.IDK_MESSAGE:
        return ai.IDK_MESSAGE

    def section(title: str, items: list[str]) -> str:
        body = "\n".join(f"- {i}" for i in items) if items else "- None"
        return f"### {title}\n{body}"

    return "\n\n".join([
        f"### Answer\n{memo.answer}",
        section("Key figures", memo.key_figures),
        section("Sources", memo.sources),
        section("What I couldn't verify", memo.unverified),
    ])


_CITATION_OR_DATE = re.compile(r"\[[^\]]*\]|\d{4}-\d{2}-\d{2}")
_NUMBER = re.compile(r"(?<![\w.])\$?-?\d[\d,]*(?:\.\d+)?%?")
_SCALES = (1, 1e3, 1e-3, 1e6, 1e-6, 100, 1e-2)


def _numbers(text: str) -> list[tuple[str, float, bool]]:
    """(token, value, significant) for every number outside citations and dates."""
    found = []
    for token in _NUMBER.findall(_CITATION_OR_DATE.sub(" ", text)):
        clean = token.strip("$%").replace(",", "")
        try:
            value = float(clean)
        except ValueError:
            continue
        whole = clean.lstrip("-").split(".")[0]
        decimal, percent = "." in clean, token.endswith("%")
        is_year = not decimal and not percent and 1900 <= value <= 2100
        found.append((token, value, (decimal or percent or len(whole) >= 3) and not is_year))
    return found


def check_numbers(texts: list[str], evidence: str) -> list[str]:
    """Figures in `texts` that appear nowhere in `evidence` (exact, or off by a unit scale)."""
    known = {round(v, 6) for _, v, _ in _numbers(evidence)}
    missing: list[str] = []
    for text in texts:
        for token, value, significant in _numbers(text):
            if not significant or token in missing:
                continue
            if not any(round(value * scale, 6) in known or round(abs(value) * scale, 6) in known
                       for scale in _SCALES):
                missing.append(token)
    return missing


def markdown_safe(text: str) -> str:
    """Escape dollar signs so Streamlit does not treat '$...$' as LaTeX."""
    return re.sub(r"(?<!\\)\$", r"\\$", text)
