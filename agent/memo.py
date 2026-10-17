"""Memo helpers: the polite refusal memo, citable labels, Markdown rendering."""
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
