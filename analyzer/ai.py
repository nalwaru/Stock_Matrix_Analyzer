"""OpenAI integration: portfolio audit and chat. Receives aggregated JSON only."""
import os

import streamlit as st
from openai import OpenAI, OpenAIError

DEFAULT_MODEL = "gpt-4o-mini"
MAX_CHAT_TURNS = 6

AUDIT_SYSTEM_PROMPT = (
    "You are a Certified Financial Analyst (CFA) reviewing an investor's portfolio. "
    "You are given a JSON summary: allocation percentages by asset class, the top 3 "
    "positions, total value and total ROI. Write an executive summary covering risk "
    "exposure, concentration, and diversification, then finish with a bulleted list of "
    "concrete rebalancing action items. Use Markdown. Be concise. This is educational "
    "analysis, not personalised financial advice."
)
CHAT_SYSTEM_PROMPT = (
    "You are a Certified Financial Analyst (CFA) answering questions about the user's "
    "current portfolio. Base answers only on the portfolio JSON below; if the data "
    "cannot answer the question, say so. Use Markdown and keep answers concise. "
    "This is educational analysis, not personalised financial advice.\n\n"
    "Current portfolio (JSON):\n"
)


class AIError(RuntimeError):
    """Any failure of the AI feature, safe to show to the user."""


def is_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def get_model() -> str:
    return os.getenv("OPENAI_MODEL") or DEFAULT_MODEL


def build_audit_messages(summary_json: str) -> list[dict]:
    return [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Portfolio summary:\n{summary_json}"},
    ]


def build_chat_messages(
    summary_json: str,
    history: list[dict],
    question: str,
    max_turns: int = MAX_CHAT_TURNS,
) -> list[dict]:
    recent = history[-2 * max_turns:] if max_turns > 0 else []
    return [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT + summary_json},
        *recent,
        {"role": "user", "content": question},
    ]


def _complete(messages: list[dict], model: str, client=None) -> str:
    if client is None:
        if not is_configured():
            raise AIError("OPENAI_API_KEY is not set.")
        client = OpenAI()
    try:
        response = client.chat.completions.create(model=model, messages=messages)
    except OpenAIError as exc:
        raise AIError(f"OpenAI request failed: {exc}") from exc
    text = response.choices[0].message.content
    if not text or not text.strip():
        raise AIError("The model returned an empty response.")
    return text.strip()


def generate_audit(summary_json: str, model: str, client=None) -> str:
    return _complete(build_audit_messages(summary_json), model, client)


def chat_reply(
    summary_json: str, history: list[dict], question: str, model: str, client=None
) -> str:
    return _complete(build_chat_messages(summary_json, history, question), model, client)


@st.cache_data(show_spinner=False)
def audit(summary_json: str, model: str) -> str:
    """Cached by (payload, model); exceptions are never cached."""
    return generate_audit(summary_json, model)
