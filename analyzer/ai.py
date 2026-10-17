"""OpenAI integration: portfolio audit and chat. Receives aggregated JSON only."""
import json
import os

import pandas as pd
import streamlit as st
from openai import (
    APIConnectionError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from analyzer import portfolio

DEFAULT_MODEL = "gpt-4o-mini"
MAX_CHAT_TURNS = 6
MAX_QUESTION = 2000
MAX_MESSAGE_CHARS = 4000
MAX_OUTPUT_TOKENS = 1000
MAX_HOLDINGS = 100
HOLDING_COLUMNS = [
    "Ticker", "Asset_Class", "Shares", "Purchase_Price", "Price",
    "Current_Value", "Gain_Loss", "Gain_Loss_Pct", "Day_Change",
]

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
    recent = [
        {"role": m["role"], "content": m["content"][:MAX_MESSAGE_CHARS]}
        for m in (history[-2 * max_turns:] if max_turns > 0 else [])
    ]
    return [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT + summary_json},
        *recent,
        {"role": "user", "content": question},
    ]


def build_context(df: pd.DataFrame, max_rows: int = MAX_HOLDINGS) -> str:
    """Bounded JSON of the filtered holdings: totals, allocation and per-holding rows."""
    if df.empty:
        raise AIError("No holdings match the current filters.")
    holdings = (
        df[[c for c in HOLDING_COLUMNS if c in df.columns]]
        .sort_values("Current_Value", ascending=False, na_position="last")
        .round(2)
    )
    holdings["Ticker"] = holdings["Ticker"].astype(str).str.slice(0, 50)
    context = {
        **portfolio.ai_summary(df),
        "total_cost": round(portfolio.summary(df)["total_cost"], 2),
        "unpriced_holdings": portfolio.summary(df)["unpriced"],
        # pandas serialises NaN/Infinity as JSON null
        "holdings": json.loads(holdings.head(max_rows).to_json(orient="records")),
        "omitted_holdings": max(0, len(holdings) - max_rows),
    }
    return json.dumps(context, sort_keys=True, allow_nan=False)


def _complete(messages: list[dict], model: str, client=None) -> str:
    if client is None:
        if not is_configured():
            raise AIError("OPENAI_API_KEY is not set.")
        client = OpenAI(timeout=30.0, max_retries=1)
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=MAX_OUTPUT_TOKENS
        )
    except AuthenticationError:
        raise AIError("The OpenAI API key was rejected. Check OPENAI_API_KEY.") from None
    except RateLimitError:
        raise AIError("OpenAI quota or rate limit reached. Check billing or retry later.") from None
    except APIConnectionError:
        raise AIError("Could not reach OpenAI or the request timed out.") from None
    except OpenAIError:
        raise AIError("OpenAI could not complete the request.") from None
    text = response.choices[0].message.content
    if not text or not text.strip():
        raise AIError("The model returned an empty response.")
    return text.strip()


def generate_audit(summary_json: str, model: str, client=None) -> str:
    return _complete(build_audit_messages(summary_json), model, client)


def chat_reply(
    summary_json: str, history: list[dict], question: str, model: str, client=None
) -> str:
    question = question.strip()
    if not question or len(question) > MAX_QUESTION:
        raise AIError(f"Enter a question between 1 and {MAX_QUESTION} characters.")
    return _complete(build_chat_messages(summary_json, history, question), model, client)


@st.cache_data(show_spinner=False)
def audit(summary_json: str, model: str) -> str:
    """Cached by (payload, model); exceptions are never cached."""
    return generate_audit(summary_json, model)
