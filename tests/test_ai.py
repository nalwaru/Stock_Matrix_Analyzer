import json

import numpy as np
import pandas as pd
from types import SimpleNamespace as NS

import openai
import pytest

from analyzer import ai

SUMMARY = json.dumps({"total_value": 1000.0, "allocation_pct": {"Stocks": 100.0}}, sort_keys=True)


class FakeClient:
    def __init__(self, text="ok", error=None):
        self.calls = []
        self._text, self._error = text, error
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return NS(choices=[NS(message=NS(content=self._text))])


def test_chat_messages_prepend_current_portfolio_and_trim_history():
    history = []
    for i in range(10):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    msgs = ai.build_chat_messages(SUMMARY, history, "new question", max_turns=2)
    assert msgs[0]["role"] == "system" and SUMMARY in msgs[0]["content"]
    assert [m["content"] for m in msgs[1:-1]] == ["q8", "a8", "q9", "a9"]
    assert msgs[-1] == {"role": "user", "content": "new question"}


def test_chat_reply_returns_text():
    client = FakeClient(text="Tech is 60%.")
    assert ai.chat_reply(SUMMARY, [], "tech exposure?", "m", client=client) == "Tech is 60%."


def test_openai_error_is_wrapped_in_aierror():
    client = FakeClient(error=openai.OpenAIError("secret provider detail"))
    with pytest.raises(ai.AIError, match="could not complete") as exc:
        ai.chat_reply(SUMMARY, [], "q", "m", client=client)
    assert "secret" not in str(exc.value)


def test_empty_response_raises_aierror():
    with pytest.raises(ai.AIError):
        ai.chat_reply(SUMMARY, [], "q", "m", client=FakeClient(text=""))


def test_missing_api_key_raises_aierror_and_is_not_configured(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert ai.is_configured() is False
    with pytest.raises(ai.AIError, match="OPENAI_API_KEY"):
        ai.chat_reply(SUMMARY, [], "q", "m")


def test_model_comes_from_env_with_default(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert ai.get_model() == ai.DEFAULT_MODEL
    monkeypatch.setenv("OPENAI_MODEL", "custom")
    assert ai.get_model() == "custom"


def holdings(n=3):
    return pd.DataFrame({
        "Ticker": [f"T{i}" for i in range(n)],
        "Asset_Class": ["Stocks"] * n,
        "Shares": [1.0] * n,
        "Purchase_Price": [10.0] * n,
        "Price": [12.0] * n,
        "Cost_Basis": [10.0] * n,
        "Current_Value": [12.0 + i for i in range(n)],
        "Gain_Loss": [2.0] * n,
        "Gain_Loss_Pct": [20.0] * n,
        "Day_Change": [np.nan] * n,
    })


def test_build_context_is_sorted_json_with_nulls():
    ctx = json.loads(ai.build_context(holdings()))
    assert [h["Ticker"] for h in ctx["holdings"]] == ["T2", "T1", "T0"]
    assert ctx["holdings"][0]["Day_Change"] is None
    assert ctx["omitted_holdings"] == 0 and ctx["total_cost"] == 30.0


def test_build_context_bounds_rows_and_reports_omitted():
    ctx = json.loads(ai.build_context(holdings(5), max_rows=2))
    assert len(ctx["holdings"]) == 2 and ctx["omitted_holdings"] == 3


def test_build_context_empty_raises():
    with pytest.raises(ai.AIError):
        ai.build_context(holdings(0))


def test_chat_reply_rejects_blank_and_oversized_questions():
    for bad in ("   ", "x" * (ai.MAX_QUESTION + 1)):
        with pytest.raises(ai.AIError):
            ai.chat_reply(SUMMARY, [], bad, "m", client=FakeClient())


def test_chat_history_messages_are_truncated():
    history = [{"role": "user", "content": "x" * 10_000}]
    msgs = ai.build_chat_messages(SUMMARY, history, "q")
    assert len(msgs[1]["content"]) == ai.MAX_MESSAGE_CHARS


def test_request_caps_output_tokens():
    client = FakeClient()
    ai.chat_reply(SUMMARY, [], "q", "m", client=client)
    assert client.calls[0]["max_tokens"] == ai.MAX_OUTPUT_TOKENS


def passage(text="Revenue fell 5% on weaker iPhone demand.", **kw):
    from types import SimpleNamespace
    base = dict(ticker="AAPL", form="8-K", filing_date="2026-10-01", section="Item 2", text=text)
    return SimpleNamespace(**{**base, **kw})


def test_messages_without_passages_unchanged():
    assert ai.build_chat_messages(SUMMARY, [], "q") == ai.build_chat_messages(SUMMARY, [], "q", passages=None)
    assert "Filing passages" not in ai.build_chat_messages(SUMMARY, [], "q")[0]["content"]


def test_passages_appear_with_citation_labels():
    system = ai.build_chat_messages(SUMMARY, [], "why down?", passages=[passage()])[0]["content"]
    assert "[AAPL 8-K, 2026-10-01, Item 2]" in system
    assert "Revenue fell 5%" in system
    assert system.index(SUMMARY) < system.index("Filing passages")


def test_prompt_requires_not_explained_fallback_and_untrusted_passages():
    system = ai.build_chat_messages(SUMMARY, [], "q", passages=[passage()])[0]["content"]
    assert "not explained by filings" in system
    assert "untrusted" in system.lower()
    empty = ai.build_chat_messages(SUMMARY, [], "q", passages=[])[0]["content"]
    assert "not explained by filings" in empty


def test_passage_text_truncated_per_passage():
    system = ai.build_chat_messages(SUMMARY, [], "q", passages=[passage(text="Q" * 10_000)])[0]["content"]
    assert system.count("Q") == ai.MAX_MESSAGE_CHARS


def test_chat_reply_passes_passages_through():
    client = FakeClient(text="ok")
    ai.chat_reply(SUMMARY, [], "q", "m", client=client, passages=[passage()])
    assert "[AAPL 8-K, 2026-10-01, Item 2]" in client.calls[0]["messages"][0]["content"]


def test_system_prompt_forbids_outside_knowledge_and_gives_polite_refusal():
    system = ai.build_chat_messages(SUMMARY, [], "q", passages=[passage()])[0]["content"]
    assert "outside knowledge" in system.lower()
    assert ai.IDK_MESSAGE in system


def test_requests_use_temperature_zero():
    client = FakeClient(text="ok")
    ai.chat_reply(SUMMARY, [], "q", "m", client=client)
    assert client.calls[0]["temperature"] == 0


def test_reply_citing_a_source_that_was_not_retrieved_becomes_idk():
    client = FakeClient(text="Sales fell [AAPL 10-K, 2020-01-01, Item 7].")
    out = ai.chat_reply(SUMMARY, [], "q", "m", client=client, passages=[passage()])
    assert out == ai.IDK_MESSAGE


def test_reply_citing_a_retrieved_source_is_kept():
    text = "Revenue fell [AAPL 8-K, 2026-10-01, Item 2]."
    out = ai.chat_reply(SUMMARY, [], "q", "m", client=FakeClient(text=text), passages=[passage()])
    assert out == text


def test_citation_without_any_passages_becomes_idk():
    client = FakeClient(text="See [AAPL 10-K, 2025-11-01, Item 7].")
    assert ai.chat_reply(SUMMARY, [], "q", "m", client=client) == ai.IDK_MESSAGE


def test_safe_error_names_exhausted_credits_without_blaming_openai():
    import httpx
    import openai
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(402, request=request, json={"error": {"message": "credits"}})
    exc = openai.APIStatusError("credits", response=response, body=None)
    message = str(ai.safe_error(exc))
    assert "credits" in message.lower() and "OpenAI could not" not in message


def test_the_audit_feature_is_gone():
    for name in ("AUDIT_SYSTEM_PROMPT", "build_audit_messages", "generate_audit", "audit"):
        assert not hasattr(ai, name), name
