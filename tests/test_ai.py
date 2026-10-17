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


def test_audit_messages_have_cfa_persona_and_summary_only():
    msgs = ai.build_audit_messages(SUMMARY)
    assert msgs[0]["role"] == "system"
    assert "Certified Financial Analyst" in msgs[0]["content"]
    assert "bullet" in msgs[0]["content"].lower()
    assert SUMMARY in msgs[1]["content"]


def test_chat_messages_prepend_current_portfolio_and_trim_history():
    history = []
    for i in range(10):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    msgs = ai.build_chat_messages(SUMMARY, history, "new question", max_turns=2)
    assert msgs[0]["role"] == "system" and SUMMARY in msgs[0]["content"]
    assert [m["content"] for m in msgs[1:-1]] == ["q8", "a8", "q9", "a9"]
    assert msgs[-1] == {"role": "user", "content": "new question"}


def test_generate_audit_uses_model_and_returns_stripped_text():
    client = FakeClient(text="  Summary\n- Action  ")
    out = ai.generate_audit(SUMMARY, "test-model", client=client)
    assert out == "Summary\n- Action"
    assert client.calls[0]["model"] == "test-model"


def test_chat_reply_returns_text():
    client = FakeClient(text="Tech is 60%.")
    assert ai.chat_reply(SUMMARY, [], "tech exposure?", "m", client=client) == "Tech is 60%."


def test_openai_error_is_wrapped_in_aierror():
    client = FakeClient(error=openai.OpenAIError("secret provider detail"))
    with pytest.raises(ai.AIError, match="could not complete") as exc:
        ai.generate_audit(SUMMARY, "m", client=client)
    assert "secret" not in str(exc.value)


def test_empty_response_raises_aierror():
    with pytest.raises(ai.AIError):
        ai.generate_audit(SUMMARY, "m", client=FakeClient(text=""))


def test_missing_api_key_raises_aierror_and_is_not_configured(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert ai.is_configured() is False
    with pytest.raises(ai.AIError, match="OPENAI_API_KEY"):
        ai.generate_audit(SUMMARY, "m")


def test_model_comes_from_env_with_default(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert ai.get_model() == ai.DEFAULT_MODEL
    monkeypatch.setenv("OPENAI_MODEL", "custom")
    assert ai.get_model() == "custom"


def test_audit_is_streamlit_cached():
    assert hasattr(ai.audit, "clear")


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
