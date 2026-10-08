import json
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
    client = FakeClient(error=openai.OpenAIError("rate limited"))
    with pytest.raises(ai.AIError, match="rate limited"):
        ai.generate_audit(SUMMARY, "m", client=client)


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
