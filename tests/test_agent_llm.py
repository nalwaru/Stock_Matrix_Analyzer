from types import SimpleNamespace as NS

import openai
import pytest
from pydantic import BaseModel

from agent import llm
from agent.schemas import GradeResult, Memo, PassageGrade, RewriteResult, RouteDecision
from analyzer import ai


class Answer(BaseModel):
    value: int


class FakeClient:
    def __init__(self, parsed=None, refusal=None, error=None):
        self.calls = []
        self._parsed, self._refusal, self._error = parsed, refusal, error
        self.chat = NS(completions=NS(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return NS(choices=[NS(message=NS(parsed=self._parsed, refusal=self._refusal))])


MESSAGES = [{"role": "user", "content": "hi"}]


def test_ask_structured_returns_parsed_model():
    out = llm.ask_structured(MESSAGES, Answer, "m", client=FakeClient(parsed=Answer(value=3)))
    assert out == Answer(value=3)


def test_ask_structured_passes_schema_and_temperature_zero():
    client = FakeClient(parsed=Answer(value=1))
    llm.ask_structured(MESSAGES, Answer, "test-model", client=client)
    call = client.calls[0]
    assert call["response_format"] is Answer and call["temperature"] == 0
    assert call["model"] == "test-model" and call["messages"] == MESSAGES


def test_refusal_raises_aierror():
    with pytest.raises(ai.AIError, match="declined"):
        llm.ask_structured(MESSAGES, Answer, "m", client=FakeClient(refusal="no"))


def test_openai_error_raises_aierror_without_provider_text():
    client = FakeClient(error=openai.OpenAIError("secret provider detail"))
    with pytest.raises(ai.AIError) as exc:
        llm.ask_structured(MESSAGES, Answer, "m", client=client)
    assert "secret" not in str(exc.value)


def test_none_parsed_raises_aierror():
    with pytest.raises(ai.AIError, match="structured"):
        llm.ask_structured(MESSAGES, Answer, "m", client=FakeClient(parsed=None))


def test_missing_api_key_raises_aierror(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ai.AIError, match="OPENAI_API_KEY"):
        llm.ask_structured(MESSAGES, Answer, "m")


def test_schemas_validate_routes_and_fields():
    assert RouteDecision(route="agent", tickers=["AAPL"], reason="r").route == "agent"
    with pytest.raises(Exception):
        RouteDecision(route="nonsense", tickers=[], reason="r")
    assert GradeResult(grades=[PassageGrade(id="a", relevant=True, reason="r")]).grades[0].relevant
    assert RewriteResult(query="q").query == "q"
    assert Memo(answer="a", key_figures=[], sources=[], unverified=[]).answer == "a"


def test_chat_with_tools_returns_message_and_passes_tools():
    from tests.agent_fakes import ScriptedToolClient, assistant
    client = ScriptedToolClient([assistant("done")])
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    message = llm.chat_with_tools(MESSAGES, tools, "m", client=client)
    assert message.content == "done"
    assert client.calls[0]["tools"] == tools and client.calls[0]["temperature"] == 0


def test_chat_with_tools_maps_openai_errors_to_aierror():
    from tests.agent_fakes import ScriptedToolClient
    error = openai.APIConnectionError(request=NS())
    with pytest.raises(ai.AIError):
        llm.chat_with_tools(MESSAGES, [], "m", client=ScriptedToolClient([error]))


def test_chat_with_tools_requires_key_when_no_client(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ai.AIError):
        llm.chat_with_tools(MESSAGES, [], "m")


def test_ask_structured_uses_the_default_output_cap_unless_a_larger_one_is_given():
    client = FakeClient(parsed=Answer(value=1))
    llm.ask_structured(MESSAGES, Answer, "m", client=client)
    assert client.calls[0]["max_tokens"] == ai.MAX_OUTPUT_TOKENS
    llm.ask_structured(MESSAGES, Answer, "m", client=client, max_tokens=2500)
    assert client.calls[1]["max_tokens"] == 2500


class LengthThenOk:
    """parse() raises LengthFinishReasonError for the first `failures` calls, then succeeds."""

    def __init__(self, failures):
        self.failures, self.calls = failures, []
        self.chat = NS(completions=NS(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            raise openai.LengthFinishReasonError(completion=NS(usage=None, choices=[]))
        return NS(choices=[NS(message=NS(parsed=Answer(value=7), refusal=None))])


def test_a_runaway_completion_is_retried_once():
    client = LengthThenOk(failures=1)
    assert llm.ask_structured(MESSAGES, Answer, "m", client=client) == Answer(value=7)
    assert len(client.calls) == 2


def test_two_runaway_completions_end_in_a_clear_too_long_error():
    client = LengthThenOk(failures=2)
    with pytest.raises(ai.AIError, match="too long"):
        llm.ask_structured(MESSAGES, Answer, "m", client=client)
    assert len(client.calls) == 2


def test_other_openai_errors_are_not_retried():
    client = FakeClient(error=openai.OpenAIError("boom"))
    with pytest.raises(ai.AIError):
        llm.ask_structured(MESSAGES, Answer, "m", client=client)
    assert len(client.calls) == 1
