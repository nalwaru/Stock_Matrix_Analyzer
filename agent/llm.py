"""One structured-output call to OpenAI; every failure becomes a safe AIError."""
from typing import TypeVar

from openai import OpenAI, OpenAIError
from pydantic import BaseModel

from analyzer import ai

T = TypeVar("T", bound=BaseModel)


def _client(client):
    if client is not None:
        return client
    if not ai.is_configured():
        raise ai.AIError("OPENAI_API_KEY is not set.")
    return OpenAI(timeout=30.0, max_retries=1)


def chat_with_tools(messages: list[dict], tools: list[dict], model: str, client=None):
    """One tool-calling turn; returns the assistant message. Failures become AIError."""
    client = _client(client)
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice="auto",
            temperature=0, max_tokens=ai.MAX_OUTPUT_TOKENS,
        )
    except OpenAIError as exc:
        raise ai.safe_error(exc) from None
    return response.choices[0].message


def ask_structured(messages: list[dict], schema: type[T], model: str, client=None, extra_body: dict | None = None) -> T:
    if client is None:
        if not ai.is_configured():
            raise ai.AIError("OPENAI_API_KEY is not set.")
        client = OpenAI(timeout=30.0, max_retries=1)
    try:
        response = client.chat.completions.parse(
            model=model, messages=messages, response_format=schema,
            temperature=0, max_tokens=ai.MAX_OUTPUT_TOKENS,
            **({"extra_body": extra_body} if extra_body else {}),
        )
    except OpenAIError as exc:
        raise ai.safe_error(exc) from None
    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise ai.AIError("The model declined to answer.")
    if message.parsed is None:
        raise ai.AIError("The model returned no structured result.")
    return message.parsed
