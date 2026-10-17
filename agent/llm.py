"""One structured-output call to OpenAI; every failure becomes a safe AIError."""
from typing import TypeVar

from openai import OpenAI, OpenAIError
from pydantic import BaseModel

from analyzer import ai

T = TypeVar("T", bound=BaseModel)


def ask_structured(messages: list[dict], schema: type[T], model: str, client=None) -> T:
    if client is None:
        if not ai.is_configured():
            raise ai.AIError("OPENAI_API_KEY is not set.")
        client = OpenAI(timeout=30.0, max_retries=1)
    try:
        response = client.chat.completions.parse(
            model=model, messages=messages, response_format=schema,
            temperature=0, max_tokens=ai.MAX_OUTPUT_TOKENS,
        )
    except OpenAIError as exc:
        raise ai.safe_error(exc) from None
    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise ai.AIError("The model declined to answer.")
    if message.parsed is None:
        raise ai.AIError("The model returned no structured result.")
    return message.parsed
