"""Shared Gemini call for the LLM stages (refinement and documentation).

complete_json() sends one request that must come back as a JSON object
matching `schema`. The reply is parsed and passed to `check`; if either
fails, the request is retried once with the error, and a second failure
raises LLMError. Callers turn LLMError into their own stage error.

Credentials come from the environment (GEMINI_API_KEY in .env), never from
code. genai.Client() reads the key itself.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Callable

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

MAX_OUTPUT_TOKENS = 32768  # includes the model's thinking tokens
TIMEOUT_MS = 180_000

# Quick transient failures (rate limit, overload) are retried by the SDK
# with exponential backoff before complete_json sees them. Timeouts (504) are
# not retried: each attempt would wait the full TIMEOUT_MS again.
RETRY = types.HttpRetryOptions(
    attempts=4,
    initial_delay=2.0,
    max_delay=30.0,
    http_status_codes=[429, 500, 502, 503],
)

# Effort setting (.env) -> Gemini thinking level. xhigh/max are accepted so
# older .env files keep working; Gemini's highest level is HIGH.
THINKING_LEVELS = {
    "minimal": types.ThinkingLevel.MINIMAL,
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
    "xhigh": types.ThinkingLevel.HIGH,
    "max": types.ThinkingLevel.HIGH,
}

# Finish reasons meaning the model would not answer this content.
_DECLINED = {
    types.FinishReason.SAFETY,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.SPII,
    types.FinishReason.RECITATION,
}

RETRY_MESSAGE = (
    "Your previous reply could not be used: {error}\n"
    "Reply again with only the JSON object, following the schema exactly."
)


class LLMError(Exception):
    """The model could not produce a usable reply. `message` is user-facing."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    return genai.Client(http_options=types.HttpOptions(timeout=TIMEOUT_MS, retry_options=RETRY))


def complete_json(
    *,
    model: str,
    system: str,
    user: str,
    schema: dict,
    check: Callable[[dict], None] | None = None,
    effort: str | None = None,
) -> dict:
    """Return the model's reply as a dict that matches `schema` and passes `check`.

    `check` raises ValueError to reject a reply that is valid JSON but
    unusable (e.g. a required field is empty).
    """
    messages = [{"role": "user", "content": user}]
    error = None
    for _attempt in range(2):
        text = _send(model, system, messages, schema, effort)
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("the reply is not a JSON object")
            if check:
                check(data)
            return data
        except ValueError as exc:  # json.JSONDecodeError is a ValueError
            error = str(exc)
            messages = messages + [
                {"role": "model", "content": text or "(empty reply)"},
                {"role": "user", "content": RETRY_MESSAGE.format(error=error)},
            ]
    raise LLMError(f"Model {model} returned malformed JSON twice ({error}).")


def _send(model: str, system: str, messages: list, schema: dict, effort: str | None) -> str:
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_json_schema=schema,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    if effort:
        level = THINKING_LEVELS.get(effort.lower())
        if level is None:
            raise LLMError(f"Unknown effort '{effort}'. Use one of: minimal, low, medium, high.")
        config.thinking_config = types.ThinkingConfig(thinking_level=level)

    contents = [
        types.Content(role=m["role"], parts=[types.Part(text=m["content"])]) for m in messages
    ]

    try:
        response = _client().models.generate_content(model=model, contents=contents, config=config)
    except ValueError as exc:
        # genai.Client() raises ValueError before sending when no key is set.
        if "API key" not in str(exc):
            raise
        raise LLMError("No Gemini API key found. Set GEMINI_API_KEY in .env.")
    except errors.APIError as exc:
        raise LLMError(_describe_api_error(exc, model))
    except httpx.TimeoutException:
        raise LLMError(f"The request to {model} timed out. Try again, or use a shorter recording.")
    except httpx.TransportError:
        raise LLMError("Could not reach the Gemini API. Check the internet connection.")

    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        raise LLMError(f"Model {model} declined to process this transcript ({feedback.block_reason}).")
    candidates = response.candidates or []
    if candidates and candidates[0].finish_reason in _DECLINED:
        raise LLMError(f"Model {model} declined to process this transcript ({candidates[0].finish_reason}).")

    # A reply cut off at MAX_TOKENS is incomplete JSON; returning it lets the
    # parse fail and the retry happen.
    return response.text or ""


def _describe_api_error(exc: errors.APIError, model: str) -> str:
    message = exc.message or str(exc)
    if exc.code in (400, 401, 403) and "api key" in message.lower():
        return "The Gemini API key is invalid. Check GEMINI_API_KEY in .env."
    if exc.code == 403:
        return f"The Gemini API key is not allowed to use {model}: {message}"
    if exc.code == 404:
        return f"Model '{model}' is not available: {message} Check the model name in .env."
    if exc.code == 429:
        return (f"The Gemini API quota or rate limit was reached for {model}. "
                f"Wait a minute and try again, or choose another model in .env. ({message})")
    if exc.code in (500, 502, 503, 504):
        return f"Model {model} is temporarily unavailable ({exc.code}: {message}). Try again in a few minutes."
    if exc.code == 400:
        return f"The request to {model} was rejected: {message}"
    return f"The Gemini API returned an error ({exc.code}): {message}"
