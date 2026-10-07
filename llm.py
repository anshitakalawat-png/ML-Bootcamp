"""Shared Claude call for the LLM stages (refinement and documentation).

complete_json() sends one request that must come back as a JSON object
matching `schema`. The reply is parsed and passed to `check`; if either
fails, the request is retried once with the error, and a second failure
raises LLMError. Callers turn LLMError into their own stage error.

Credentials come from the environment (ANTHROPIC_API_KEY in .env), never
from code.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Callable

import anthropic
from dotenv import load_dotenv

load_dotenv()

MAX_TOKENS = 16000

# Models that accept `fallbacks: "default"`, which re-runs a request on
# another model server-side if a safety classifier declines it.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

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
def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


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
                {"role": "assistant", "content": text or "(empty reply)"},
                {"role": "user", "content": RETRY_MESSAGE.format(error=error)},
            ]
    raise LLMError(f"Model {model} returned malformed JSON twice ({error}).")


def _send(model: str, system: str, messages: list, schema: dict, effort: str | None) -> str:
    output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
    if effort and not model.startswith("claude-haiku"):  # Haiku rejects effort
        output_config["effort"] = effort
    kwargs: dict = {}
    if model in _FALLBACK_MODELS:
        kwargs = {"betas": [_FALLBACK_BETA], "fallbacks": "default"}

    try:
        response = _client().beta.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            output_config=output_config,
            **kwargs,
        )
    except anthropic.AuthenticationError:
        raise LLMError("The Anthropic API key is invalid. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.CredentialsError as exc:
        raise LLMError(f"Anthropic credentials could not be loaded: {exc}")
    except TypeError as exc:
        # With no credentials at all, the SDK raises TypeError before sending.
        if "authentication" not in str(exc):
            raise
        raise LLMError("No Anthropic API key found. Set ANTHROPIC_API_KEY in .env.")
    except anthropic.NotFoundError:
        raise LLMError(f"Model '{model}' was not found. Check the model name in .env.")
    except anthropic.BadRequestError as exc:
        raise LLMError(f"The request to {model} was rejected: {exc.message}")
    except anthropic.RateLimitError:
        raise LLMError("The Anthropic API rate limit was reached. Try again in a minute.")
    except anthropic.APIStatusError as exc:
        raise LLMError(f"The Anthropic API returned an error ({exc.status_code}): {exc.message}")
    except anthropic.APIConnectionError:
        raise LLMError("Could not reach the Anthropic API. Check the internet connection.")

    if response.stop_reason == "refusal":
        raise LLMError(f"Model {model} declined to process this transcript.")
    # A reply cut off at max_tokens is incomplete JSON; returning it lets the
    # parse fail and the retry happen.
    return "".join(b.text for b in response.content if b.type == "text")
