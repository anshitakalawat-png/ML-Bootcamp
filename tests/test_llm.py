from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors, types

import llm

SCHEMA = {"type": "object"}


def reply(text, finish=types.FinishReason.STOP, block_reason=None):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish)],
        prompt_feedback=SimpleNamespace(block_reason=block_reason) if block_reason else None,
    )


class FakeClient:
    """Stands in for genai.Client: records generate_content calls."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)  # responses, or exceptions to raise
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def client(monkeypatch):
    def install(*outcomes):
        fake = FakeClient(outcomes)
        monkeypatch.setattr(llm, "_client", lambda: fake)
        return fake

    return install


def call(**overrides):
    kwargs = dict(model="gemini-3.8-flash", system="sys", user="hello", schema=SCHEMA, effort="medium")
    kwargs.update(overrides)
    return llm.complete_json(**kwargs)


def texts(contents):
    return [(c.role, c.parts[0].text) for c in contents]


def api_error(code, message, status="ERROR"):
    return errors.ClientError(code, {"error": {"code": code, "message": message, "status": status}}) \
        if code < 500 else errors.ServerError(code, {"error": {"code": code, "message": message, "status": status}})


def test_valid_reply(client):
    fake = client(reply('{"a": 1}'))
    assert call() == {"a": 1}
    sent = fake.calls[0]
    assert sent["model"] == "gemini-3.8-flash"
    assert texts(sent["contents"]) == [("user", "hello")]
    config = sent["config"]
    assert config.system_instruction == "sys"
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == SCHEMA
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM


def test_malformed_then_valid_retries_once(client):
    fake = client(reply('{"a": '), reply('{"a": 2}'))
    assert call() == {"a": 2}
    assert len(fake.calls) == 2
    retry = texts(fake.calls[1]["contents"])
    assert [role for role, _ in retry] == ["user", "model", "user"]
    assert retry[1][1] == '{"a": '


def test_malformed_twice_raises(client):
    fake = client(reply("not json"), reply("still not"))
    with pytest.raises(llm.LLMError) as exc:
        call()
    assert "malformed JSON twice" in exc.value.message
    assert len(fake.calls) == 2


def test_truncated_reply_is_retried(client):
    fake = client(reply('{"a": "cut o', finish=types.FinishReason.MAX_TOKENS), reply('{"a": 3}'))
    assert call() == {"a": 3}
    assert len(fake.calls) == 2


def test_empty_reply_is_retried(client):
    fake = client(reply(None), reply('{"a": 4}'))
    assert call() == {"a": 4}
    assert texts(fake.calls[1]["contents"])[1] == ("model", "(empty reply)")


def test_check_failure_is_retried(client):
    def check(data):
        if not data.get("ok"):
            raise ValueError('"ok" must be true')

    fake = client(reply('{"ok": false}'), reply('{"ok": true}'))
    assert call(check=check) == {"ok": True}
    assert '"ok" must be true' in texts(fake.calls[1]["contents"])[2][1]


def test_non_object_json_is_rejected(client):
    client(reply("[1, 2]"), reply('"text"'))
    with pytest.raises(llm.LLMError):
        call()


@pytest.mark.parametrize("finish", [types.FinishReason.SAFETY, types.FinishReason.PROHIBITED_CONTENT])
def test_safety_stop_raises_without_retry(client, finish):
    fake = client(reply("", finish=finish))
    with pytest.raises(llm.LLMError) as exc:
        call()
    assert "declined" in exc.value.message
    assert len(fake.calls) == 1


def test_blocked_prompt_raises(client):
    client(reply(None, block_reason="SAFETY"))
    with pytest.raises(llm.LLMError) as exc:
        call()
    assert "declined" in exc.value.message


@pytest.mark.parametrize("effort, level", [
    ("minimal", types.ThinkingLevel.MINIMAL),
    ("low", types.ThinkingLevel.LOW),
    ("HIGH", types.ThinkingLevel.HIGH),
    ("xhigh", types.ThinkingLevel.HIGH),
    ("max", types.ThinkingLevel.HIGH),
])
def test_effort_maps_to_thinking_level(client, effort, level):
    fake = client(reply("{}"))
    call(effort=effort)
    assert fake.calls[0]["config"].thinking_config.thinking_level == level


def test_no_effort_sends_no_thinking_config(client):
    fake = client(reply("{}"))
    call(effort=None)
    assert fake.calls[0]["config"].thinking_config is None


def test_unknown_effort_is_a_clear_error(client):
    fake = client(reply("{}"))
    with pytest.raises(llm.LLMError) as exc:
        call(effort="extreme")
    assert "Unknown effort" in exc.value.message
    assert fake.calls == []


@pytest.mark.parametrize("error, expected", [
    (api_error(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT"), "GEMINI_API_KEY"),
    (api_error(404, "This model models/x is no longer available to new users.", "NOT_FOUND"), "is not available"),
    (api_error(429, "You exceeded your current quota", "RESOURCE_EXHAUSTED"), "quota or rate limit"),
    (api_error(503, "This model is currently experiencing high demand.", "UNAVAILABLE"), "temporarily unavailable"),
    (api_error(400, "Invalid JSON schema", "INVALID_ARGUMENT"), "was rejected"),
    (httpx.ConnectError("no route"), "Could not reach the Gemini API"),
    (httpx.ReadTimeout("slow"), "timed out"),
])
def test_api_errors_become_clear_messages(client, error, expected):
    client(error)
    with pytest.raises(llm.LLMError) as exc:
        call()
    assert expected in exc.value.message


def test_missing_api_key_gives_clear_error():
    # conftest removes GEMINI_API_KEY / GOOGLE_API_KEY and clears the cached client.
    with pytest.raises(llm.LLMError) as exc:
        call()
    assert exc.value.message == "No Gemini API key found. Set GEMINI_API_KEY in .env."


def test_client_retries_transient_errors():
    options = llm.RETRY
    assert options.attempts >= 3
    assert {429, 503} <= set(options.http_status_codes)
    assert 504 not in options.http_status_codes  # a timeout would wait the full timeout again
    assert 400 not in options.http_status_codes and 404 not in options.http_status_codes
