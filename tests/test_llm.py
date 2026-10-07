from types import SimpleNamespace

import pytest

import llm

SCHEMA = {"type": "object"}


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)  # (stop_reason, text)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        stop_reason, text = self.replies.pop(0)
        return SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        )


@pytest.fixture
def client(monkeypatch):
    def install(*replies):
        fake = FakeClient(replies)
        monkeypatch.setattr(llm, "_client", lambda: fake)
        return fake

    return install


def call(**overrides):
    kwargs = dict(model="claude-opus-5-5", system="sys", user="hello", schema=SCHEMA, effort="medium")
    kwargs.update(overrides)
    return llm.complete_json(**kwargs)


def test_valid_reply(client):
    fake = client(("end_turn", '{"a": 1}'))
    assert call() == {"a": 1}
    sent = fake.calls[0]
    assert sent["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "medium"}
    assert sent["fallbacks"] == "default"
    assert sent["system"] == "sys"


def test_malformed_then_valid_retries_once(client):
    fake = client(("end_turn", '{"a": '), ("end_turn", '{"a": 2}'))
    assert call() == {"a": 2}
    assert len(fake.calls) == 2
    retry_messages = fake.calls[1]["messages"]
    assert [m["role"] for m in retry_messages] == ["user", "assistant", "user"]
    assert retry_messages[1]["content"] == '{"a": '


def test_malformed_twice_raises(client):
    fake = client(("end_turn", "not json"), ("end_turn", "still not"))
    with pytest.raises(llm.LLMError) as exc:
        call()
    assert "malformed JSON twice" in exc.value.message
    assert len(fake.calls) == 2


def test_truncated_reply_is_retried(client):
    fake = client(("max_tokens", '{"a": "cut o'), ("end_turn", '{"a": 3}'))
    assert call() == {"a": 3}
    assert len(fake.calls) == 2


def test_check_failure_is_retried(client):
    def check(data):
        if not data.get("ok"):
            raise ValueError('"ok" must be true')

    fake = client(("end_turn", '{"ok": false}'), ("end_turn", '{"ok": true}'))
    assert call(check=check) == {"ok": True}
    assert '"ok" must be true' in fake.calls[1]["messages"][2]["content"]


def test_non_object_json_is_rejected(client):
    client(("end_turn", "[1, 2]"), ("end_turn", '"text"'))
    with pytest.raises(llm.LLMError):
        call()


def test_refusal_raises_without_retry(client):
    fake = client(("refusal", ""))
    with pytest.raises(llm.LLMError) as exc:
        call()
    assert "declined" in exc.value.message
    assert len(fake.calls) == 1


def test_missing_api_key_gives_clear_error(monkeypatch):
    import anthropic

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm, "_client", lambda: anthropic.Anthropic(api_key=None, max_retries=0))
    with pytest.raises(llm.LLMError) as exc:
        call()
    assert "ANTHROPIC_API_KEY" in exc.value.message


def test_haiku_gets_no_effort_or_fallbacks(client):
    fake = client(("end_turn", "{}"))
    call(model="claude-haiku-4-5")
    sent = fake.calls[0]
    assert "effort" not in sent["output_config"]
    assert "fallbacks" not in sent and "betas" not in sent
