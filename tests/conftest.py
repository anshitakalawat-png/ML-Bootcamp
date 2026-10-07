import pytest

import llm


@pytest.fixture(autouse=True)
def no_real_gemini(monkeypatch):
    """Tests must never call the real Gemini API.

    llm.py loads .env at import, so a real key may be in the environment.
    Remove it and drop any cached client, so an unmocked call fails with
    "No Gemini API key found" instead of spending quota.
    """
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    real_client = llm._client  # a test may replace llm._client with a fake
    real_client.cache_clear()
    yield
    real_client.cache_clear()
