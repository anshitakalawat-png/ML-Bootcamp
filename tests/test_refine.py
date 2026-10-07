import copy
import re

import pytest

import llm
import refine
from refine import RefinementError

RAW = {
    "segments": [
        {"start": 0.0, "end": 3.0, "text": " PREA will update the cube control config by Friday."},
        {"start": 3.0, "end": 6.0, "text": " We will not migrate the database this sprint."},
        {"start": 6.0, "end": 8.0, "text": " The budget is 42,000 rupees."},
    ],
}
RAW["text"] = "".join(s["text"] for s in RAW["segments"])


def transcript_of(user_message: str) -> str:
    return re.search(r"<transcript>\n(.*)\n</transcript>", user_message, re.S).group(1)


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a fake complete_json. `respond(chunk)` returns the reply dict."""

    def install(respond):
        calls = []

        def complete_json(**kwargs):
            calls.append(kwargs)
            reply = respond(transcript_of(kwargs["user"]))
            if "check" in kwargs and kwargs["check"]:
                kwargs["check"](reply)
            return reply

        monkeypatch.setattr(llm, "complete_json", complete_json)
        return calls

    return install


def echo(chunk):
    return {"refined_transcript": chunk, "corrections": []}


def test_refines_and_never_touches_raw(fake_llm):
    def respond(chunk):
        return {
            "refined_transcript": chunk.replace("PREA", "Priya").replace("cube control", "kubectl"),
            "corrections": [
                {"heard": "PREA", "corrected": "Priya"},
                {"heard": "cube control", "corrected": "kubectl"},
            ],
        }

    calls = fake_llm(respond)
    before = copy.deepcopy(RAW)

    result = refine.refine(RAW, vocabulary=["Priya", " kubectl ", "", "priya"])

    assert RAW == before
    assert result["refined_transcript"] == (
        "Priya will update the kubectl config by Friday. "
        "We will not migrate the database this sprint. The budget is 42,000 rupees."
    )
    assert result["corrections"] == [
        {"heard": "PREA", "corrected": "Priya"},
        {"heard": "cube control", "corrected": "kubectl"},
    ]
    assert result["warnings"] == [] and result["fallback_chunks"] == []
    assert "<vocabulary>\nPriya\nkubectl\n</vocabulary>" in calls[0]["user"]
    assert calls[0]["system"] == refine.PROMPT_PATH.read_text(encoding="utf-8")
    assert calls[0]["schema"] == refine.SCHEMA


def test_split_at_sentence_boundaries():
    text = " One two three. Four five six! Seven eight nine? Ten eleven twelve."
    chunks = refine.split_into_chunks(text, max_chars=36)
    assert chunks == ["One two three. Four five six!", "Seven eight nine? Ten eleven twelve."]


def test_long_unpunctuated_run_split_at_words():
    text = " ".join(f"word{i}" for i in range(50))
    chunks = refine.split_into_chunks(text, max_chars=40)
    assert all(len(c) <= 40 for c in chunks)
    assert " ".join(chunks) == text


def test_chunks_rejoined_in_order(fake_llm, monkeypatch):
    monkeypatch.setenv("REFINE_CHUNK_CHARS", "60")
    calls = fake_llm(echo)
    result = refine.refine(RAW)
    assert result["chunks"] == len(calls) == 3
    assert result["refined_transcript"] == RAW["text"].strip()


def test_drastically_shorter_falls_back_to_raw(fake_llm, monkeypatch):
    monkeypatch.setenv("REFINE_CHUNK_CHARS", "60")

    def respond(chunk):
        if "database" in chunk:
            return {"refined_transcript": "Database later.", "corrections": []}
        return echo(chunk)

    fake_llm(respond)
    result = refine.refine(RAW)
    assert result["refined_transcript"] == RAW["text"].strip()
    assert result["fallback_chunks"] == [2]
    assert "Part 2 of 3" in result["warnings"][0] and "length" in result["warnings"][0]


def test_drastically_longer_falls_back_to_raw(fake_llm):
    def respond(chunk):
        return {"refined_transcript": chunk + " In summary, the team agreed on many things today and more.", "corrections": []}

    fake_llm(respond)
    result = refine.refine(RAW)
    assert result["refined_transcript"] == RAW["text"].strip()
    assert len(result["warnings"]) == 1


def test_dropped_negation_falls_back(fake_llm):
    fake_llm(lambda c: {"refined_transcript": c.replace("will not", "will"), "corrections": []})
    result = refine.refine(RAW)
    assert "will not migrate" in result["refined_transcript"]
    assert "negation" in result["warnings"][0]


def test_contraction_counts_as_same_negation():
    assert refine.meaning_problems("We won't ship it.", "We will not ship it.", []) == []


def test_changed_number_falls_back(fake_llm):
    fake_llm(lambda c: {"refined_transcript": c.replace("42,000", "40,000"), "corrections": []})
    result = refine.refine(RAW)
    assert "42,000" in result["refined_transcript"]
    assert "numbers" in result["warnings"][0]


def test_number_inside_vocabulary_term_is_allowed():
    assert refine.meaning_problems("We use GPT four now.", "We use GPT-4 now.", ["GPT-4"]) == []
    assert refine.meaning_problems("We use GPT four now.", "We use GPT-4 now.", []) != []


def test_fallback_discards_that_chunks_corrections(fake_llm):
    fake_llm(lambda c: {"refined_transcript": "Short.", "corrections": [{"heard": "PREA", "corrected": "Priya"}]})
    result = refine.refine(RAW)
    assert result["corrections"] == []


def test_corrections_that_did_not_happen_are_dropped(fake_llm):
    fake_llm(lambda c: {
        "refined_transcript": c,
        "corrections": [
            {"heard": "Sundar", "corrected": "Sunder"},  # not in the transcript
            {"heard": "PREA", "corrected": "Priya"},     # not applied in the text
            {"heard": "Friday", "corrected": "Friday"},  # no change
        ],
    })
    assert refine.refine(RAW)["corrections"] == []


def test_llm_failure_raises_refinement_error(monkeypatch):
    def fail(**kwargs):
        raise llm.LLMError("Model x returned malformed JSON twice (bad).")

    monkeypatch.setattr(llm, "complete_json", fail)
    with pytest.raises(RefinementError) as exc:
        refine.refine(RAW)
    assert "malformed JSON twice" in exc.value.message


def test_reply_shape_check():
    with pytest.raises(ValueError):
        refine._check_reply({"refined_transcript": "", "corrections": []})
    with pytest.raises(ValueError):
        refine._check_reply({"refined_transcript": "x", "corrections": [{"heard": "a"}]})
    refine._check_reply({"refined_transcript": "x", "corrections": []})


def test_empty_raw_transcript():
    with pytest.raises(RefinementError):
        refine.refine({"segments": [], "text": "  "})


def test_model_configurable(fake_llm, monkeypatch):
    calls = fake_llm(echo)
    monkeypatch.setenv("REFINE_MODEL", "gemini-3.7-flash")
    assert refine.refine(RAW)["model"] == "gemini-3.7-flash"
    assert refine.refine(RAW, model="gemini-3.5-flash-lite")["model"] == "gemini-3.5-flash-lite"
    assert [c["model"] for c in calls] == ["gemini-3.7-flash", "gemini-3.5-flash-lite"]


def test_prompt_covers_the_rules():
    prompt = refine.PROMPT_PATH.read_text(encoding="utf-8").lower()
    for phrase in ["negation", "number", "commitment", "vocabulary", "speaker labels", "summarise", "reorder"]:
        assert phrase in prompt
