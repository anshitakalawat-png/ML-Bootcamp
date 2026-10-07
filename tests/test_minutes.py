import copy

import pytest

import llm
import minutes
import refine
from minutes import UNSPECIFIED, MinutesError

TRANSCRIPT = (
    "Good morning everyone. Let's start the review. Priya will update the Kubernetes "
    "deployment by Friday. We agreed we will not migrate the database this sprint. "
    "Maybe we could try a new logging tool. Rahul and Meena will write the test plan. "
    "Someone should look at the budget. The budget is 42,000 rupees. Who owns the demo?"
)
REFINED = {"refined_transcript": TRANSCRIPT, "corrections": [], "warnings": []}


def record(**overrides):
    base = {
        "summary": "Review meeting covering deployment and the database.",
        "minutes": [{"topic": "Deployment", "points": ["Priya will update the Kubernetes deployment."]}],
        "key_decisions": [
            {"decision": "No database migration this sprint.",
             "evidence": "we will not migrate the database this sprint"},
        ],
        "action_items": [
            {"task": "Update the Kubernetes deployment", "owner": "Priya", "deadline": "by Friday",
             "evidence": "Priya will update the Kubernetes deployment by Friday"},
        ],
        "open_questions": ["Who owns the demo?"],
    }
    base.update(overrides)
    return base


@pytest.fixture
def fake_llm(monkeypatch):
    def install(*replies):
        calls = []
        queue = list(replies)

        def complete_json(**kwargs):
            calls.append(kwargs)
            reply = copy.deepcopy(queue.pop(0))
            kwargs["check"](reply)
            return reply

        monkeypatch.setattr(llm, "complete_json", complete_json)
        return calls

    return install


def test_valid_record_passes_unchanged(fake_llm):
    calls = fake_llm(record())
    result = minutes.generate_minutes(REFINED)
    assert result["record"] == record()
    assert result["warnings"] == []
    assert calls[0]["system"] == minutes.PROMPT_PATH.read_text(encoding="utf-8")
    assert calls[0]["user"] == f"<transcript>\n{TRANSCRIPT}\n</transcript>"
    assert calls[0]["schema"] == minutes.SCHEMA


def test_accepts_plain_text(fake_llm):
    fake_llm(record())
    assert minutes.generate_minutes(TRANSCRIPT)["record"] == record()


def test_raw_stt_output_is_rejected():
    with pytest.raises(MinutesError) as exc:
        minutes.generate_minutes({"segments": [], "text": TRANSCRIPT})
    assert "refined transcript" in exc.value.message


def test_invented_owner_becomes_unspecified(fake_llm):
    item = {"task": "Update deployment", "owner": "Arjun", "deadline": "by Friday",
            "evidence": "Priya will update the Kubernetes deployment by Friday"}
    fake_llm(record(action_items=[item]))
    result = minutes.generate_minutes(REFINED)
    assert result["record"]["action_items"][0]["owner"] == UNSPECIFIED
    assert result["record"]["action_items"][0]["deadline"] == "by Friday"
    assert len(result["warnings"]) == 1 and '"Arjun"' in result["warnings"][0]


def test_invented_deadline_becomes_unspecified(fake_llm):
    item = {"task": "Update deployment", "owner": "Priya", "deadline": "2026-10-09",
            "evidence": "Priya will update the Kubernetes deployment by Friday"}
    fake_llm(record(action_items=[item]))
    result = minutes.generate_minutes(REFINED)
    assert result["record"]["action_items"][0]["deadline"] == UNSPECIFIED
    assert "deadline" in result["warnings"][0]


def test_multiple_owners_each_checked(fake_llm):
    ok = {"task": "Write test plan", "owner": "Rahul and Meena", "deadline": "unspecified",
          "evidence": "Rahul and Meena will write the test plan"}
    bad = {"task": "Write test plan", "owner": "Rahul, Vikram", "deadline": "unspecified",
           "evidence": "Rahul and Meena will write the test plan"}
    fake_llm(record(action_items=[ok, bad]))
    items = minutes.generate_minutes(REFINED)["record"]["action_items"]
    assert items[0]["owner"] == "Rahul and Meena"
    assert items[1]["owner"] == UNSPECIFIED


def test_owner_must_match_whole_words(fake_llm):
    # "Ever" only occurs inside "everyone", so it is not a name from the transcript.
    item = {"task": "Look at budget", "owner": "Ever", "deadline": "unspecified",
            "evidence": "Someone should look at the budget"}
    fake_llm(record(action_items=[item]))
    assert minutes.generate_minutes(REFINED)["record"]["action_items"][0]["owner"] == UNSPECIFIED


def test_unstated_spellings_normalised_without_warning(fake_llm):
    item = {"task": "Look at budget", "owner": "TBD", "deadline": "N/A",
            "evidence": "Someone should look at the budget"}
    fake_llm(record(action_items=[item]))
    result = minutes.generate_minutes(REFINED)
    assert result["record"]["action_items"][0]["owner"] == UNSPECIFIED
    assert result["record"]["action_items"][0]["deadline"] == UNSPECIFIED
    assert result["warnings"] == []


def test_matching_ignores_case_and_punctuation(fake_llm):
    item = {"task": "Update deployment", "owner": "PRIYA", "deadline": "By Friday.",
            "evidence": "priya will update the kubernetes deployment, by friday"}
    fake_llm(record(action_items=[item]))
    result = minutes.generate_minutes(REFINED)
    assert result["record"]["action_items"][0]["owner"] == "PRIYA"
    assert result["warnings"] == []


def test_unfound_evidence_is_flagged(fake_llm):
    decision = {"decision": "Adopt a new logging tool", "evidence": "we agreed to adopt the logging tool"}
    fake_llm(record(key_decisions=[decision]))
    result = minutes.generate_minutes(REFINED)
    assert len(result["warnings"]) == 1
    assert "Adopt a new logging tool" in result["warnings"][0]


@pytest.mark.parametrize("broken", [
    {"summary": "x"},  # keys missing
    record(summary="  "),
    record(action_items=[{"task": "t", "owner": "Priya", "evidence": "e"}]),  # no deadline
    record(minutes=[{"topic": "t", "points": "not a list"}]),
    record(open_questions=[{"q": 1}]),
])
def test_check_record_rejects_malformed(broken):
    with pytest.raises(ValueError):
        minutes.check_record(broken)


def test_llm_failure_raises_minutes_error(monkeypatch):
    def fail(**kwargs):
        raise llm.LLMError("Model x returned malformed JSON twice (bad).")

    monkeypatch.setattr(llm, "complete_json", fail)
    with pytest.raises(MinutesError) as exc:
        minutes.generate_minutes(REFINED)
    assert "malformed JSON twice" in exc.value.message


def test_default_model_differs_from_refinement(fake_llm, monkeypatch):
    for var in ("MINUTES_MODEL", "REFINE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    calls = fake_llm(record())
    minutes.generate_minutes(REFINED)
    assert calls[0]["model"] == "gemini-3.1-flash-lite"
    assert calls[0]["effort"] == "high"

    refine_calls = []
    monkeypatch.setattr(llm, "complete_json", lambda **kw: refine_calls.append(kw) or
                        {"refined_transcript": TRANSCRIPT, "corrections": []})
    refine.refine({"text": TRANSCRIPT})
    assert refine_calls[0]["model"] != calls[0]["model"]


def test_prompt_covers_the_rules():
    prompt = minutes.PROMPT_PATH.read_text(encoding="utf-8").lower()
    for phrase in ['"unspecified"', "proposals", "not decisions", "speaker", "evidence",
                   "copied exactly", "open_questions"]:
        assert phrase in prompt
