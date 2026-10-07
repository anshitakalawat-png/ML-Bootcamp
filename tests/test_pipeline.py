import json
import wave

import numpy as np
import pytest

import minutes
import pipeline
import refine
import stt
import validate
from pipeline import PipelineError

SR = 16_000

RAW = {
    "segments": [
        {"start": 0.0, "end": 2.0, "text": " PREA will update the cube control config by Friday."},
        {"start": 2.0, "end": 4.0, "text": " We will not migrate this sprint."},
    ],
}
RAW["text"] = "".join(s["text"] for s in RAW["segments"])
REFINED_TEXT = "Priya will update the kubectl config by Friday. We will not migrate this sprint."
RECORD = {
    "summary": "Deployment update.",
    "minutes": [{"topic": "Deployment", "points": ["Priya will update the kubectl config."]}],
    "key_decisions": [{"decision": "No migration this sprint.", "evidence": "We will not migrate this sprint."}],
    "action_items": [{"task": "Update config", "owner": "Priya", "deadline": "by Friday",
                      "evidence": "Priya will update the kubectl config by Friday"}],
    "open_questions": [],
}


def write_tone(path, seconds=2.0):
    t = np.arange(int(seconds * SR)) / SR
    pcm = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())
    return str(path)


@pytest.fixture
def stages(monkeypatch):
    """Replace stages 1-3 with fakes that record their calls. Stage 0 stays real."""
    calls = []

    def fake_stt(audio):
        calls.append(("stt", audio))
        return {"segments": [dict(s) for s in RAW["segments"]], "text": RAW["text"]}

    def fake_refine(raw, vocabulary=None):
        calls.append(("refine", raw, vocabulary))
        return {"refined_transcript": REFINED_TEXT,
                "corrections": [{"heard": "PREA", "corrected": "Priya"}],
                "warnings": ["Part 2 of 2: refined text was rejected."]}

    def fake_minutes(refined):
        calls.append(("minutes", refined))
        return {"record": json.loads(json.dumps(RECORD)),
                "warnings": ['Action item "x": owner "Arjun" ... changed to "unspecified".'],
                "model": "fake"}

    monkeypatch.setattr(stt, "transcribe_validated", fake_stt)
    monkeypatch.setattr(refine, "refine", fake_refine)
    monkeypatch.setattr(minutes, "generate_minutes", fake_minutes)
    return calls


def names(calls):
    return [c[0] for c in calls]


def test_success_runs_stages_in_order(tmp_path, stages):
    statuses = []
    result = pipeline.run_pipeline(write_tone(tmp_path / "a.wav"), vocabulary=["Priya"], on_status=statuses.append)

    assert statuses == ["Validating audio...", "Transcribing...", "Refining transcript...", "Generating minutes...", "Done"]
    assert names(stages) == ["stt", "refine", "minutes"]
    assert isinstance(stages[0][1], validate.ValidatedAudio)
    assert stages[1][2] == ["Priya"]
    assert stages[2][1]["refined_transcript"] == REFINED_TEXT

    assert result.raw_text == RAW["text"]
    assert result.segments == RAW["segments"]
    assert result.refined_transcript == REFINED_TEXT
    assert result.corrections == [{"heard": "PREA", "corrected": "Priya"}]
    assert result.record == RECORD
    assert result.warnings == ["Part 2 of 2: refined text was rejected.",
                               'Action item "x": owner "Arjun" ... changed to "unspecified".']


def test_validation_failure_stops_everything(tmp_path, stages):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    statuses = []
    with pytest.raises(PipelineError) as exc:
        pipeline.run_pipeline(str(empty), on_status=statuses.append)

    assert exc.value.stage == 0
    assert exc.value.message == "Stage 0 (validation) failed: The uploaded file is empty."
    assert stages == []
    assert statuses == ["Validating audio..."]


@pytest.mark.parametrize("name, content", [
    ("fake.mp3", b"not audio at all " * 100),
    ("notes.txt", b"hello"),
])
def test_bad_files_never_reach_transcription(tmp_path, stages, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(PipelineError) as exc:
        pipeline.run_pipeline(str(path))
    assert exc.value.stage == 0
    assert stages == []


def test_silent_file_never_reaches_transcription(tmp_path, stages):
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(np.zeros(SR * 3, dtype="<i2").tobytes())
    with pytest.raises(PipelineError) as exc:
        pipeline.run_pipeline(str(path))
    assert "silent" in exc.value.message
    assert stages == []


def test_no_speech_is_a_transcription_failure(tmp_path, stages, monkeypatch):
    def no_speech(audio):
        raise validate.AudioValidationError(validate.MSG_NO_SPEECH)

    monkeypatch.setattr(stt, "transcribe_validated", no_speech)
    with pytest.raises(PipelineError) as exc:
        pipeline.run_pipeline(write_tone(tmp_path / "a.wav"))
    assert exc.value.message == "Stage 1 (transcription) failed: No speech was detected in this recording."
    assert stages == []


def test_unexpected_error_still_names_its_stage(tmp_path, stages, monkeypatch):
    def broken(audio):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(stt, "transcribe_validated", broken)
    with pytest.raises(PipelineError) as exc:
        pipeline.run_pipeline(write_tone(tmp_path / "a.wav"))
    assert exc.value.stage == 1
    assert "Stage 1 (transcription) failed" in exc.value.message
    assert "model download failed" in exc.value.message
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_refinement_failure_stops_before_minutes(tmp_path, stages, monkeypatch):
    def fail(raw, vocabulary=None):
        raise refine.RefinementError("No Anthropic API key found. Set ANTHROPIC_API_KEY in .env.")

    monkeypatch.setattr(refine, "refine", fail)
    statuses = []
    with pytest.raises(PipelineError) as exc:
        pipeline.run_pipeline(write_tone(tmp_path / "a.wav"), on_status=statuses.append)
    assert exc.value.message == "Stage 2 (refinement) failed: No Anthropic API key found. Set ANTHROPIC_API_KEY in .env."
    assert names(stages) == ["stt"]
    assert statuses[-1] == "Refining transcript..."


def test_minutes_failure(tmp_path, stages, monkeypatch):
    def fail(refined):
        raise minutes.MinutesError("Model x returned malformed JSON twice (bad).")

    monkeypatch.setattr(minutes, "generate_minutes", fail)
    statuses = []
    with pytest.raises(PipelineError) as exc:
        pipeline.run_pipeline(write_tone(tmp_path / "a.wav"), on_status=statuses.append)
    assert exc.value.stage == 3
    assert exc.value.message.startswith("Stage 3 (documentation) failed: ")
    assert "Done" not in statuses


def test_refinement_cannot_change_raw_transcript(tmp_path, stages, monkeypatch):
    def tampering_refine(raw, vocabulary=None):
        raw["text"] = "TAMPERED"
        raw["segments"][0]["text"] = "TAMPERED"
        return {"refined_transcript": REFINED_TEXT, "corrections": [], "warnings": []}

    monkeypatch.setattr(refine, "refine", tampering_refine)
    result = pipeline.run_pipeline(write_tone(tmp_path / "a.wav"))
    assert result.raw_text == RAW["text"]
    assert result.segments == RAW["segments"]


def test_result_exports_all_files(tmp_path, stages):
    result = pipeline.run_pipeline(write_tone(tmp_path / "a.wav"))
    paths = result.export(str(tmp_path / "out"))
    assert open(paths["raw"], encoding="utf-8", newline="").read() == RAW["text"]
    assert open(paths["refined"], encoding="utf-8").read() == REFINED_TEXT
    data = json.load(open(paths["json"], encoding="utf-8"))
    assert data["action_items"] == RECORD["action_items"]
    assert data["warnings"] == result.warnings


def test_result_is_read_only(tmp_path, stages):
    result = pipeline.run_pipeline(write_tone(tmp_path / "a.wav"))

    record = result.record
    record["action_items"][0]["owner"] = "Someone else"
    record["key_decisions"].clear()
    segments = result.segments
    segments[0]["text"] = "TAMPERED"
    result.warnings.append("added")
    result.corrections.clear()

    assert result.record == RECORD
    assert result.segments == RAW["segments"]
    assert len(result.warnings) == 2
    assert len(result.corrections) == 1
    with pytest.raises(AttributeError):
        result.raw_text = "TAMPERED"
    with pytest.raises(AttributeError):
        result.new_field = 1


def test_result_values_are_plain_json(tmp_path, stages):
    result = pipeline.run_pipeline(write_tone(tmp_path / "a.wav"))
    json.dumps({"record": result.record, "segments": result.segments, "warnings": result.warnings})
