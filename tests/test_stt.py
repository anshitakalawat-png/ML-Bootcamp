import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import stt
import validate
from validate import AudioValidationError

SR = 16_000


def write_tone(path: Path, seconds: float = 2.0) -> Path:
    t = np.arange(int(seconds * SR)) / SR
    pcm = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())
    return path


class FakeModel:
    def __init__(self, segments):
        self.segments = segments
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter(self.segments), SimpleNamespace(language="en")


def seg(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


@pytest.fixture
def fake_model(monkeypatch):
    holder = {}

    def install(segments):
        model = FakeModel(segments)
        holder["loads"] = []

        def load(size, device, compute_type):
            holder["loads"].append((size, device, compute_type))
            return model

        monkeypatch.setattr(stt, "_load_model", load)
        return model, holder["loads"]

    return install


def test_validation_runs_before_model_load(tmp_path, monkeypatch):
    def must_not_load(*args):
        raise AssertionError("model loaded for an invalid file")

    monkeypatch.setattr(stt, "_load_model", must_not_load)
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(AudioValidationError) as exc:
        stt.transcribe(empty)
    assert exc.value.message == validate.MSG_EMPTY


def test_text_is_exactly_whisper_output(tmp_path, fake_model):
    texts = [" Hello,  everyone.", " We will NOT ship on Friday ", "...um, 3.5k users"]
    model, _ = fake_model([seg(0.0, 1.0, texts[0]), seg(1.0, 2.0, texts[1]), seg(2.0, 3.0, texts[2])])

    result = stt.transcribe(write_tone(tmp_path / "a.wav"))

    assert [s["text"] for s in result["segments"]] == texts
    assert result["text"] == "".join(texts)
    assert result["segments"][1] == {"start": 1.0, "end": 2.0, "text": texts[1]}
    assert set(result) == {"segments", "text"}


def test_uses_vad_english_and_validated_samples(tmp_path, fake_model):
    model, _ = fake_model([seg(0.0, 1.0, " hi")])
    stt.transcribe(write_tone(tmp_path / "a.wav"))

    audio, kwargs = model.calls[0]
    assert kwargs["vad_filter"] is True
    assert kwargs["language"] == "en"
    assert isinstance(audio, np.ndarray) and audio.dtype == np.float32


def test_no_segments_raises_no_speech(tmp_path, fake_model):
    fake_model([])
    with pytest.raises(AudioValidationError) as exc:
        stt.transcribe(write_tone(tmp_path / "a.wav"))
    assert exc.value.message == validate.MSG_NO_SPEECH


def test_on_segment_called_in_order(tmp_path, fake_model):
    fake_model([seg(0.0, 1.0, " one"), seg(1.0, 2.0, " two")])
    seen = []
    stt.transcribe(write_tone(tmp_path / "a.wav"), on_segment=seen.append)
    assert [s["text"] for s in seen] == [" one", " two"]


def test_defaults_small_int8_on_cpu(tmp_path, fake_model, monkeypatch):
    for var in ("WHISPER_MODEL_SIZE", "WHISPER_COMPUTE_TYPE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    _, loads = fake_model([seg(0.0, 1.0, " hi")])
    stt.transcribe(write_tone(tmp_path / "a.wav"))
    assert loads == [("small", "cpu", "int8")]


def test_empty_env_values_use_defaults(tmp_path, fake_model, monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "")
    _, loads = fake_model([seg(0.0, 1.0, " hi")])
    stt.transcribe(write_tone(tmp_path / "a.wav"))
    assert loads == [("small", "cpu", "int8")]


def test_model_size_configurable(tmp_path, fake_model, monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "base")
    _, loads = fake_model([seg(0.0, 1.0, " hi")])
    stt.transcribe(write_tone(tmp_path / "a.wav"))
    stt.transcribe(write_tone(tmp_path / "b.wav"), model_size="tiny")
    assert [l[0] for l in loads] == ["base", "tiny"]
