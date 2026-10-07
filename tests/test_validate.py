import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

import validate
from validate import AudioValidationError, ensure_speech_detected, validate_audio

ROOT = Path(__file__).resolve().parent.parent
SR = 16_000


def write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> Path:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())
    return path


def tone(seconds: float, sr: int = SR, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * 220 * t)


def assert_rejected(path, message_part: str):
    with pytest.raises(AudioValidationError) as exc:
        validate_audio(path)
    assert message_part in exc.value.message


def test_empty_file(tmp_path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    assert_rejected(path, validate.MSG_EMPTY)


def test_missing_file(tmp_path):
    assert_rejected(tmp_path / "nope.wav", validate.MSG_EMPTY)


def test_unsupported_extension_lists_formats(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(AudioValidationError) as exc:
        validate_audio(path)
    for ext in validate.SUPPORTED_EXTENSIONS:
        assert ext in exc.value.message


def test_fake_mp3(tmp_path):
    path = tmp_path / "fake.mp3"
    path.write_text("this is a text file renamed to mp3\n" * 50)
    assert_rejected(path, "unreadable or corrupted")


def test_too_short(tmp_path):
    path = write_wav(tmp_path / "short.wav", tone(0.5))
    assert_rejected(path, "too short")


def test_pure_silence(tmp_path):
    path = write_wav(tmp_path / "silence.wav", np.zeros(SR * 3))
    assert_rejected(path, validate.MSG_SILENT)


def test_near_silence_noise_floor(tmp_path):
    rng = np.random.default_rng(0)
    path = write_wav(tmp_path / "hiss.wav", rng.normal(0, 1e-4, SR * 3))
    assert_rejected(path, validate.MSG_SILENT)


def test_valid_wav(tmp_path):
    path = write_wav(tmp_path / "ok.wav", tone(2.0))
    audio = validate_audio(path)
    assert audio.sample_rate == SR
    assert audio.duration_s == pytest.approx(2.0, abs=0.05)
    assert audio.samples.dtype == np.float32
    assert audio.active_fraction > 0.9


def test_valid_wav_is_resampled_to_16k(tmp_path):
    path = write_wav(tmp_path / "ok44k.wav", tone(2.0, sr=44_100), sr=44_100)
    audio = validate_audio(path)
    assert audio.sample_rate == SR
    assert audio.duration_s == pytest.approx(2.0, abs=0.05)


def test_valid_flac(tmp_path):
    path = tmp_path / "ok.flac"
    samples = (tone(2.0) * 32767).astype(np.int16).reshape(1, -1)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("flac", rate=SR, layout="mono")
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = SR
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    assert validate_audio(path).duration_s == pytest.approx(2.0, abs=0.05)


def test_no_speech_segments():
    with pytest.raises(AudioValidationError) as exc:
        ensure_speech_detected(iter([]))
    assert exc.value.message == validate.MSG_NO_SPEECH

    with pytest.raises(AudioValidationError):
        ensure_speech_detected([SimpleNamespace(text="   ")])


def test_speech_segments_pass_through():
    segs = ensure_speech_detected(iter([SimpleNamespace(text=" hello ")]))
    assert len(segs) == 1


def test_validation_never_loads_whisper(tmp_path):
    """A bad file must be rejected without faster-whisper even being imported."""
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    code = (
        "import sys, validate\n"
        f"try: validate.validate_audio({str(empty)!r})\n"
        "except validate.AudioValidationError: pass\n"
        "assert 'faster_whisper' not in sys.modules\n"
        "assert 'ctranslate2' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_cli(tmp_path):
    good = write_wav(tmp_path / "ok.wav", tone(2.0))
    silent = write_wav(tmp_path / "silence.wav", np.zeros(SR * 3))

    ok = subprocess.run(
        [sys.executable, "validate.py", str(good)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert ok.returncode == 0
    assert ok.stdout.startswith("OK")

    bad = subprocess.run(
        [sys.executable, "validate.py", str(good), str(silent)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert bad.returncode == 1
    assert validate.MSG_SILENT in bad.stdout
