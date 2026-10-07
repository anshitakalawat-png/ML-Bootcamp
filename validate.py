"""Stage 0: audio validation.

Runs before anything else in the pipeline. If the file is missing, empty,
unsupported, unreadable, too short or silent, it raises AudioValidationError
and no later stage (transcription, refinement, documentation) may run.

This module must stay light: it only imports PyAV and numpy, never
faster-whisper, so a bad file is rejected before any model is loaded.

CLI:  python validate.py <audio file> [<audio file> ...]
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable

import av
import numpy as np

SUPPORTED_EXTENSIONS = ("wav", "mp3", "m4a", "flac", "ogg", "aac")

# Audio is decoded once to the format faster-whisper expects, so stage 1 can
# reuse the samples instead of decoding the file again.
SAMPLE_RATE = 16_000

MIN_DURATION_S = 1.0

# Silence detection. Audio is split into short frames; a frame counts as
# active if its loudness is above ACTIVE_FRAME_DBFS. The file is rejected as
# silent if the whole file is quieter than MIN_OVERALL_DBFS, or if fewer than
# MIN_ACTIVE_FRACTION of its frames are active.
FRAME_MS = 30
ACTIVE_FRAME_DBFS = -45.0
MIN_OVERALL_DBFS = -60.0
MIN_ACTIVE_FRACTION = 0.02

MSG_EMPTY = "The uploaded file is empty."
MSG_UNSUPPORTED = (
    "Unsupported file format '.{ext}'. Supported formats: "
    + ", ".join(SUPPORTED_EXTENSIONS)
    + "."
)
MSG_UNREADABLE = (
    "The audio file is unreadable or corrupted. "
    "Please upload a valid audio recording."
)
MSG_TOO_SHORT = (
    "The recording is too short ({duration:.2f} s). "
    "Please upload audio of at least {min:.0f} second."
)
MSG_SILENT = "No audio signal detected. The file appears to be silent."
MSG_NO_SPEECH = "No speech was detected in this recording."


class AudioValidationError(Exception):
    """Raised when the audio cannot go through the pipeline.

    `message` is safe to show to the user as is.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class ValidatedAudio:
    path: str
    samples: np.ndarray  # mono float32 in [-1, 1] at SAMPLE_RATE
    sample_rate: int
    duration_s: float
    rms_dbfs: float
    active_fraction: float


def validate_audio(path: str | os.PathLike) -> ValidatedAudio:
    """Check the file and decode it. Raises AudioValidationError on failure."""
    path = os.fspath(path)

    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise AudioValidationError(MSG_EMPTY)

    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext not in SUPPORTED_EXTENSIONS:
        raise AudioValidationError(MSG_UNSUPPORTED.format(ext=ext or "(none)"))

    samples = _decode(path)

    duration_s = len(samples) / SAMPLE_RATE
    if duration_s < MIN_DURATION_S:
        raise AudioValidationError(
            MSG_TOO_SHORT.format(duration=duration_s, min=MIN_DURATION_S)
        )

    rms_dbfs, active_fraction = _signal_stats(samples)
    if rms_dbfs < MIN_OVERALL_DBFS or active_fraction < MIN_ACTIVE_FRACTION:
        raise AudioValidationError(MSG_SILENT)

    return ValidatedAudio(
        path=path,
        samples=samples,
        sample_rate=SAMPLE_RATE,
        duration_s=duration_s,
        rms_dbfs=rms_dbfs,
        active_fraction=active_fraction,
    )


def ensure_speech_detected(segments: Iterable) -> list:
    """Final check, called by stage 1 right after Whisper runs.

    `segments` are faster-whisper segments (anything with a `.text`).
    Returns them as a list so the caller does not consume a generator twice.
    """
    segments = list(segments)
    if not any(seg.text.strip() for seg in segments):
        raise AudioValidationError(MSG_NO_SPEECH)
    return segments


def _decode(path: str) -> np.ndarray:
    """Decode any supported file to mono float32 at SAMPLE_RATE."""
    try:
        with av.open(path) as container:
            if not container.streams.audio:
                raise AudioValidationError(MSG_UNREADABLE)
            stream = container.streams.audio[0]
            resampler = av.audio.resampler.AudioResampler(
                format="s16", layout="mono", rate=SAMPLE_RATE
            )
            chunks = []
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1))
            for out in resampler.resample(None):  # flush
                chunks.append(out.to_ndarray().reshape(-1))
    except AudioValidationError:
        raise
    except Exception as exc:  # any decoder failure means the file is unusable
        raise AudioValidationError(MSG_UNREADABLE) from exc

    if not chunks:
        raise AudioValidationError(MSG_UNREADABLE)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _signal_stats(samples: np.ndarray) -> tuple[float, float]:
    """Return (overall RMS in dBFS, fraction of frames above ACTIVE_FRAME_DBFS)."""
    eps = 1e-10
    rms_dbfs = 20 * np.log10(np.sqrt(np.mean(samples**2)) + eps)

    frame_len = SAMPLE_RATE * FRAME_MS // 1000
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return float(rms_dbfs), 0.0
    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_dbfs = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + eps)
    active_fraction = float(np.mean(frame_dbfs > ACTIVE_FRAME_DBFS))
    return float(rms_dbfs), active_fraction


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python validate.py <audio file> [<audio file> ...]")
        return 2
    failed = False
    for path in argv:
        try:
            audio = validate_audio(path)
        except AudioValidationError as err:
            failed = True
            print(f"FAIL  {path}: {err.message}")
        else:
            print(
                f"OK    {path}: {audio.duration_s:.2f} s, "
                f"RMS {audio.rms_dbfs:.1f} dBFS, "
                f"{audio.active_fraction:.0%} active frames"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
