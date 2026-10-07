"""Stage 1: speech-to-text with faster-whisper.

transcribe(path) always runs stage 0 (validate_audio) first, so the Whisper
model is only loaded for a file that passed validation.

The output is the raw transcript. Segment texts and the full text are exactly
what Whisper produced: no stripping, joining or other cleanup. Later stages
must never modify it.

Configuration (.env or environment):
  WHISPER_MODEL_SIZE    default "small"
  WHISPER_DEVICE        "auto" (default), "cpu" or "cuda"
  WHISPER_COMPUTE_TYPE  default "int8" on CPU, "float16" on CUDA
  WHISPER_BEAM_SIZE     default 5

CLI:  python stt.py <audio file> [--model SIZE] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import lru_cache
from typing import Callable

from dotenv import load_dotenv

from validate import (
    AudioValidationError,
    ValidatedAudio,
    ensure_speech_detected,
    validate_audio,
)

load_dotenv()

LANGUAGE = "en"


def _env(name: str, default: str) -> str:
    # An empty value in .env (e.g. "WHISPER_COMPUTE_TYPE=") means "use default".
    return os.getenv(name, "").strip() or default


def _config() -> dict:
    device = _env("WHISPER_DEVICE", "auto").lower()
    if device == "auto":
        import ctranslate2

        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    default_compute = "float16" if device == "cuda" else "int8"
    return {
        "model_size": _env("WHISPER_MODEL_SIZE", "small"),
        "device": device,
        "compute_type": _env("WHISPER_COMPUTE_TYPE", default_compute),
        "beam_size": int(_env("WHISPER_BEAM_SIZE", "5")),
    }


@lru_cache(maxsize=2)
def _load_model(model_size: str, device: str, compute_type: str):
    # Imported here so that importing stt (or failing validation) never pays
    # for loading faster-whisper.
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe(
    path: str | os.PathLike,
    model_size: str | None = None,
    on_segment: Callable[[dict], None] | None = None,
) -> dict:
    """Validate the file, then transcribe it.

    Returns {"segments": [{"start", "end", "text"}], "text": full_plain_text}.
    Raises AudioValidationError if validation fails or no speech is found.
    """
    audio = validate_audio(path)  # stage 0: must pass before the model loads
    return transcribe_validated(audio, model_size=model_size, on_segment=on_segment)


def transcribe_validated(
    audio: ValidatedAudio,
    model_size: str | None = None,
    on_segment: Callable[[dict], None] | None = None,
) -> dict:
    """Transcribe audio that has already passed validate_audio.

    on_segment is called with each segment as Whisper produces it, for
    progress display.
    """
    cfg = _config()
    if model_size:
        cfg["model_size"] = model_size
    model = _load_model(cfg["model_size"], cfg["device"], cfg["compute_type"])

    whisper_segments, _info = model.transcribe(
        audio.samples,
        language=LANGUAGE,
        beam_size=cfg["beam_size"],
        vad_filter=True,
    )

    produced = []
    for seg in whisper_segments:  # lazy: decoding happens while iterating
        produced.append(seg)
        if on_segment:
            on_segment(_as_dict(seg))

    ensure_speech_detected(produced)

    segments = [_as_dict(seg) for seg in produced]
    # Whisper segment texts carry their own leading spaces, so joining with ""
    # gives Whisper's text as is.
    return {"segments": segments, "text": "".join(s["text"] for s in segments)}


def _as_dict(seg) -> dict:
    return {"start": seg.start, "end": seg.end, "text": seg.text}


def _fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Transcribe an audio file.")
    parser.add_argument("audio")
    parser.add_argument("--model", help="Whisper model size (default from .env, else small)")
    parser.add_argument("--json", metavar="OUT", help="also write the result as JSON")
    args = parser.parse_args(argv)

    # Windows consoles default to cp1252, which cannot print every character
    # Whisper may output.
    sys.stdout.reconfigure(encoding="utf-8")

    def show(seg: dict) -> None:
        print(f"[{_fmt(seg['start'])} -> {_fmt(seg['end'])}]{seg['text']}", flush=True)

    started = time.perf_counter()
    try:
        result = transcribe(args.audio, model_size=args.model, on_segment=show)
    except AudioValidationError as err:
        print(f"FAIL  {args.audio}: {err.message}")
        return 1
    elapsed = time.perf_counter() - started

    print("\n--- full text ---")
    print(result["text"])
    print(f"\n{len(result['segments'])} segments in {elapsed:.1f} s")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
