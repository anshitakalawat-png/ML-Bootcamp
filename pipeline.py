"""The full workflow: validate -> transcribe -> refine -> minutes.

run_pipeline() runs the four stages in that order. Each stage runs only if
the one before it succeeded. Any failure raises PipelineError, which names
the stage that failed; nothing after that stage runs.

CLI:  python pipeline.py <audio file> [--vocab "Term A, Term B"] [--out DIR]
"""

from __future__ import annotations

import argparse
import copy
import sys
from typing import Callable

import exporters
import minutes
import refine
import stt
import validate

STAGES = {
    0: "validation",
    1: "transcription",
    2: "refinement",
    3: "documentation",
}

STATUS = {
    0: "Validating audio...",
    1: "Transcribing...",
    2: "Refining transcript...",
    3: "Generating minutes...",
}
STATUS_DONE = "Done"

# The errors each stage raises on purpose, with a user-facing `.message`.
_EXPECTED = {
    0: validate.AudioValidationError,
    1: validate.AudioValidationError,  # "No speech was detected" after Whisper runs
    2: refine.RefinementError,
    3: minutes.MinutesError,
}


class PipelineError(Exception):
    """A stage failed. `message` is user-facing and names the stage."""

    def __init__(self, stage: int, reason: str):
        self.stage = stage
        self.stage_name = STAGES[stage]
        self.reason = reason
        self.message = f"Stage {stage} ({self.stage_name}) failed: {reason}"
        super().__init__(self.message)


def _copied(name: str) -> property:
    return property(lambda self: copy.deepcopy(getattr(self, name)))


class PipelineResult:
    """Read-only result of one run.

    The result keeps its own copy of everything, and the list and dict
    attributes return a fresh copy on each access. Callers get ordinary,
    JSON-serialisable values, but nothing they do to them can change the
    result (in particular the raw transcript). Setting an attribute raises.

      raw_text            str, exactly as Whisper produced it
      segments            [{"start", "end", "text"}] from Whisper
      refined_transcript  str
      corrections         [{"heard", "corrected"}]
      record              {summary, minutes, key_decisions, action_items, open_questions}
      warnings            [str] from refinement and documentation, for the user
    """

    __slots__ = ("_raw_text", "_segments", "_refined_transcript", "_corrections", "_record", "_warnings")

    def __init__(self, *, raw_text, segments, refined_transcript, corrections, record, warnings):
        values = {
            "raw_text": raw_text,
            "segments": segments,
            "refined_transcript": refined_transcript,
            "corrections": corrections,
            "record": record,
            "warnings": warnings,
        }
        for name, value in values.items():
            object.__setattr__(self, f"_{name}", copy.deepcopy(value))

    def __setattr__(self, name, value):
        raise AttributeError("PipelineResult is read-only")

    raw_text = property(lambda self: self._raw_text)
    refined_transcript = property(lambda self: self._refined_transcript)
    segments = _copied("_segments")
    corrections = _copied("_corrections")
    record = _copied("_record")
    warnings = _copied("_warnings")

    def export(self, out_dir: str) -> dict:
        """Write the five output files. Returns {kind: path}."""
        return exporters.export_all(
            raw={"segments": self.segments, "text": self.raw_text},
            refined={"refined_transcript": self.refined_transcript},
            minutes_result={"record": self.record, "warnings": self.warnings},
            out_dir=out_dir,
        )


def run_pipeline(
    path: str,
    vocabulary: list[str] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Run all stages on one recording. Raises PipelineError on any failure."""
    status = on_status or (lambda _msg: None)

    status(STATUS[0])
    audio = _run_stage(0, validate.validate_audio, path)

    status(STATUS[1])
    raw = _run_stage(1, stt.transcribe_validated, audio)

    status(STATUS[2])
    # Stage 2 gets a copy, so the raw transcript cannot be changed by it.
    refined = _run_stage(2, refine.refine, copy.deepcopy(raw), vocabulary)

    status(STATUS[3])
    documented = _run_stage(3, minutes.generate_minutes, refined)

    status(STATUS_DONE)
    return PipelineResult(
        raw_text=raw["text"],
        segments=raw["segments"],
        refined_transcript=refined["refined_transcript"],
        corrections=refined["corrections"],
        record=documented["record"],
        warnings=refined["warnings"] + documented["warnings"],
    )


def _run_stage(stage: int, func: Callable, *args):
    try:
        return func(*args)
    except _EXPECTED[stage] as err:
        raise PipelineError(stage, err.message) from err
    except Exception as err:  # anything else is still reported against its stage
        raise PipelineError(stage, f"unexpected error ({type(err).__name__}: {err})") from err


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the full meeting pipeline on one recording.")
    parser.add_argument("audio")
    parser.add_argument("--vocab", default="", help='comma-separated terms, e.g. "Priya, Kubernetes"')
    parser.add_argument("--out", default="output", help="folder for the output files (default: output)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    try:
        result = run_pipeline(args.audio, vocabulary=args.vocab.split(","), on_status=print)
    except PipelineError as err:
        print(f"FAIL  {err.message}")
        return 1

    for w in result.warnings:
        print(f"WARNING: {w}")
    for kind, path in result.export(args.out).items():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
