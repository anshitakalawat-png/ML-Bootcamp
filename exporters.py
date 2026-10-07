"""Write the outputs of one processed recording.

export_all() writes, into one folder:
  raw_transcript.txt      the raw Whisper text, byte for byte
  refined_transcript.txt  the refined text from stage 2
  meeting_record.json     the meeting record from stage 3
  meeting_record.md       the same record, for people
  subtitles.srt           subtitles from the raw Whisper segments

The JSON and Markdown records are rendered from one record object built by
build_record(), so their decisions and action items cannot differ. The
exporters only format: they never change decisions, tasks, owners,
deadlines or evidence. The one normalisation is that a blank or missing
owner/deadline is written as "unspecified".

All files are UTF-8. Files are written with newline="" so text is not
altered by platform line-ending conversion.
"""

from __future__ import annotations

import copy
import json
import os
import re

from minutes import UNSPECIFIED

FILENAMES = {
    "raw": "raw_transcript.txt",
    "refined": "refined_transcript.txt",
    "json": "meeting_record.json",
    "markdown": "meeting_record.md",
    "srt": "subtitles.srt",
}

RECORD_KEYS = ("summary", "minutes", "key_decisions", "action_items", "open_questions")

# Characters that Markdown could interpret as formatting.
_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>#|])")


def build_record(minutes_result: dict) -> dict:
    """The single record object both meeting_record files are rendered from.

    `minutes_result` is the dict returned by minutes.generate_minutes.
    """
    source = minutes_result["record"]
    record = {key: copy.deepcopy(source[key]) for key in RECORD_KEYS}
    for item in record["action_items"]:
        for field in ("owner", "deadline"):
            if not str(item.get(field) or "").strip():
                item[field] = UNSPECIFIED
    record["warnings"] = list(minutes_result.get("warnings", []))
    return record


def export_raw_transcript(raw: dict, path: str | os.PathLike) -> str:
    """Write the raw Whisper text exactly as stt.transcribe returned it."""
    return _write(path, raw["text"])


def export_refined_transcript(refined: dict, path: str | os.PathLike) -> str:
    """Write the refined transcript returned by refine.refine."""
    return _write(path, refined["refined_transcript"])


def export_meeting_record_json(record: dict, path: str | os.PathLike) -> str:
    """Write a record from build_record() as JSON."""
    return _write(path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")


def export_meeting_record_markdown(record: dict, path: str | os.PathLike) -> str:
    """Write a record from build_record() as Markdown."""
    return _write(path, render_markdown(record))


def export_srt(raw: dict, path: str | os.PathLike) -> str:
    """Write subtitles from the raw Whisper segments."""
    return _write(path, render_srt(raw["segments"]))


def export_all(raw: dict, refined: dict, minutes_result: dict, out_dir: str | os.PathLike) -> dict:
    """Write all five files into out_dir. Returns {kind: path}."""
    os.makedirs(out_dir, exist_ok=True)
    record = build_record(minutes_result)

    def target(kind: str) -> str:
        return os.path.join(out_dir, FILENAMES[kind])

    return {
        "raw": export_raw_transcript(raw, target("raw")),
        "refined": export_refined_transcript(refined, target("refined")),
        "json": export_meeting_record_json(record, target("json")),
        "markdown": export_meeting_record_markdown(record, target("markdown")),
        "srt": export_srt(raw, target("srt")),
    }


def render_markdown(record: dict) -> str:
    lines = ["# Meeting Record", ""]

    lines += ["## Summary", "", _md(record["summary"]), ""]

    lines += ["## Minutes", ""]
    if not record["minutes"]:
        lines += ["_None recorded._", ""]
    for topic in record["minutes"]:
        lines += [f"### {_md(topic['topic'])}", ""]
        lines += [f"- {_md(point)}" for point in topic["points"]]
        lines.append("")

    lines += ["## Key Decisions", ""]
    if not record["key_decisions"]:
        lines += ["_None recorded._", ""]
    for n, item in enumerate(record["key_decisions"], start=1):
        lines += [
            f"{n}. **Decision:** {_md(item['decision'])}",
            f"   - **Evidence:** \"{_md(item['evidence'])}\"",
        ]
    if record["key_decisions"]:
        lines.append("")

    lines += ["## Action Items", ""]
    if not record["action_items"]:
        lines += ["_None recorded._", ""]
    for n, item in enumerate(record["action_items"], start=1):
        lines += [
            f"{n}. **Task:** {_md(item['task'])}",
            f"   - **Owner:** {_md(item['owner'])}",
            f"   - **Deadline:** {_md(item['deadline'])}",
            f"   - **Evidence:** \"{_md(item['evidence'])}\"",
        ]
    if record["action_items"]:
        lines.append("")

    lines += ["## Open Questions", ""]
    if not record["open_questions"]:
        lines += ["_None recorded._", ""]
    lines += [f"- {_md(q)}" for q in record["open_questions"]]
    if record["open_questions"]:
        lines.append("")

    if record.get("warnings"):
        lines += ["## Review Notes", "", "Automatic checks flagged the following:", ""]
        lines += [f"- {_md(w)}" for w in record["warnings"]]
        lines.append("")

    return "\n".join(lines)


def render_srt(segments: list[dict]) -> str:
    """Number cues from 1 and use each segment's own start/end times.

    Whitespace-only segments are skipped: an empty text line would end the
    cue and make the file invalid.
    """
    cues = []
    for seg in segments:
        if not seg["text"].strip():
            continue
        n = len(cues) + 1
        cues.append(f"{n}\n{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n{seg['text']}\n")
    return "\n".join(cues)


def srt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm, rounded to the nearest millisecond."""
    total_ms = round(seconds * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _md(text: str) -> str:
    """Escape Markdown formatting characters and keep a value on one line.

    Nothing is removed: escapes and <br> (for a line break inside a value)
    render as the original characters, so the record reads exactly as in
    the JSON.
    """
    text = _MD_SPECIAL.sub(r"\\\1", text)
    return text.replace("\r\n", "\n").replace("\n", "<br>")


def _write(path: str | os.PathLike, content: str) -> str:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return os.fspath(path)
