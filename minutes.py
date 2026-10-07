"""Stage 3: meeting documentation (LLM #2).

generate_minutes(refined) turns the refined transcript into the meeting
record: summary, minutes by topic, key decisions, action items and open
questions. It uses its own prompt (prompts/minutes.txt) and by default a
different model from stage 2.

After the model replies, the record is checked in code:
- an owner or deadline whose wording does not appear in the transcript is
  replaced with "unspecified" and a warning is returned;
- evidence that is not found in the transcript gets a warning.

Configuration (.env or environment):
  GEMINI_API_KEY      required
  MINUTES_MODEL       default "gemini-3.1-flash-lite"
  MINUTES_EFFORT      default "high" (minimal | low | medium | high)

CLI:  python minutes.py <refined.json from refine.py | transcript .txt> [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import llm

PROMPT_PATH = Path(__file__).parent / "prompts" / "minutes.txt"
UNSPECIFIED = "unspecified"

# Values a model may use to mean "not stated"; all become UNSPECIFIED.
_UNSTATED = {"", "unspecified", "none", "n/a", "na", "unknown", "tbd", "not specified", "not stated", "-"}

_STRING = {"type": "string"}
SCHEMA = {
    "type": "object",
    "properties": {
        "summary": _STRING,
        "minutes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"topic": _STRING, "points": {"type": "array", "items": _STRING}},
                "required": ["topic", "points"],
                "additionalProperties": False,
            },
        },
        "key_decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"decision": _STRING, "evidence": _STRING},
                "required": ["decision", "evidence"],
                "additionalProperties": False,
            },
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"task": _STRING, "owner": _STRING, "deadline": _STRING, "evidence": _STRING},
                "required": ["task", "owner", "deadline", "evidence"],
                "additionalProperties": False,
            },
        },
        "open_questions": {"type": "array", "items": _STRING},
    },
    "required": ["summary", "minutes", "key_decisions", "action_items", "open_questions"],
    "additionalProperties": False,
}


class MinutesError(Exception):
    """Stage 3 could not run. `message` is user-facing."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _env(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


def generate_minutes(refined: str | dict, model: str | None = None) -> dict:
    """Write the meeting record from the refined transcript.

    `refined` is the dict returned by refine.refine, or the transcript text.
    Returns {"record": {summary, minutes, key_decisions, action_items,
    open_questions}, "warnings": [str], "model": str}.
    Raises MinutesError if the model cannot be reached or keeps returning
    malformed JSON.
    """
    if isinstance(refined, dict):
        if "refined_transcript" not in refined:
            raise MinutesError("Expected the refined transcript from stage 2 (refinement).")
        transcript = refined["refined_transcript"]
    else:
        transcript = refined
    if not transcript.strip():
        raise MinutesError("The transcript is empty, so there is nothing to document.")

    model = model or _env("MINUTES_MODEL", "gemini-3.1-flash-lite")
    try:
        record = llm.complete_json(
            model=model,
            system=PROMPT_PATH.read_text(encoding="utf-8"),
            user=f"<transcript>\n{transcript}\n</transcript>",
            schema=SCHEMA,
            check=check_record,
            effort=_env("MINUTES_EFFORT", "high"),
        )
    except llm.LLMError as err:
        raise MinutesError(err.message) from err

    warnings = verify_against_transcript(record, transcript)
    return {"record": record, "warnings": warnings, "model": model}


def check_record(data: dict) -> None:
    """Raise ValueError if the reply does not match SCHEMA."""
    for key in SCHEMA["required"]:
        if key not in data:
            raise ValueError(f'missing key "{key}"')
    if not isinstance(data["summary"], str) or not data["summary"].strip():
        raise ValueError('"summary" must be a non-empty string')

    item_fields = {
        "minutes": ("topic",),
        "key_decisions": ("decision", "evidence"),
        "action_items": ("task", "owner", "deadline", "evidence"),
    }
    for key, fields in item_fields.items():
        if not isinstance(data[key], list):
            raise ValueError(f'"{key}" must be a list')
        for i, item in enumerate(data[key]):
            if not isinstance(item, dict):
                raise ValueError(f'"{key}"[{i}] must be an object')
            for field in fields:
                if not isinstance(item.get(field), str):
                    raise ValueError(f'"{key}"[{i}].{field} must be a string')
    for i, topic in enumerate(data["minutes"]):
        points = topic.get("points")
        if not isinstance(points, list) or not all(isinstance(p, str) for p in points):
            raise ValueError(f'"minutes"[{i}].points must be a list of strings')
    if not isinstance(data["open_questions"], list) or not all(isinstance(q, str) for q in data["open_questions"]):
        raise ValueError('"open_questions" must be a list of strings')


def verify_against_transcript(record: dict, transcript: str) -> list[str]:
    """Fix owners/deadlines not found in the transcript; return warnings.

    Modifies `record` in place.
    """
    text = _normalise(transcript)
    warnings = []

    for item in record["action_items"]:
        task = item["task"]
        for field in ("owner", "deadline"):
            value = item[field].strip()
            if value.lower() in _UNSTATED:
                item[field] = UNSPECIFIED
                continue
            parts = _owner_parts(value) if field == "owner" else [value]
            missing = [p for p in parts if not _appears(p, text)]
            if missing:
                item[field] = UNSPECIFIED
                warnings.append(
                    f'Action item "{task}": {field} "{value}" does not appear in the '
                    f'transcript, so it was changed to "{UNSPECIFIED}".'
                )
            else:
                item[field] = value

    for kind, label in (("key_decisions", "decision"), ("action_items", "task")):
        for item in record[kind]:
            if not _appears(item["evidence"], text):
                warnings.append(
                    f'The evidence for {label} "{item[label]}" was not found word for word '
                    "in the transcript. Check this item against the recording."
                )
    return warnings


def _owner_parts(owner: str) -> list[str]:
    # "Priya and Rahul" or "Priya, Rahul": each name must appear on its own.
    parts = re.split(r"\s*(?:,|&|/|\band\b)\s*", owner)
    return [p for p in parts if p.strip()]


def _normalise(text: str) -> str:
    """Lowercase, unify quotes, and turn punctuation runs into single spaces."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    return " " + " ".join(re.findall(r"[a-z0-9']+", text)) + " "


def _appears(phrase: str, normalised_text: str) -> bool:
    words = _normalise(phrase).strip()
    return bool(words) and f" {words} " in normalised_text


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Write the meeting record (stage 3).")
    parser.add_argument("transcript", help="refined JSON from refine.py --json, or a .txt transcript")
    parser.add_argument("--model", help="model name (default from .env)")
    parser.add_argument("--json", metavar="OUT", help="also write the result as JSON")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    with open(args.transcript, encoding="utf-8") as f:
        source = json.load(f) if args.transcript.lower().endswith(".json") else f.read()

    try:
        result = generate_minutes(source, model=args.model)
    except MinutesError as err:
        print(f"FAIL  documentation: {err.message}")
        return 1

    print(json.dumps(result["record"], ensure_ascii=False, indent=2))
    for w in result["warnings"]:
        print(f"\nWARNING: {w}")
    print(f"\nmodel {result['model']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
