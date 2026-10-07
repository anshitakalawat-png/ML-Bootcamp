"""Stage 2: transcript refinement (LLM #1).

refine(raw, vocabulary) takes the raw transcript from stt.transcribe and
returns a new, separate refined transcript. The raw transcript is only read,
never modified.

Long transcripts are split into chunks at sentence boundaries, refined one
chunk at a time and rejoined in order. Each refined chunk is checked in code:
if its length changed drastically, or its numbers or negations differ from
the raw text, the refinement of that chunk is rejected, the raw text is kept
for it, and a warning is returned for the UI to show.

Configuration (.env or environment):
  ANTHROPIC_API_KEY   required
  REFINE_MODEL        default "claude-sonnet-5-5"
  REFINE_EFFORT       default "medium" (low | medium | high | xhigh | max)
  REFINE_CHUNK_CHARS  default 6000

CLI:  python refine.py <transcript.json from stt.py> [--vocab "Term A, Term B"] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import llm

PROMPT_PATH = Path(__file__).parent / "prompts" / "refine.txt"

# A refined chunk is rejected if its length is outside these bounds relative
# to the raw chunk, unless the difference is under LENGTH_SLACK_CHARS (short
# chunks change proportionally more when e.g. "a p i" becomes "API").
MIN_LENGTH_RATIO = 0.7
MAX_LENGTH_RATIO = 1.4
LENGTH_SLACK_CHARS = 25

SCHEMA = {
    "type": "object",
    "properties": {
        "refined_transcript": {"type": "string"},
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heard": {"type": "string"},
                    "corrected": {"type": "string"},
                },
                "required": ["heard", "corrected"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["refined_transcript", "corrections"],
    "additionalProperties": False,
}

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[a-z']+")
_NEGATIONS = {"not", "no", "never", "cannot", "none", "nobody", "nothing", "neither", "nor", "nowhere"}


class RefinementError(Exception):
    """Stage 2 could not run. `message` is user-facing."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _env(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


def refine(raw: dict, vocabulary: list[str] | None = None, model: str | None = None) -> dict:
    """Refine the raw transcript.

    Returns {
      "refined_transcript": str,
      "corrections": [{"heard", "corrected"}],
      "warnings": [str],          # show these to the user
      "fallback_chunks": [int],   # 1-based chunks where the raw text was kept
      "chunks": int,
      "model": str,
    }
    Raises RefinementError if the model cannot be reached or keeps
    returning malformed JSON.
    """
    text = raw.get("text", "")
    if not text.strip():
        raise RefinementError("The raw transcript is empty, so there is nothing to refine.")

    model = model or _env("REFINE_MODEL", "claude-sonnet-5-5")
    effort = _env("REFINE_EFFORT", "medium")
    max_chars = int(_env("REFINE_CHUNK_CHARS", "6000"))
    vocab = _clean_vocabulary(vocabulary)
    system = PROMPT_PATH.read_text(encoding="utf-8")

    chunks = split_into_chunks(text, max_chars)
    refined_parts, corrections, warnings, fallback_chunks = [], [], [], []

    for i, chunk in enumerate(chunks, start=1):
        try:
            reply = llm.complete_json(
                model=model,
                system=system,
                user=_user_message(chunk, vocab),
                schema=SCHEMA,
                check=_check_reply,
                effort=effort,
            )
        except llm.LLMError as err:
            where = f" (part {i} of {len(chunks)})" if len(chunks) > 1 else ""
            raise RefinementError(f"{err.message}{where}") from err

        refined = reply["refined_transcript"].strip()
        problems = meaning_problems(chunk, refined, vocab)
        if problems:
            where = f"Part {i} of {len(chunks)}" if len(chunks) > 1 else "The transcript"
            warnings.append(
                f"{where}: the refined text was rejected because {'; '.join(problems)}. "
                "The raw transcript is shown for this part instead."
            )
            fallback_chunks.append(i)
            refined_parts.append(chunk)
            continue

        refined_parts.append(refined)
        corrections.extend(_real_corrections(reply["corrections"], chunk, refined))

    return {
        "refined_transcript": " ".join(refined_parts),
        "corrections": corrections,
        "warnings": warnings,
        "fallback_chunks": fallback_chunks,
        "chunks": len(chunks),
        "model": model,
    }


def split_into_chunks(text: str, max_chars: int) -> list[str]:
    """Split at sentence boundaries into chunks of at most max_chars.

    A single sentence longer than max_chars (Whisper sometimes produces long
    unpunctuated runs) is split at word boundaries instead.
    """
    pieces = []
    for sentence in _SENTENCE_END.split(text.strip()):
        pieces.extend(_split_long(sentence, max_chars))

    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    parts, current = [], ""
    for word in sentence.split():
        if current and len(current) + 1 + len(word) > max_chars:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        parts.append(current)
    return parts


def meaning_problems(raw: str, refined: str, vocabulary: list[str]) -> list[str]:
    """Return reasons the refined text cannot be trusted (empty if it is fine)."""
    problems = []

    ratio = len(refined) / len(raw)
    if abs(len(refined) - len(raw)) > LENGTH_SLACK_CHARS and not (
        MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO
    ):
        problems.append(f"its length changed to {ratio:.0%} of the original")

    # Numbers inside vocabulary terms (e.g. "GPT-4") may legitimately appear
    # after a correction, so they are not compared.
    if _numbers(raw, vocabulary) != _numbers(refined, vocabulary):
        problems.append("numbers were changed")

    if _negations(raw) != _negations(refined):
        problems.append("a negation (not, no, never, ...) was added or removed")

    return problems


def _numbers(text: str, vocabulary: list[str]) -> Counter:
    for term in vocabulary:
        text = re.sub(re.escape(term), " ", text, flags=re.IGNORECASE)
    return Counter(n.replace(",", "") for n in _NUMBER.findall(text))


def _negations(text: str) -> int:
    words = _WORD.findall(text.lower().replace("’", "'"))
    return sum(1 for w in words if w in _NEGATIONS or w.endswith("n't"))


def _check_reply(data: dict) -> None:
    if not isinstance(data.get("refined_transcript"), str) or not data["refined_transcript"].strip():
        raise ValueError('"refined_transcript" is missing or empty')
    corrections = data.get("corrections")
    if not isinstance(corrections, list):
        raise ValueError('"corrections" must be a list')
    for c in corrections:
        if not (isinstance(c, dict) and isinstance(c.get("heard"), str) and isinstance(c.get("corrected"), str)):
            raise ValueError('each correction needs string fields "heard" and "corrected"')


def _real_corrections(corrections: list[dict], raw: str, refined: str) -> list[dict]:
    """Keep only corrections that actually happened in this chunk."""
    kept = []
    for c in corrections:
        heard, corrected = c["heard"].strip(), c["corrected"].strip()
        if heard and heard != corrected and heard in raw and corrected in refined:
            kept.append({"heard": heard, "corrected": corrected})
    return kept


def _clean_vocabulary(vocabulary: list[str] | None) -> list[str]:
    seen, terms = set(), []
    for term in vocabulary or []:
        term = term.strip()
        if term and term.lower() not in seen:
            seen.add(term.lower())
            terms.append(term)
    return terms


def _user_message(chunk: str, vocabulary: list[str]) -> str:
    vocab = "\n".join(vocabulary)
    return f"<vocabulary>\n{vocab}\n</vocabulary>\n<transcript>\n{chunk}\n</transcript>"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Refine a raw transcript (stage 2).")
    parser.add_argument("transcript", help="JSON written by: python stt.py AUDIO --json OUT")
    parser.add_argument("--vocab", default="", help='comma-separated terms, e.g. "Priya, Kubernetes"')
    parser.add_argument("--model", help="model name (default from .env)")
    parser.add_argument("--json", metavar="OUT", help="also write the result as JSON")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    with open(args.transcript, encoding="utf-8") as f:
        raw = json.load(f)

    try:
        result = refine(raw, vocabulary=args.vocab.split(","), model=args.model)
    except RefinementError as err:
        print(f"FAIL  refinement: {err.message}")
        return 1

    print("--- raw ---")
    print(raw["text"].strip())
    print("\n--- refined ---")
    print(result["refined_transcript"])
    print("\n--- corrections ---")
    for c in result["corrections"]:
        print(f"  {c['heard']!r} -> {c['corrected']!r}")
    if not result["corrections"]:
        print("  (none)")
    for w in result["warnings"]:
        print(f"\nWARNING: {w}")
    print(f"\n{result['chunks']} chunk(s), model {result['model']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
