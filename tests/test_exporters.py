import copy
import json
import re

import pytest

import exporters
from minutes import UNSPECIFIED

RAW = {
    "segments": [
        {"start": 0.0, "end": 2.12, "text": " Good morning everyone."},
        {"start": 2.12, "end": 4.2, "text": " PREA will update the cube control config by Friday."},
        {"start": 4.2, "end": 7.92, "text": "   "},  # whitespace-only, skipped in SRT
        {"start": 7.92, "end": 11.08, "text": " We will not migrate the database this sprint."},
        {"start": 3725.5, "end": 3729.999, "text": " The budget is ₹42,000 – café costs extra."},
    ],
}
RAW["text"] = "".join(s["text"] for s in RAW["segments"])

REFINED = {
    "refined_transcript": "Good morning everyone. Priya will update the kubectl config by Friday. "
                          "We will not migrate the database this sprint. The budget is ₹42,000 – café costs extra.",
    "corrections": [{"heard": "PREA", "corrected": "Priya"}],
    "warnings": [],
}

MINUTES_RESULT = {
    "record": {
        "summary": "Review of the *deployment* plan and the_budget.",
        "minutes": [
            {"topic": "Deployment #1", "points": ["Priya will update the kubectl config.", "Uses <k8s> [beta]."]},
            {"topic": "Budget", "points": ["The budget is ₹42,000."]},
        ],
        "key_decisions": [
            {"decision": "No database migration this sprint.",
             "evidence": "We will not migrate the database this sprint."},
            {"decision": "Keep the \"café\" vendor | for now",
             "evidence": "café costs extra"},
        ],
        "action_items": [
            {"task": "Update the kubectl config", "owner": "Priya", "deadline": "by Friday",
             "evidence": "Priya will update the kubectl config by Friday"},
            {"task": "Review costs", "owner": UNSPECIFIED, "deadline": UNSPECIFIED,
             "evidence": "café costs extra"},
            {"task": "Check *all* line\nbreaks", "owner": "", "deadline": None,
             "evidence": "Good morning everyone."},
        ],
        "open_questions": ["Who owns the demo?"],
    },
    "warnings": ['Action item "Review costs": owner "Arjun" does not appear in the transcript, so it was changed to "unspecified".'],
    "model": "gemini-3.8-flash",
}


@pytest.fixture
def outputs(tmp_path):
    paths = exporters.export_all(RAW, REFINED, MINUTES_RESULT, tmp_path / "run")
    return {kind: open(path, encoding="utf-8", newline="").read() for kind, path in paths.items()}


def md_section(md: str, title: str) -> str:
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", md, re.M | re.S)
    assert match, f"section {title!r} missing"
    return match.group(1)


def unescape(value: str) -> str:
    return re.sub(r"\\(.)", r"\1", value.replace("<br>", "\n"))


def md_items(section: str, fields: list[str]) -> list[dict]:
    """Parse numbered items: '1. **Field:** value' then '   - **Field:** value' lines."""
    items = []
    for line in section.splitlines():
        m = re.match(r"^(?:\d+\. |   - )\*\*(\w+):\*\* (.*)$", line)
        if not m:
            continue
        key, value = m.group(1).lower(), m.group(2)
        if key == fields[0]:
            items.append({})
        if key == "evidence":
            assert value.startswith('"') and value.endswith('"')
            value = value[1:-1]
        items[-1][key] = unescape(value)
    return items


def test_export_all_writes_five_files(tmp_path):
    paths = exporters.export_all(RAW, REFINED, MINUTES_RESULT, tmp_path / "run")
    assert sorted((tmp_path / "run").iterdir()) == sorted(tmp_path / "run" / name for name in exporters.FILENAMES.values())
    assert set(paths) == {"raw", "refined", "json", "markdown", "srt"}


def test_raw_transcript_unchanged(outputs):
    assert outputs["raw"] == RAW["text"]  # leading spaces, whitespace segment and unicode all kept


def test_raw_file_bytes_are_utf8_of_raw_text(tmp_path):
    path = exporters.export_raw_transcript(RAW, tmp_path / "raw.txt")
    assert open(path, "rb").read() == RAW["text"].encode("utf-8")


def test_refined_transcript(outputs):
    assert outputs["refined"] == REFINED["refined_transcript"]


def test_json_is_valid_and_matches_record(outputs):
    data = json.loads(outputs["json"])
    source = MINUTES_RESULT["record"]
    assert data["summary"] == source["summary"]
    assert data["minutes"] == source["minutes"]
    assert data["key_decisions"] == source["key_decisions"]
    assert data["open_questions"] == source["open_questions"]
    assert data["warnings"] == MINUTES_RESULT["warnings"]
    assert len(data["action_items"]) == 3


def test_markdown_has_all_sections(outputs):
    md = outputs["markdown"]
    assert md.startswith("# Meeting Record\n")
    for title in ["Summary", "Minutes", "Key Decisions", "Action Items", "Open Questions", "Review Notes"]:
        md_section(md, title)


def test_empty_sections_still_present(tmp_path):
    empty = {"record": {"summary": "Short call.", "minutes": [], "key_decisions": [],
                        "action_items": [], "open_questions": []}}
    md = exporters.render_markdown(exporters.build_record(empty))
    for title in ["Minutes", "Key Decisions", "Action Items", "Open Questions"]:
        assert "_None recorded._" in md_section(md, title)
    assert "## Review Notes" not in md


def test_decisions_identical_in_markdown_and_json(outputs):
    json_decisions = json.loads(outputs["json"])["key_decisions"]
    md_decisions = md_items(md_section(outputs["markdown"], "Key Decisions"), ["decision", "evidence"])
    assert md_decisions == json_decisions


def test_action_items_identical_in_markdown_and_json(outputs):
    json_items = json.loads(outputs["json"])["action_items"]
    md_tasks = md_items(md_section(outputs["markdown"], "Action Items"), ["task", "owner", "deadline", "evidence"])
    assert md_tasks == json_items


def test_unspecified_preserved_and_blank_values_become_unspecified(outputs):
    items = json.loads(outputs["json"])["action_items"]
    assert (items[1]["owner"], items[1]["deadline"]) == (UNSPECIFIED, UNSPECIFIED)
    assert (items[2]["owner"], items[2]["deadline"]) == (UNSPECIFIED, UNSPECIFIED)
    assert items[0]["owner"] == "Priya" and items[0]["deadline"] == "by Friday"

    section = md_section(outputs["markdown"], "Action Items")
    assert section.count("**Owner:** unspecified\n") == 2
    assert section.count("**Deadline:** unspecified\n") == 2


def test_evidence_preserved_exactly(outputs):
    data = json.loads(outputs["json"])
    expected = [d["evidence"] for d in MINUTES_RESULT["record"]["key_decisions"]] + \
               [a["evidence"] for a in MINUTES_RESULT["record"]["action_items"]]
    assert [d["evidence"] for d in data["key_decisions"]] + [a["evidence"] for a in data["action_items"]] == expected

    md = outputs["markdown"]
    md_evidence = [unescape(m) for m in re.findall(r'\*\*Evidence:\*\* "(.*)"$', md, re.M)]
    assert md_evidence == expected


def test_exporters_do_not_modify_inputs(tmp_path):
    before = copy.deepcopy((RAW, REFINED, MINUTES_RESULT))
    exporters.export_all(RAW, REFINED, MINUTES_RESULT, tmp_path)
    assert (RAW, REFINED, MINUTES_RESULT) == before


SRT_TIME = r"\d{2}:\d{2}:\d{2},\d{3}"


def srt_cues(srt: str) -> list[tuple[int, str, str, str]]:
    cues = []
    for block in srt.strip("\n").split("\n\n"):
        number, times, text = block.split("\n", 2)
        start, end = times.split(" --> ")
        cues.append((int(number), start, end, text))
    return cues


def to_ms(stamp: str) -> int:
    h, m, rest = stamp.split(":")
    s, ms = rest.split(",")
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def test_srt_numbering_and_format(outputs):
    cues = srt_cues(outputs["srt"])
    assert [c[0] for c in cues] == list(range(1, len(cues) + 1))
    for _, start, end, _text in cues:
        assert re.fullmatch(SRT_TIME, start) and re.fullmatch(SRT_TIME, end)
    assert re.fullmatch(rf"(\d+\n{SRT_TIME} --> {SRT_TIME}\n[^\n]+\n)(\n\d+\n{SRT_TIME} --> {SRT_TIME}\n[^\n]+\n)*",
                        outputs["srt"])


def test_srt_times_and_text_match_segments(outputs):
    spoken = [s for s in RAW["segments"] if s["text"].strip()]
    cues = srt_cues(outputs["srt"])
    assert len(cues) == len(spoken) == 4
    for (_, start, end, text), seg in zip(cues, spoken):
        assert to_ms(start) == round(seg["start"] * 1000)
        assert to_ms(end) == round(seg["end"] * 1000)
        assert text == seg["text"]
        assert not re.match(r"\s*(speaker|spk)\b", text, re.I)


@pytest.mark.parametrize("seconds, stamp", [
    (0.0, "00:00:00,000"),
    (2.12, "00:00:02,120"),
    (59.9995, "00:01:00,000"),
    (3725.5, "01:02:05,500"),
    (3729.999, "01:02:09,999"),
])
def test_srt_time(seconds, stamp):
    assert exporters.srt_time(seconds) == stamp
