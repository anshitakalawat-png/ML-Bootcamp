import html
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app
import pipeline

APP = str(Path(__file__).resolve().parent.parent / "app.py")

RESULT = pipeline.PipelineResult(
    raw_text=" PREA will update the cube control config by Friday. We will *not* migrate.",
    segments=[
        {"start": 0.0, "end": 3.5, "text": " PREA will update the cube control config by Friday."},
        {"start": 3.5, "end": 5.0, "text": " We will *not* migrate."},
    ],
    refined_transcript="Priya will update the kubectl config by Friday. We will *not* migrate.",
    corrections=[{"heard": "PREA", "corrected": "Priya"}, {"heard": "cube control", "corrected": "kubectl"}],
    record={
        "summary": "Config update; no *migration*.",
        "minutes": [{"topic": "Config #1", "points": ["Priya will update the kubectl config."]}],
        "key_decisions": [{"decision": "No migration.", "evidence": "We will *not* migrate."}],
        "action_items": [{"task": "Update config", "owner": "Priya", "deadline": "by Friday",
                          "evidence": "Priya will update the kubectl config by Friday"},
                         {"task": "Check <costs>", "owner": "unspecified", "deadline": "unspecified",
                          "evidence": "We will *not* migrate."}],
        "open_questions": ["Who owns the demo?"],
    },
    warnings=["Part 2 of 2: the refined text was rejected because numbers were changed."],
)
FILES = {kind: b"x" for kind in ("raw", "refined", "markdown", "json", "srt")}


def strip_tags(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment))


# ---------- helpers ----------

def test_parse_vocabulary():
    assert app.parse_vocabulary(" Priya, Kubernetes ,,\nPyTorch\n") == ["Priya", "Kubernetes", "PyTorch"]
    assert app.parse_vocabulary("") == []


def test_highlight_marks_matches_case_insensitively():
    fragment, count = app.highlight("Config the CONFIG <now> & config", "config")
    assert count == 3
    assert fragment.count("<mark>") == 3
    assert "<mark>CONFIG</mark>" in fragment
    assert "&lt;now&gt; &amp;" in fragment  # pipeline text is escaped, never interpreted
    assert strip_tags(fragment) == "Config the CONFIG <now> & config"


def test_highlight_without_query_is_just_escaped_text():
    assert app.highlight("a < b", "") == ("a &lt; b", 0)


def test_highlight_treats_query_literally():
    fragment, count = app.highlight("costs $5 (approx.)", "(approx.)")
    assert count == 1 and "<mark>(approx.)</mark>" in fragment


def test_diff_marks_changed_words():
    fragment = app.diff_html(" PREA will update the cube control config.", "Priya will update the kubectl config.")
    assert "<del>PREA</del>" in fragment and "<ins>Priya</ins>" in fragment
    assert "<del>cube control</del>" in fragment and "<ins>kubectl</ins>" in fragment
    assert "will update the" in fragment and "<del>will" not in fragment


def test_diff_of_identical_text_has_no_marks():
    fragment = app.diff_html("same <text> here", "same <text> here")
    assert "<del>" not in fragment and "<ins>" not in fragment
    assert "&lt;text&gt;" in fragment


def test_html_table_escapes_cells():
    table = app.html_table([{"a": "<b>x</b>", "b": "unspecified"}], [("a", "A"), ("b", "B")])
    assert "&lt;b&gt;x&lt;/b&gt;" in table and "<td>unspecified</td>" in table


def test_md_escape_shows_text_literally():
    assert app.md_escape('owner "*Arjun*" #1') == 'owner "\\*Arjun\\*" \\#1'


# ---------- the page ----------

def run_page(**state):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["file_key"] = None  # matches "no file uploaded", so state is kept
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    return at


def html_blocks(at) -> str:
    return "\n".join(str(el.proto.body) for el in at.get("html"))


def test_empty_page_shows_uploader_and_disabled_button():
    at = run_page()
    assert at.button[0].label == "Process" and at.button[0].disabled
    assert len(at.tabs) == 0
    assert at.sidebar.text_area[0].label == "Custom vocabulary"


@pytest.mark.parametrize("reason", [
    "The uploaded file is empty.",
    "No audio signal detected. The file appears to be silent.",
    "Unsupported file format '.txt'. Supported formats: wav, mp3, m4a, flac, ogg, aac.",
    "The audio file is unreadable or corrupted. Please upload a valid audio recording.",
])
def test_validation_error_shows_only_the_error(reason):
    error = {"stage": 0, "reason": reason, "message": f"Stage 0 (validation) failed: {reason}"}
    at = run_page(error=error, result=RESULT, files=FILES)  # stale result must not show
    assert len(at.error) == 1
    assert app.md_escape(reason) in at.error[0].value
    assert len(at.tabs) == 0 and len(at.warning) == 0
    assert len(at.get("download_button")) == 0 and len(at.text_input) == 0


def test_later_stage_error_names_the_stage():
    error = {"stage": 2, "reason": "No Anthropic API key found.",
             "message": "Stage 2 (refinement) failed: No Anthropic API key found."}
    at = run_page(error=error)
    assert "Stage 2 \\(refinement\\) failed" not in at.error[0].value  # parentheses are not escaped
    assert "Stage 2 (refinement) failed" in at.error[0].value
    assert len(at.tabs) == 0


def test_results_render_all_parts():
    at = run_page(result=RESULT, files=FILES)
    assert [t.label for t in at.tabs] == ["Raw Transcript", "Refined Transcript", "Diff", "Corrections",
                                          "Minutes and Decisions", "Action Items"]
    assert len(at.warning) == 1 and "numbers were changed" in at.warning[0].value
    assert len(at.get("download_button")) == 5
    assert at.text_input[0].label == "Search transcripts"
    assert at.toggle[0].label == "Plain text"

    blocks = html_blocks(at)
    assert "[00:00.00 → 00:03.50]" in blocks
    assert "We will *not* migrate." in blocks       # shown literally, not as italics
    assert "Check &lt;costs&gt;" in blocks
    assert blocks.count("<td>unspecified</td>") == 2
    assert "<del>PREA</del>" in blocks


def test_search_highlights_in_transcripts():
    at = run_page(result=RESULT, files=FILES)
    at.text_input[0].input("WILL").run()
    assert not at.exception
    captions = [c.value for c in at.caption]
    assert "2 match(es) in the raw transcript" in captions
    assert "2 match(es) in the refined transcript" in captions
    assert "<mark>will</mark>" in html_blocks(at)


def test_plain_text_toggle():
    at = run_page(result=RESULT, files=FILES)
    at.toggle[0].set_value(True).run()
    assert not at.exception
    blocks = html_blocks(at)
    assert "[00:00.00" not in blocks
    assert html.escape(RESULT.raw_text) in blocks
