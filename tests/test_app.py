import html
import io
import re
from pathlib import Path

import av
import numpy as np
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
        {"start": 3725.123, "end": 3729.0, "text": " Late <segment> & more."},
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
        "open_questions": ["Who owns the demo?", "Budget?"],
    },
    warnings=["Part 2 of 2: the refined text was rejected because numbers were changed."],
)
FILES = {kind: b"x" for kind in ("raw", "refined", "markdown", "json", "srt")}


def strip_tags(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment))


def tone(seconds=2.0, sr=16_000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# ---------- text helpers ----------

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


def test_card_head_escapes_text():
    fragment = app.card_head_html("Start <here>", "A & B", "x < y")
    assert "Start &lt;here&gt;" in fragment and "A &amp; B" in fragment and "x &lt; y" in fragment
    assert '<p class="text">' not in app.card_head_html("Eyebrow", "Title")


def test_html_table_escapes_cells():
    table = app.html_table([{"a": "<b>x</b>", "b": "unspecified"}], [("a", "A"), ("b", "B")])
    assert "&lt;b&gt;x&lt;/b&gt;" in table and "<td>unspecified</td>" in table


def test_md_escape_shows_text_literally():
    assert app.md_escape('owner "*Arjun*" #1') == 'owner "\\*Arjun\\*" \\#1'


@pytest.mark.parametrize("seconds, label", [(0, "00:00"), (8.9, "00:08"), (99.5, "01:39"), (3725.1, "1:02:05")])
def test_fmt_clock(seconds, label):
    assert app.fmt_clock(seconds) == label


# ---------- waveform and playback ----------

def test_waveform_peaks_come_from_the_samples():
    samples = np.concatenate([np.zeros(8000, np.float32), tone(1.0), np.zeros(8000, np.float32)])
    peaks = app.waveform_peaks(samples, bars=4)
    assert len(peaks) == 4
    assert peaks[0] == 0.0 and peaks[3] == 0.0       # silent edges stay flat
    assert peaks[1] == pytest.approx(1.0) and peaks[2] == pytest.approx(1.0)
    assert app.waveform_peaks(samples, bars=4) == peaks  # deterministic


def test_waveform_peaks_of_silence_and_empty_audio():
    assert app.waveform_peaks(np.zeros(1000, np.float32), bars=5) == [0.0] * 5
    assert app.waveform_peaks(np.zeros(0, np.float32), bars=3) == [0.0] * 3


def test_encode_playback_m4a_keeps_duration():
    samples = tone(2.0)
    data = app.encode_playback_m4a(samples, 16_000)
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        assert stream.codec_context.name == "aac"
        decoded = sum(frame.samples for frame in container.decode(stream))
    assert decoded / 16_000 == pytest.approx(2.0, abs=0.15)


def test_playback_source_uses_original_when_small(monkeypatch):
    data, mime = app.playback_source(b"RIFFdata", "wav", tone(), 16_000)
    assert (data, mime) == (b"RIFFdata", "audio/wav")


def test_playback_source_transcodes_large_files(monkeypatch):
    monkeypatch.setattr(app, "MAX_INLINE_AUDIO_BYTES", 4)
    data, mime = app.playback_source(b"RIFFdata", "wav", tone(), 16_000)
    assert mime == "audio/mp4" and data != b"RIFFdata"


# ---------- transcript rows and templates ----------

def test_segment_rows_keep_exact_timestamps_and_escape_text():
    rows, count = app.segment_rows_html(RESULT.segments, "")
    assert count == 0
    assert rows.count('class="seg"') == 3
    assert 'data-start="3725.123" data-end="3729.0"' in rows
    assert 'aria-label="Play from 1:02:05">1:02:05</button>' in rows
    assert "Late &lt;segment&gt; &amp; more." in rows
    assert "We will *not* migrate." in rows
    assert not re.search(r"speaker", rows, re.I)


def test_segment_rows_highlight_search():
    rows, count = app.segment_rows_html(RESULT.segments, "will")
    assert count == 2 and rows.count("<mark>will</mark>") == 2


def test_fill_template_is_single_pass():
    page = app.fill_template("transcript.html", {"SEGMENTS": "text with __CHANNEL__ inside",
                                                 "CHANNEL": "mt-abc"})
    assert "text with __CHANNEL__ inside" in page   # values are not re-expanded
    assert 'new BroadcastChannel("mt-abc")' in page
    assert ".list { height: 100vh;" in page          # list fills the iframe, which CSS sizes


def test_script_json_cannot_close_the_script():
    assert "</script>" not in app.script_json(["</script><b>"])


def test_player_template_has_real_data():
    page = app.fill_template("player.html", {
        "FILENAME": "a &amp; b.wav", "DURATION_LABEL": "00:02", "DURATION": "2.0",
        "PEAKS": "[0.1, 1.0]", "CHANNEL": "mt-x", "AUDIO_SRC": "data:audio/wav;base64,AAAA",
    })
    assert 'var peaks = [0.1, 1.0];' in page
    assert 'src="data:audio/wav;base64,AAAA"' in page
    assert re.search(r"__[A-Z_]+__", page) is None


# ---------- progress and result widgets ----------

def test_progress_html_marks_each_stage():
    fragment = app.progress_html(["done", "done", "active", "pending"])
    assert '<li class="done"><span class="dot">✓</span>Audio validation</li>' in fragment
    assert '<li class="active"><span class="dot">●</span>Transcript refinement</li>' in fragment
    assert '<li class="pending"><span class="dot">○</span>Meeting documentation</li>' in fragment


def test_status_messages_map_to_stages():
    assert [app.STATUS_TO_STAGE[pipeline.STATUS[i]] for i in range(4)] == [0, 1, 2, 3]


def test_stat_cards_count_the_record():
    fragment = app.stat_cards_html(RESULT.record)
    values = re.findall(r'<div class="value">(\d+)</div>', fragment)
    assert values == ["1", "2", "2"]


def test_process_button_states():
    assert app.process_button_state(processing=False, has_file=False) == ("Process meeting", True, "process")
    assert app.process_button_state(processing=False, has_file=True) == ("Process meeting", False, "process")
    label, disabled, key = app.process_button_state(processing=True, has_file=True)
    assert disabled and key == "process_running" and label.startswith("Processing meeting")


def test_decisions_sidebar_uses_record_exactly():
    assert app.decisions_sidebar_html([]) == '<div class="mt-empty">No agreed decisions found.</div>'
    fragment = app.decisions_sidebar_html([{"decision": "Ship <v2>", "evidence": "we *agreed*"}])
    assert "Ship &lt;v2&gt;" in fragment and "we *agreed*" in fragment
    assert fragment.count('class="mt-side-decision"') == 1 and "✓" in fragment


def test_chip_keeps_unspecified_exactly():
    assert app.chip("Owner", "unspecified") == \
        '<span class="mt-chip unspecified"><span>Owner</span>unspecified</span>'
    assert 'class="mt-chip"' in app.chip("Owner", "Priya")


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


def iframes(at) -> list[str]:
    return [el.proto.srcdoc for el in at.get("iframe")]


def test_empty_page_shows_hero_and_disabled_button():
    at = run_page()
    assert at.button[0].label == "Process meeting" and at.button[0].disabled
    assert len(at.tabs) == 0
    blocks = html_blocks(at)
    assert "Clear decisions from every recording." in blocks
    assert "Upload a recording and turn the discussion into clear decisions" in blocks
    assert '<h1 class="mt-title">Meeting <span>Assistant</span></h1>' in blocks and "Ready to listen" in blocks
    assert "Bring in a recording" in blocks and "Waiting for audio" in blocks
    assert "Your recording appears here" in blocks
    assert iframes(at) == []


def test_custom_vocabulary_is_gone():
    for at in (run_page(), run_page(result=RESULT, files=FILES)):
        assert len(at.text_area) == 0
        assert all("vocabulary" not in e.label.lower() for e in at.expander)
        assert "vocabulary" not in html_blocks(at).split("</style>")[-1].lower()


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
    assert iframes(at) == []
    assert '<div class="mt-panel-head">' not in html_blocks(at)  # no transcript panel after a failure


def test_later_stage_error_names_the_stage():
    error = {"stage": 2, "reason": "No Gemini API key found.",
             "message": "Stage 2 (refinement) failed: No Gemini API key found."}
    at = run_page(error=error)
    assert "Stage 2 (refinement) failed" in at.error[0].value
    assert len(at.tabs) == 0
    assert "Needs attention" in html_blocks(at)


def test_results_render_all_parts():
    at = run_page(result=RESULT, files=FILES)
    assert [t.label for t in at.tabs] == ["Overview", "Refined", "Decisions", "Actions", "Questions"]
    assert len(at.warning) == 1 and "numbers were changed" in at.warning[0].value
    assert len(at.get("download_button")) == 5
    assert at.text_input[0].label == "Search transcript"
    assert at.toggle[0].label == "Plain text"

    blocks = html_blocks(at)
    assert re.findall(r'<div class="value">(\d+)</div>', blocks) == ["1", "2", "2"]
    assert "Meeting Summary" in blocks and "Config update; no *migration*." in blocks  # literal, not italics
    assert "Agreed" in blocks and "No migration." in blocks
    assert "Check &lt;costs&gt;" in blocks
    assert blocks.count('<span class="mt-chip unspecified"><span>Owner</span>unspecified</span>') == 1
    assert blocks.count('<span class="mt-chip unspecified"><span>Deadline</span>unspecified</span>') == 1
    assert blocks.count('class="mt-question"') == 2
    assert "Raw · unchanged" in blocks
    assert "Done" in blocks
    assert '<span class="meta">1 agreed</span>' in blocks       # decisions side panel
    assert blocks.count('class="mt-side-decision"') == 1
    assert "<details><summary>Evidence</summary>" in blocks

    transcript = [doc for doc in iframes(at) if 'class="seg"' in doc]
    assert len(transcript) == 1
    assert 'data-start="0.0" data-end="3.5"' in transcript[0]
    assert ">00:00</button>" in transcript[0] and ">1:02:05</button>" in transcript[0]


def test_refined_views():
    at = run_page(result=RESULT, files=FILES)
    assert "Refined transcript" in html_blocks(at) and "Priya will update the kubectl config" in html_blocks(at)
    at.button_group[0].set_value("Changes").run()
    assert not at.exception
    assert "<del>PREA</del>" in html_blocks(at)
    at.button_group[0].set_value("Corrections").run()
    assert "<td>cube control</td>" in html_blocks(at)


def test_search_highlights_in_transcripts():
    at = run_page(result=RESULT, files=FILES)
    at.text_input[0].input("WILL").run()
    assert not at.exception
    captions = [c.value for c in at.caption]
    assert "2 match(es) in the raw transcript" in captions
    assert "2 match(es) in the refined transcript" in captions
    transcript = [doc for doc in iframes(at) if 'class="seg"' in doc][0]
    assert transcript.count("<mark>will</mark>") == 2


def test_plain_text_toggle():
    at = run_page(result=RESULT, files=FILES)
    at.toggle[0].set_value(True).run()
    assert not at.exception
    assert not [doc for doc in iframes(at) if 'class="seg"' in doc]
    assert html.escape(RESULT.raw_text) in html_blocks(at)


def _app_with_fake_upload_and_pipeline():
    """Runs the real app.main() with a fake upload and a fake pipeline.

    AppTest runs only this function's source, so everything is set up inside.
    """
    import io as _io
    import wave as _wave

    import numpy as _np
    import streamlit as st

    import app
    import pipeline

    buf = _io.BytesIO()
    t = _np.arange(32_000) / 16_000
    with _wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes((0.3 * _np.sin(2 * _np.pi * 220 * t) * 32767).astype("<i2").tobytes())

    class FakeUpload:
        name = "meeting.wav"
        file_id = "fake-file-1"

        def getvalue(self):
            return buf.getvalue()

    st.file_uploader = lambda *args, **kwargs: FakeUpload()

    def fake_run_pipeline(path, vocabulary="not passed", on_status=None):
        log = st.session_state.setdefault("pipeline_calls", [])
        log.append({
            "vocabulary": vocabulary,
            "processing_flag": st.session_state.get("processing"),
            # The disabled "Processing meeting…" button was drawn before the pipeline started.
            "running_button_drawn": "process_running" in st.session_state,
        })
        for message in pipeline.STATUS.values():
            on_status(message)
        on_status(pipeline.STATUS_DONE)
        return pipeline.PipelineResult(
            raw_text=" Hello there.", segments=[{"start": 0.0, "end": 2.0, "text": " Hello there."}],
            refined_transcript="Hello there.", corrections=[],
            record={"summary": "Greeting.", "minutes": [], "key_decisions": [], "action_items": [],
                    "open_questions": []},
            warnings=[],
        )

    pipeline.run_pipeline = fake_run_pipeline
    app.main()


def test_process_runs_pipeline_once_with_disabled_button_and_no_vocabulary(monkeypatch):
    import streamlit as st

    # The script replaces these in this process; registering them here makes
    # pytest put the originals back afterwards.
    monkeypatch.setattr(pipeline, "run_pipeline", pipeline.run_pipeline)
    monkeypatch.setattr(st, "file_uploader", st.file_uploader)

    at = AppTest.from_function(_app_with_fake_upload_and_pipeline, default_timeout=60)
    at.run()
    assert not at.exception
    assert at.button(key="process").label == "Process meeting" and not at.button(key="process").disabled

    at.button(key="process").click().run()
    assert not at.exception
    calls = at.session_state["pipeline_calls"]
    assert len(calls) == 1
    assert calls[0] == {"vocabulary": None, "processing_flag": True, "running_button_drawn": True}

    # After the run the button is back to normal and results show.
    assert at.session_state["processing"] is False
    assert at.button(key="process").label == "Process meeting" and not at.button(key="process").disabled
    assert [t.label for t in at.tabs] == ["Overview", "Refined", "Decisions", "Actions", "Questions"]
    assert "No agreed decisions found." in html_blocks(at)

    # An unrelated rerun (e.g. typing a search) does not start the pipeline again.
    at.text_input(key="search").input("hello").run()
    assert len(at.session_state["pipeline_calls"]) == 1
