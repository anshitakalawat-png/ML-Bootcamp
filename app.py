"""Streamlit UI for the meeting assistant.

Run:  streamlit run app.py

This file only displays results. All processing happens in pipeline.py. Text
from the pipeline is never edited here: it is HTML-escaped and shown as is,
never rendered as Markdown (where characters like * or # would change how it
looks).

Layout: a compact header and upload card (with the audio player), then a
three-column workspace: main content | raw transcript | decisions. The audio
player and the transcript are two small HTML components that talk over a
BroadcastChannel: clicking a timestamp seeks the audio, and playback
highlights the current segment. Styling lives in ui/style.css; the
components in ui/*.html.
"""

from __future__ import annotations

import base64
import difflib
import hashlib
import html
import io
import json
import os
import re
import tempfile
from pathlib import Path

import av
import numpy as np
import streamlit as st

import exporters
import pipeline
import validate
from validate import SUPPORTED_EXTENSIONS

UI_DIR = Path(__file__).parent / "ui"

DOWNLOADS = [
    ("raw", "Raw (.txt)", "text/plain"),
    ("refined", "Refined (.txt)", "text/plain"),
    ("markdown", "Record (.md)", "text/markdown"),
    ("json", "Record (.json)", "application/json"),
    ("srt", "Subtitles (.srt)", "application/x-subrip"),
]

STAGE_LABELS = ["Audio validation", "Transcription", "Transcript refinement", "Meeting documentation"]
STATUS_TO_STAGE = {message: stage for stage, message in pipeline.STATUS.items()}

# Original uploads up to this size are played as is; larger ones are
# re-encoded to compact mono AAC so the page stays light.
MAX_INLINE_AUDIO_BYTES = 8 * 1024 * 1024
BROWSER_AUDIO_TYPES = {
    "wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4",
    "aac": "audio/aac", "ogg": "audio/ogg", "flac": "audio/flac",
}
WAVEFORM_BARS = 160
PLAYER_HEIGHT = 118
# Side panels fill the window height via CSS; this is their minimum.
SIDE_PANEL_MIN_HEIGHT = 420


# ---------- pure helpers (tested in tests/test_app.py) ----------

def highlight(text: str, query: str) -> tuple[str, int]:
    """Return (escaped HTML with case-insensitive matches in <mark>, match count)."""
    if not query:
        return html.escape(text), 0
    parts, last, count = [], 0, 0
    for match in re.finditer(re.escape(query), text, re.IGNORECASE):
        parts.append(html.escape(text[last:match.start()]))
        parts.append(f"<mark>{html.escape(match.group())}</mark>")
        last = match.end()
        count += 1
    parts.append(html.escape(text[last:]))
    return "".join(parts), count


def diff_html(raw: str, refined: str) -> str:
    """Word-level diff: raw-only words in <del>, refined-only words in <ins>."""
    a, b = raw.split(), refined.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(html.escape(" ".join(a[i1:i2])))
            continue
        if i2 > i1:
            out.append(f"<del>{html.escape(' '.join(a[i1:i2]))}</del>")
        if j2 > j1:
            out.append(f"<ins>{html.escape(' '.join(b[j1:j2]))}</ins>")
    return " ".join(out)


def html_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    """Escaped HTML table. `columns` is [(key, header)]."""
    head = "".join(f"<th>{html.escape(title)}</th>" for _, title in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key, _ in columns) + "</tr>"
        for row in rows
    )
    return f'<table class="mt-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def md_escape(text: str) -> str:
    """Escape Markdown so st.warning / st.error show the text literally."""
    return re.sub(r"([\\`*_\[\]<>#|~$])", r"\\\1", text)


def fmt_clock(seconds: float) -> str:
    """mm:ss, or h:mm:ss from one hour (display only; data keeps exact floats)."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def waveform_peaks(samples: np.ndarray, bars: int = WAVEFORM_BARS) -> list[float]:
    """Peak level per bar from the real samples, scaled to 0..1.

    Deterministic: the same audio always gives the same waveform.
    """
    if samples.size == 0:
        return [0.0] * bars
    edges = np.linspace(0, samples.size, bars + 1).astype(int)
    peaks = np.array([
        float(np.max(np.abs(samples[start:end]))) if end > start else 0.0
        for start, end in zip(edges[:-1], edges[1:])
    ])
    top = peaks.max()
    if top <= 0:
        return [0.0] * bars
    return [round(float(p), 4) for p in np.sqrt(peaks / top)]  # sqrt keeps quiet speech visible


def encode_playback_m4a(samples: np.ndarray, sample_rate: int, bit_rate: int = 32_000) -> bytes:
    """Encode mono float samples as AAC in an .m4a container, for playback only."""
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("aac", rate=sample_rate, layout="mono")
        stream.bit_rate = bit_rate
        frame_size = stream.codec_context.frame_size or 1024
        data = np.ascontiguousarray(samples, dtype=np.float32)
        for start in range(0, data.size, frame_size):
            chunk = data[start:start + frame_size].reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="mono")
            frame.sample_rate = sample_rate
            frame.pts = start
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


def playback_source(data: bytes, ext: str, samples: np.ndarray, sample_rate: int) -> tuple[bytes, str]:
    """Bytes and MIME type for the in-page player."""
    if len(data) <= MAX_INLINE_AUDIO_BYTES and ext in BROWSER_AUDIO_TYPES:
        return data, BROWSER_AUDIO_TYPES[ext]
    return encode_playback_m4a(samples, sample_rate), "audio/mp4"


def segment_rows_html(segments: list[dict], query: str) -> tuple[str, int]:
    """Transcript rows for ui/transcript.html, with exact start/end in data attributes."""
    rows, total = [], 0
    for seg in segments:
        text, count = highlight(seg["text"], query)
        total += count
        clock = fmt_clock(seg["start"])
        rows.append(
            f'<div class="seg" data-start="{seg["start"]!r}" data-end="{seg["end"]!r}">'
            f'<button class="ts" type="button" aria-label="Play from {clock}">{clock}</button>'
            f'<div class="tx">{text}</div></div>'
        )
    return "".join(rows), total


def fill_template(name: str, values: dict[str, str]) -> str:
    """Replace __KEY__ placeholders in ui/<name> in one pass (values are not re-scanned)."""
    template = (UI_DIR / name).read_text(encoding="utf-8")
    return re.sub(r"__([A-Z_]+)__", lambda m: values.get(m.group(1), m.group(0)), template)


def script_json(value) -> str:
    """JSON that is safe inside a <script> element."""
    return json.dumps(value).replace("</", "<\\/")


def progress_html(states: list[str]) -> str:
    """Stage list. Each state is "done", "active", "failed" or "pending"."""
    marks = {"done": "✓", "active": "●", "failed": "✕", "pending": "○"}
    items = "".join(
        f'<li class="{state}"><span class="dot">{marks[state]}</span>{html.escape(label)}</li>'
        for label, state in zip(STAGE_LABELS, states)
    )
    return f'<div class="mt-section-title">Processing</div><ul class="mt-steps">{items}</ul>'


def stat_cards_html(record: dict) -> str:
    """Counts come straight from the meeting record."""
    cards = [
        ("decisions", "Decisions", len(record["key_decisions"])),
        ("actions", "Action items", len(record["action_items"])),
        ("questions", "Open questions", len(record["open_questions"])),
    ]
    return '<div class="mt-stats">' + "".join(
        f'<div class="mt-stat {cls}"><div class="label">{label}</div><div class="value">{count}</div></div>'
        for cls, label, count in cards
    ) + "</div>"


def chip(label: str, value: str) -> str:
    cls = "mt-chip unspecified" if value == exporters.UNSPECIFIED else "mt-chip"
    return f'<span class="{cls}"><span>{html.escape(label)}</span>{html.escape(value)}</span>'


def evidence_html(evidence: str) -> str:
    return f'<div class="evidence"><b>Evidence</b>“{html.escape(evidence)}”</div>'


# ---------- page pieces ----------

def show_html(fragment: str) -> None:
    st.html(fragment)


def inject_css() -> None:
    st.html(f"<style>{(UI_DIR / 'style.css').read_text(encoding='utf-8')}</style>")


def process_button_state(processing: bool, has_file: bool) -> tuple[str, bool, str]:
    """(label, disabled, key) for the Process button.

    While the pipeline runs the button is disabled under a different key, so
    its style can show a spinner and no click can start a second run.
    """
    if processing:
        return "Processing meeting…", True, "process_running"
    return "Process meeting", not has_file, "process"


def start_processing() -> None:
    """Button callback: runs before the rerun, so the rerun already shows the disabled button."""
    st.session_state["processing"] = True


def render_header(slot, status: str) -> None:
    labels = {"ready": "Ready", "processing": "Processing", "done": "Done", "error": "Needs attention"}
    bars = "".join(f'<span style="height:{h}px"></span>' for h in (8, 16, 24, 14, 20, 10))
    slot.html(
        '<div class="mt-header">'
        f'<div class="mt-brand"><div class="mt-logo" aria-hidden="true"><div>{bars}</div></div>'
        '<div><div class="mt-title">Meeting Assistant</div>'
        '<div class="mt-subtitle">AI-powered meeting intelligence</div></div></div>'
        f'<div class="mt-status {status}"><i></i>{labels[status]}</div>'
        "</div>"
    )


@st.cache_data(max_entries=3, show_spinner=False)
def build_preview(data: bytes, name: str) -> dict | None:
    """Waveform and player audio for an upload, cached by content.

    Uses the same decoder as stage 0. If the file cannot be decoded, there is
    no preview; Process still runs the real validation and shows its error.
    """
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "preview." + ext)
        with open(path, "wb") as f:
            f.write(data)
        try:
            audio = validate.validate_audio(path)
        except validate.AudioValidationError:
            return None
    audio_bytes, mime = playback_source(data, ext, audio.samples, audio.sample_rate)
    return {
        "name": name,
        "duration": audio.duration_s,
        "peaks": waveform_peaks(audio.samples),
        "src": f"data:{mime};base64," + base64.b64encode(audio_bytes).decode("ascii"),
    }


def channel_name(file_id: str) -> str:
    return "mt-" + hashlib.sha1(file_id.encode("utf-8")).hexdigest()[:12]


def render_audio_player(preview: dict | None, channel: str | None) -> None:
    with st.container(key="audio_card"):
        if preview is None:
            show_html('<div class="mt-audio-empty">Your recording appears here, with its waveform, once you upload it.</div>')
            return
        st.iframe(
            fill_template("player.html", {
                "FILENAME": html.escape(preview["name"]),
                "DURATION_LABEL": fmt_clock(preview["duration"]),
                "DURATION": script_json(preview["duration"]),
                "PEAKS": script_json(preview["peaks"]),
                "CHANNEL": channel,
                "AUDIO_SRC": preview["src"],
            }),
            height=PLAYER_HEIGHT,
        )


def render_upload_section(processing: bool):
    """Compact hero: upload and Process on the left, the audio player on the right."""
    with st.container(key="hero"):
        left, right = st.columns([1, 1.15], gap="medium", vertical_alignment="center")
        with left:
            show_html(
                '<h1 class="mt-hero-title">Turn conversations into clear decisions.</h1>'
                "<p class=\"mt-hero-text\">Upload your meeting recording and we'll handle the rest.</p>"
            )
            # No `type=` filter: unsupported formats must reach stage 0 so the
            # user gets its message.
            uploaded = st.file_uploader(
                "Upload meeting recording", key="upload", label_visibility="collapsed",
                disabled=processing, help="Supported formats: " + ", ".join(SUPPORTED_EXTENSIONS),
            )
            st.caption(" · ".join(SUPPORTED_EXTENSIONS))
            label, disabled, key = process_button_state(processing, uploaded is not None)
            st.button(label, type="primary", disabled=disabled, key=key, on_click=start_processing)
        with right:
            preview = build_preview(uploaded.getvalue(), uploaded.name) if uploaded else None
            channel = channel_name(uploaded.file_id) if uploaded else None
            render_audio_player(preview, channel)
    return uploaded, preview, channel


def process(uploaded, progress_slot, header_slot) -> None:
    """Run the pipeline on the upload and keep the outcome in session state."""
    for key in ("result", "files", "error"):
        st.session_state.pop(key, None)

    states = ["pending"] * len(STAGE_LABELS)

    def show_progress() -> None:
        progress_slot.html(f'<div class="mt-card mt-progress">{progress_html(states)}</div>')

    def on_status(message: str) -> None:
        if message == pipeline.STATUS_DONE:
            states[:] = ["done"] * len(STAGE_LABELS)
        elif message in STATUS_TO_STAGE:
            stage = STATUS_TO_STAGE[message]
            for i in range(len(states)):
                states[i] = "done" if i < stage else ("active" if i == stage else "pending")
        show_progress()

    render_header(header_slot, "processing")
    show_progress()

    # Keep the original extension: stage 0 checks it.
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload" + suffix)
        with open(path, "wb") as f:
            f.write(uploaded.getvalue())
        try:
            # Custom vocabulary is no longer offered in the UI.
            result = pipeline.run_pipeline(path, vocabulary=None, on_status=on_status)
        except pipeline.PipelineError as err:
            states[err.stage] = "failed"
            st.session_state["error"] = {"stage": err.stage, "reason": err.reason, "message": err.message,
                                         "states": list(states)}
            return

        paths = result.export(os.path.join(tmp, "out"))
        files = {}
        for kind, file_path in paths.items():
            with open(file_path, "rb") as f:
                files[kind] = f.read()

    st.session_state["result"] = result
    st.session_state["files"] = files


def show_error(error: dict) -> None:
    if "states" in error:
        show_html(f'<div class="mt-card mt-progress">{progress_html(error["states"])}</div>')
    if error["stage"] == 0:
        st.error(f"**This file can't be processed.** {md_escape(error['reason'])}")
    else:
        st.error(md_escape(error["message"]))


# ---------- results ----------

def render_summary(record: dict) -> None:
    show_html(
        '<div class="mt-card summary"><h3>Meeting Summary</h3>'
        f'<div class="body">{html.escape(record["summary"])}</div></div>'
    )
    if record["minutes"]:
        topics = "".join(
            f'<div class="mt-topic">{html.escape(t["topic"])}</div>'
            "<ul>" + "".join(f"<li>{html.escape(p)}</li>" for p in t["points"]) + "</ul>"
            for t in record["minutes"]
        )
        show_html(f'<div class="mt-card"><h3>Minutes</h3>{topics}</div>')
    else:
        show_html('<div class="mt-empty">No minutes recorded.</div>')


def render_decisions(record: dict) -> None:
    decisions = record["key_decisions"]
    if not decisions:
        show_html('<div class="mt-empty">No agreed decisions found. Proposals and suggestions are not listed as decisions.</div>')
        return
    show_html("".join(
        '<div class="mt-item decision"><span class="mt-badge">Agreed</span>'
        f'<div class="title">{html.escape(d["decision"])}</div>{evidence_html(d["evidence"])}</div>'
        for d in decisions
    ))


def render_action_items(record: dict) -> None:
    items = record["action_items"]
    if not items:
        show_html('<div class="mt-empty">No action items recorded.</div>')
        return
    show_html("".join(
        '<div class="mt-item action">'
        f'<div class="title">{html.escape(a["task"])}</div>'
        f'<div class="mt-meta">{chip("Owner", a["owner"])}{chip("Deadline", a["deadline"])}</div>'
        f'{evidence_html(a["evidence"])}</div>'
        for a in items
    ))


def render_open_questions(record: dict) -> None:
    questions = record["open_questions"]
    if not questions:
        show_html('<div class="mt-empty">No open questions recorded.</div>')
        return
    show_html("".join(f'<div class="mt-question">{html.escape(q)}</div>' for q in questions))


def render_refined(result: pipeline.PipelineResult, query: str) -> None:
    view = st.segmented_control(
        "Refined view", ["Refined transcript", "Changes", "Corrections"],
        default="Refined transcript", key="refined_view", label_visibility="collapsed",
    ) or "Refined transcript"
    if view == "Refined transcript":
        fragment, count = highlight(result.refined_transcript, query)
        if query:
            st.caption(f"{count} match(es) in the refined transcript")
        show_html(
            '<div class="mt-card"><span class="mt-kind">Refined transcript</span>'
            '<div class="mt-muted">Domain terms corrected. The raw transcript in the side panel is unchanged.</div>'
            f'<div class="mt-text">{fragment}</div></div>'
        )
    elif view == "Changes":
        show_html(
            '<div class="mt-card"><span class="mt-kind raw">Raw</span> → <span class="mt-kind">Refined</span>'
            '<div class="mt-muted">Word-by-word comparison. Struck through: only in the raw transcript. '
            "Green: only in the refined transcript.</div>"
            f'<div class="mt-text">{diff_html(result.raw_text, result.refined_transcript)}</div></div>'
        )
    else:
        corrections = result.corrections
        if corrections:
            show_html('<div class="mt-card">' + html_table(corrections, [("heard", "Heard"), ("corrected", "Corrected")]) + "</div>")
        else:
            show_html('<div class="mt-empty">No corrections were made.</div>')


def render_export_section(files: dict) -> None:
    with st.container(key="exports"):
        columns = st.columns([0.9] + [1] * len(DOWNLOADS), vertical_alignment="center")
        columns[0].html('<div class="mt-section-title">Export meeting</div>')
        for column, (kind, label, mime) in zip(columns[1:], DOWNLOADS):
            column.download_button(
                label, data=files[kind], file_name=exporters.FILENAMES[kind], mime=mime,
                key=f"download_{kind}", use_container_width=True, on_click="ignore",
                help=exporters.FILENAMES[kind],
            )


def render_main_content(result: pipeline.PipelineResult, files: dict, query: str) -> None:
    warnings = result.warnings
    if warnings:
        with st.expander(f"⚠️ {len(warnings)} review note(s) from automatic checks", expanded=False):
            for warning in warnings:
                st.warning(md_escape(warning), icon="⚠️")

    record = result.record
    show_html(stat_cards_html(record))

    overview, refined, decisions, actions, questions = st.tabs(
        ["Overview", "Refined", "Decisions", "Actions", "Questions"]
    )
    with overview:
        render_summary(record)
    with refined:
        render_refined(result, query)
    with decisions:
        render_decisions(record)
    with actions:
        render_action_items(record)
    with questions:
        render_open_questions(record)

    render_export_section(files)


def render_transcript_sidebar(result: pipeline.PipelineResult, duration: float | None, channel: str | None) -> str:
    """Compact side panel with the raw transcript. Returns the search query."""
    with st.container(key="transcript_panel"):
        meta = fmt_clock(duration) if duration else ""
        show_html('<div class="mt-panel-head"><span class="title">Transcript</span>'
                  f'<span class="meta">Raw · unchanged{" · " + meta if meta else ""}</span></div>')
        query = st.text_input("Search transcript", key="search", placeholder="Search transcript",
                              label_visibility="collapsed")
        plain = st.toggle("Plain text", key="raw_plain")

        if plain:
            fragment, count = highlight(result.raw_text, query)
            if query:
                st.caption(f"{count} match(es) in the raw transcript")
            show_html(f'<div class="mt-text mt-plain">{fragment}</div>')
        else:
            rows, count = segment_rows_html(result.segments, query)
            if query:
                st.caption(f"{count} match(es) in the raw transcript")
            st.iframe(
                fill_template("transcript.html", {"SEGMENTS": rows, "CHANNEL": channel or "mt-none"}),
                height=SIDE_PANEL_MIN_HEIGHT,
            )
        return query


def decisions_sidebar_html(decisions: list[dict]) -> str:
    """Compact decision cards; evidence is folded into a <details> element."""
    if not decisions:
        return '<div class="mt-empty">No agreed decisions found.</div>'
    return '<div class="mt-side-list">' + "".join(
        '<div class="mt-side-decision">'
        '<span class="mt-badge"><span class="tick">✓</span>Agreed</span>'
        f'<div class="text">{html.escape(d["decision"])}</div>'
        f'<details><summary>Evidence</summary><div class="evidence">“{html.escape(d["evidence"])}”</div></details>'
        "</div>"
        for d in decisions
    ) + "</div>"


def render_decisions_sidebar(record: dict) -> None:
    decisions = record["key_decisions"]
    with st.container(key="decisions_panel"):
        show_html('<div class="mt-panel-head"><span class="title">Decisions</span>'
                  f'<span class="meta">{len(decisions)} agreed</span></div>')
        show_html(decisions_sidebar_html(decisions))


# ---------- page ----------

def main() -> None:
    st.set_page_config(page_title="Meeting Assistant", page_icon="🎙", layout="wide",
                       initial_sidebar_state="collapsed")
    inject_css()

    header_slot = st.empty()
    processing = st.session_state.get("processing", False)

    uploaded, preview, channel = render_upload_section(processing)

    file_key = uploaded.file_id if uploaded else None
    if st.session_state.get("file_key") != file_key:  # new or removed file: old results no longer apply
        st.session_state["file_key"] = file_key
        for key in ("result", "files", "error"):
            st.session_state.pop(key, None)

    progress_slot = st.empty()
    if processing:
        try:
            if uploaded is not None:
                process(uploaded, progress_slot, header_slot)
        finally:
            # Cleared even if the run fails or is interrupted, so the button
            # always comes back.
            st.session_state["processing"] = False
        st.rerun()  # redraw with the button enabled again and the outcome below

    status = "error" if "error" in st.session_state else "done" if "result" in st.session_state else "ready"
    render_header(header_slot, status)

    if "error" in st.session_state:
        show_error(st.session_state["error"])
        st.stop()  # nothing else renders after a failure

    result = st.session_state.get("result")
    if result is None:
        return

    main_col, transcript_col, decisions_col = st.columns([0.575, 0.2125, 0.2125], gap="small")
    with transcript_col:
        query = render_transcript_sidebar(result, preview["duration"] if preview else None, channel)
    with decisions_col:
        render_decisions_sidebar(result.record)
    with main_col:
        render_main_content(result, st.session_state["files"], query)


if __name__ == "__main__":  # `streamlit run app.py` runs the script as __main__
    main()
