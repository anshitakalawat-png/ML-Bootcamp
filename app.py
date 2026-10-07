"""Streamlit UI for the meeting assistant.

Run:  streamlit run app.py

This file only displays results. All processing happens in pipeline.py. Text
from the pipeline is never edited here: it is HTML-escaped and shown as is,
never rendered as Markdown (where characters like * or # would change how it
looks).
"""

from __future__ import annotations

import difflib
import html
import os
import re
import tempfile

import streamlit as st

import exporters
import pipeline
from validate import SUPPORTED_EXTENSIONS

DOWNLOADS = [
    ("raw", "Raw transcript (.txt)", "text/plain"),
    ("refined", "Refined transcript (.txt)", "text/plain"),
    ("markdown", "Meeting record (.md)", "text/markdown"),
    ("json", "Meeting record (.json)", "application/json"),
    ("srt", "Subtitles (.srt)", "application/x-subrip"),
]

CSS = """
<style>
.mt-text { white-space: pre-wrap; line-height: 1.6; }
.mt-ts { opacity: 0.6; font-family: monospace; margin-right: 0.5em; white-space: nowrap; }
.mt-text mark { background: rgba(255, 200, 0, 0.45); color: inherit; padding: 0 1px; border-radius: 2px; }
.mt-text del { background: rgba(255, 80, 80, 0.25); text-decoration: line-through; }
.mt-text ins { background: rgba(60, 200, 90, 0.3); text-decoration: none; }
table.mt-table { border-collapse: collapse; width: 100%; }
table.mt-table th, table.mt-table td { border: 1px solid rgba(128, 128, 128, 0.35); padding: 6px 8px;
  text-align: left; vertical-align: top; white-space: pre-wrap; }
table.mt-table th { background: rgba(128, 128, 128, 0.12); }
</style>
"""


# ---------- pure helpers (tested in tests/test_app.py) ----------

def parse_vocabulary(text: str) -> list[str]:
    """Split the sidebar text on commas (or new lines) into terms."""
    return [term.strip() for term in re.split(r"[,\n]", text) if term.strip()]


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


def fmt_time(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:05.2f}"


def show_html(fragment: str) -> None:
    st.html(CSS + fragment)


def show_text(text: str) -> None:
    show_html(f'<div class="mt-text">{html.escape(text)}</div>')


# ---------- processing ----------

def process(uploaded, vocabulary: list[str]) -> None:
    """Run the pipeline on the upload and keep the outcome in session state."""
    for key in ("result", "files", "error"):
        st.session_state.pop(key, None)

    # Keep the original extension: stage 0 checks it.
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload" + suffix)
        with open(path, "wb") as f:
            f.write(uploaded.getvalue())

        with st.status("Processing...", expanded=True) as box:
            try:
                result = pipeline.run_pipeline(path, vocabulary=vocabulary, on_status=box.write)
            except pipeline.PipelineError as err:
                box.update(label=f"Stopped: stage {err.stage} ({err.stage_name}) failed", state="error")
                st.session_state["error"] = {"stage": err.stage, "reason": err.reason, "message": err.message}
                return
            box.update(label="Done", state="complete", expanded=False)

        paths = result.export(os.path.join(tmp, "out"))
        files = {}
        for kind, file_path in paths.items():
            with open(file_path, "rb") as f:
                files[kind] = f.read()

    st.session_state["result"] = result
    st.session_state["files"] = files


def show_error(error: dict) -> None:
    if error["stage"] == 0:
        st.error(f"**This file can't be processed.** {md_escape(error['reason'])}")
    else:
        st.error(md_escape(error["message"]))


# ---------- results ----------

def show_results(result: pipeline.PipelineResult, files: dict) -> None:
    for warning in result.warnings:
        st.warning(md_escape(warning), icon="⚠️")

    st.subheader("Downloads")
    columns = st.columns(len(DOWNLOADS))
    for column, (kind, label, mime) in zip(columns, DOWNLOADS):
        column.download_button(label, data=files[kind], file_name=exporters.FILENAMES[kind],
                               mime=mime, key=f"download_{kind}", use_container_width=True)

    query = st.text_input("Search transcripts", key="search", placeholder="Find a word or phrase")

    raw_tab, refined_tab, diff_tab, corrections_tab, minutes_tab, actions_tab = st.tabs([
        "Raw Transcript", "Refined Transcript", "Diff", "Corrections", "Minutes and Decisions", "Action Items",
    ])

    with raw_tab:
        plain = st.toggle("Plain text", key="raw_plain")
        if plain:
            fragment, count = highlight(result.raw_text, query)
            body = f'<div class="mt-text">{fragment}</div>'
        else:
            lines, count = [], 0
            for seg in result.segments:
                fragment, n = highlight(seg["text"], query)
                count += n
                stamp = f"[{fmt_time(seg['start'])} → {fmt_time(seg['end'])}]"
                lines.append(f'<div class="mt-text"><span class="mt-ts">{stamp}</span>{fragment}</div>')
            body = "".join(lines)
        if query:
            st.caption(f"{count} match(es) in the raw transcript")
        show_html(body)

    with refined_tab:
        fragment, count = highlight(result.refined_transcript, query)
        if query:
            st.caption(f"{count} match(es) in the refined transcript")
        show_html(f'<div class="mt-text">{fragment}</div>')

    with diff_tab:
        st.caption("Word-by-word comparison. Red, struck through: only in the raw transcript. "
                   "Green: only in the refined transcript.")
        show_html(f'<div class="mt-text">{diff_html(result.raw_text, result.refined_transcript)}</div>')

    with corrections_tab:
        if result.corrections:
            show_html(html_table(result.corrections, [("heard", "Heard"), ("corrected", "Corrected")]))
        else:
            st.info("No corrections were made.")

    record = result.record
    with minutes_tab:
        st.subheader("Summary")
        show_text(record["summary"])

        st.subheader("Minutes")
        if not record["minutes"]:
            st.info("No minutes recorded.")
        for topic in record["minutes"]:
            points = "".join(f"<li>{html.escape(p)}</li>" for p in topic["points"])
            show_html(f'<div class="mt-text"><strong>{html.escape(topic["topic"])}</strong></div><ul>{points}</ul>')

        st.subheader("Key Decisions")
        if record["key_decisions"]:
            show_html(html_table(record["key_decisions"], [("decision", "Decision"), ("evidence", "Evidence")]))
        else:
            st.info("No decisions recorded.")

        st.subheader("Open Questions")
        if record["open_questions"]:
            show_html("<ul>" + "".join(f"<li>{html.escape(q)}</li>" for q in record["open_questions"]) + "</ul>")
        else:
            st.info("No open questions recorded.")

    with actions_tab:
        if record["action_items"]:
            show_html(html_table(record["action_items"], [
                ("task", "Task"), ("owner", "Owner"), ("deadline", "Deadline"), ("evidence", "Evidence"),
            ]))
        else:
            st.info("No action items recorded.")


# ---------- page ----------

def main() -> None:
    st.set_page_config(page_title="Meeting Assistant", page_icon="📝", layout="wide")
    st.title("Meeting Assistant")
    st.caption("Upload an English meeting recording to get a transcript and a meeting record.")

    with st.sidebar:
        vocab_text = st.text_area(
            "Custom vocabulary",
            key="vocabulary",
            placeholder="Priya, Kubernetes, PyTorch",
            help="Comma-separated names and terms used in the meeting. "
                 "Refinement uses these spellings when a word sounds like one of them.",
        )

    # No `type=` filter: unsupported formats must reach stage 0 so the user
    # gets its message.
    uploaded = st.file_uploader("Meeting recording", key="upload",
                                help="Supported formats: " + ", ".join(SUPPORTED_EXTENSIONS))

    file_key = (uploaded.file_id if uploaded else None)
    if st.session_state.get("file_key") != file_key:  # new or removed file: old results no longer apply
        st.session_state["file_key"] = file_key
        for key in ("result", "files", "error"):
            st.session_state.pop(key, None)

    if st.button("Process", type="primary", disabled=uploaded is None):
        process(uploaded, parse_vocabulary(vocab_text))

    if "error" in st.session_state:
        show_error(st.session_state["error"])
        st.stop()  # nothing else renders after a failure

    if "result" in st.session_state:
        show_results(st.session_state["result"], st.session_state["files"])


if __name__ == "__main__":  # `streamlit run app.py` runs the script as __main__
    main()
