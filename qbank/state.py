"""Run state — chapter-granular resume, same file location as v1."""

from __future__ import annotations

import json
from pathlib import Path

from . import config


def load_state() -> dict:
    if config.STATE_FILE.exists():
        try:
            return json.loads(config.STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"pdf_progress": {}}


def save_state(state: dict) -> None:
    config.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.STATE_FILE.write_text(json.dumps(state, indent=2))


def progress(state: dict, subject: str) -> dict:
    """Per-subject progress. Backfills missing keys: a state.json
    written by an older build has no `chapters_llm_failed`, and the
    first retry-carrying run after an upgrade must not crash on it."""
    prog = state.setdefault("pdf_progress", {}).setdefault(subject, {})
    prog.setdefault("chapters_done", [])
    prog.setdefault("chapters_llm_failed", {})
    return prog


def note_llm_failures(state: dict, subject: str, chapter_id: str,
                      failures: dict) -> None:
    """Remember that a chapter finished with Gemini tables that never
    came back (no-answer / rejected / invalid). A chapter like that is
    only HALF done: the deterministic text is on disk, the model pass
    is not. The next `run` retries exactly these chapters instead of
    resuming past them (see run.run_book), so a book interrupted by a
    quota/window/network error finishes itself on the next run — the
    old behaviour marked it "done" and never looked again."""
    prog = progress(state, subject)
    failed = {k: v for k, v in (failures or {}).items() if v}
    if failed:
        prog["chapters_llm_failed"][chapter_id] = failed
    else:
        prog["chapters_llm_failed"].pop(chapter_id, None)
