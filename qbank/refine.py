"""Gemini table refinement: rearranges badly-arranged extracted tables
into clean pipe-markdown — "only same content of table".

The model sees the page image (layout reference) plus the current
extraction and may rearrange / split glued fragments, but the result
is accepted ONLY if it passes the same-content envelope (identical
alphanumeric character stream: nothing added, nothing deleted).  A
refined table stays in the human REVIEW queue — the user's approval
is still the gate before the final zip."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from . import config

LEDGER = "refine_ledger.jsonl"


def norm_concat(md: str | None) -> str:
    return re.sub(r"[^0-9a-z]+", "", (md or "").lower())


def same_content(old: str, new: str | None) -> bool:
    """Nothing added, nothing deleted — splits of glued fragments keep
    the alphanumeric stream identical; any invented word breaks it."""
    return bool(new) and norm_concat(old) == norm_concat(new)


def flagged(t: dict) -> bool:
    return (((t.get("validation") or {}).get("table_qa") or {})
            .get("status") == "REVIEW")


def refine_table(t: dict, book, refine_fn, only: str,
                 memo: dict | None = None) -> str:
    """Refine one table record in place.
    Returns "replaced" | "rejected" | "skip"."""
    if only != "all" and not flagged(t):
        return "skip"
    md = t.get("markdown") or ""
    if not md.strip():
        return "skip"
    key = md  # same extraction -> same model answer (cached either way)
    if memo is not None and key in memo:
        new = memo[key]
    else:
        pg = (t.get("source_pages") or [1])[0]
        new = refine_fn(book, pg, md)
        if memo is not None:
            memo[key] = new
    if not new or new.strip() == md.strip():
        return "skip"
    if not same_content(md, new):
        return "rejected"      # invented/removed content: keep original
    t["markdown"] = new
    qa = (t.setdefault("validation", {})
          .setdefault("table_qa", {}))
    qa["refined_by_gemini"] = True
    return "replaced"


def refined_count(output_root, subject: str) -> int:
    """Distinct tables whose refinement was accepted (receipt)."""
    p = Path(output_root) / "data" / LEDGER
    if not p.exists():
        return 0
    keys = set()
    for l in p.read_text().splitlines():
        if not l.strip():
            continue
        try:
            row = json.loads(l)
        except ValueError:
            continue
        if (row.get("key") or "").startswith(subject + "|"):
            keys.add(row["key"])
    return len(keys)


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def refine_subject(output_root, subject: str, refine_fn=None,
                   only: str | None = None, log=print) -> dict:
    """Retro-pass over an already-extracted book: rewrite flagged (or
    all, only="all") table markdowns in questions/solutions jsonl.
    Decisions fingerprinted to the old markdown become stale, so the
    refined tables return to the review queue for the user."""
    out = Path(output_root)
    only = only or os.environ.get("QBANK_REFINE", "flagged")
    split = out / "split" / subject
    if not split.is_dir():
        return {"refined": 0, "rejected": 0,
                "why": f"no extracted data for {subject}"}
    if refine_fn is None:
        from . import llm as llm_mod
        if not llm_mod.enabled():
            return {"refined": 0, "rejected": 0, "why": "gemini disabled"}
        refine_fn = llm_mod.refiner(out / "llm_cache")
    from .textlayer import Book
    entry = config.load_books().get(subject)
    book = Book(str(config.resolve_book_path(entry)))

    refined = rejected = 0
    memo: dict = {}
    ledger = out / "data" / LEDGER
    for nf in ("questions.jsonl", "solutions.jsonl"):
        for qf in sorted(split.glob(f"*/{nf}")):
            rows = [json.loads(l) for l in qf.read_text().splitlines()
                    if l.strip()]
            changed = False
            for row in rows:
                for t in row.get("tables") or []:
                    st = refine_table(t, book, refine_fn, only, memo)
                    if st == "replaced":
                        if nf == "questions.jsonl":
                            refined += 1
                        changed = True
                        _append(ledger, {
                            "key": f"{subject}|{row.get('q_id')}|"
                                   f"{t.get('table_id')}",
                            "file": nf, "ts": time.strftime(
                                "%Y-%m-%dT%H:%M:%S"),
                            "envelope": "same-content"})
                    elif st == "rejected" and nf == "questions.jsonl":
                        rejected += 1
            if changed:
                qf.write_text("".join(
                    json.dumps(r, sort_keys=True) + "\n" for r in rows))
    book.close()
    log(f"[{subject}] refine: {refined} table(s) rearranged "
        f"(same-content envelope), {rejected} model answer(s) rejected")
    return {"refined": refined, "rejected": rejected}
