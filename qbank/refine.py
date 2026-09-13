"""Gemini table rearrangement — runs DURING extraction, not after.

Every table the deterministic pipeline extracts (question or solution,
flagged or clean) is sent to Gemini once: the page image as layout
reference plus the current pipe-markdown extraction, with the ask
"rearrange this table properly, per medical knowledge". The model's
markdown is what gets SAVED on the table record:

  * accepted when it parses as an even pipe-markdown table (>=2 rows,
    >=2 columns, every row the same width) — otherwise the
    deterministic extraction is kept as-is;
  * the original deterministic markdown is preserved under
    validation.pre_gemini_markdown and the table is marked
    table_qa.refined_by_gemini for provenance;
  * every accepted rearrangement is logged to data/refine_ledger.jsonl
    (the receipt's tables_refined count reads it).

No manual trigger: with GEMINI_API_KEY set the pass runs inside
run_chapter; without a key nothing is called and the deterministic
tables ship unchanged. REVIEW flags are untouched — flagged tables
still queue for the human, and the human queue still gates the zip.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import config

LEDGER = "refine_ledger.jsonl"


def md_table_shape(md: str) -> int | None:
    """Column count of an even pipe-markdown table (separator row
    excluded), else None. Needs >=2 rows and >=2 columns."""
    lines = [l for l in (md or "").strip().splitlines() if l.strip()]
    if len(lines) >= 2 and re.fullmatch(r"[\s|:-]+", lines[1] or " "):
        lines.pop(1)
    if len(lines) < 2:
        return None
    widths = set()
    for l in lines:
        cells = [c for c in l.strip().strip("|").split("|")]
        if len(cells) < 2 or any(not c.strip() for c in cells):
            return None
        widths.add(len(cells))
    return widths.pop() if len(widths) == 1 else None


def valid_rearrangement(new_md: str | None) -> bool:
    """A model answer is saved only when it IS a table — even, wide
    enough, no prose/fences riding along."""
    if not new_md or not new_md.strip():
        return False
    if "```" in new_md:
        return False
    return md_table_shape(new_md) is not None


def flagged(t: dict) -> bool:
    return (((t.get("validation") or {}).get("table_qa") or {})
            .get("status") == "REVIEW")


def refine_table(t: dict, book, refine_fn, only: str = "all",
                 memo: dict | None = None, ledger_key: str | None = None,
                 ledger_path: Path | None = None) -> str:
    """Rearrange one table record in place with Gemini's medical
    rearrangement and SAVE the model's output. Returns
    "replaced" | "skip" | "invalid"."""
    if only != "all" and not flagged(t):
        return "skip"
    md = t.get("markdown") or ""
    if not md.strip():
        return "skip"
    key = md  # same extraction -> same model answer (cached either way)
    if memo is not None and key in memo:
        new = memo[key]
    else:
        # ALL pages the table spans go to the model (cross-page
        # merged tables need both page images to be rearranged whole)
        pgs = [int(p) for p in (t.get("source_pages") or [1])] or [1]
        new = refine_fn(book, pgs, md)
        if memo is not None:
            memo[key] = new
    if not new or new.strip() == md.strip():
        return "skip"
    if not valid_rearrangement(new):
        return "invalid"       # not a table: keep the deterministic one
    val = t.setdefault("validation", {})
    val.setdefault("pre_gemini_markdown", md)
    qa = val.setdefault("table_qa", {})
    qa["refined_by_gemini"] = True
    t["markdown"] = new.strip()
    if ledger_key and ledger_path is not None:
        _append(ledger_path, {"key": ledger_key, "ts":
                              time.strftime("%Y-%m-%dT%H:%M:%S")})
    return "replaced"


def refined_count(output_root, subject: str) -> int:
    """Distinct tables Gemini rearranged during extraction (receipt)."""
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
