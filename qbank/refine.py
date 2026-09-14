"""Gemini table rearrangement — runs DURING extraction, not after.

Every table the deterministic pipeline extracts (question or solution,
flagged or clean) is sent to Gemini once as TEXT — the current
pipe-markdown extraction itself, no page images — with the ask
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


# Models answer a "return ONLY the table" prompt with a sentence of
# preamble, a fence, or a closing remark surprisingly often. Throwing
# the whole rearrangement away for that cosmetically broke the pass
# (the run showed "invalid" and shipped raw tables). ONE unambiguous
# table block is salvaged; anything ambiguous is still refused —
# arrangement is the model's job, guessing is not.
def salvage_table(text: str) -> str | None:
    """The single pipe-table block inside a chatty answer, or None.
    Refuses when the answer contains more than one block (which table
    is the table is then genuinely unclear)."""
    if not text:
        return None
    lines = [l for l in text.splitlines()
             if not re.fullmatch(r"\s*`{3,}[A-Za-z]*\s*", l or "")]
    runs, cur = [], []
    for line in lines:
        if re.fullmatch(r"\s*\|.*\|\s*", line):
            cur.append(line.strip())
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    runs = [r for r in runs if len(r) >= 3]      # header + sep + 1 row
    if len(runs) != 1:
        return None
    cand = "\n".join(runs[0])
    return cand if valid_rearrangement(cand) else None


# A rejected answer used to be invisible: the stats line said
# "N invalid" and nobody could tell WHAT the model had sent. Show the
# first few, then stay quiet — a hundred of them would drown the log.
REJECT_SAMPLES = 3
_reject_samples_left = REJECT_SAMPLES


SALVAGE_SAMPLES = 3
_salvage_samples_left = SALVAGE_SAMPLES


def reset_samples() -> None:
    """Give a new run its own sample budget: the counters are process
    globals, so without this the second book in a long-lived dashboard
    process would log no samples at all."""
    global _reject_samples_left, _salvage_samples_left
    _reject_samples_left = REJECT_SAMPLES
    _salvage_samples_left = SALVAGE_SAMPLES


def _note_salvage(t: dict) -> None:
    global _salvage_samples_left
    if _salvage_samples_left <= 0:
        return
    _salvage_samples_left -= 1
    print(f"[gemini] {(t.get('table_id') or '?')}: answer had wrapper text "
          f"(prose/fences) — salvaged the single table block from it",
          flush=True)


def _note_reject(t: dict, answer: str) -> None:
    global _reject_samples_left
    if _reject_samples_left <= 0:
        return
    _reject_samples_left -= 1
    head = " / ".join((answer or "").strip().splitlines()[:2])[:200]
    print(f"[gemini] {(t.get('table_id') or '?')}: answer rejected — not an "
          f"even pipe-markdown table, deterministic one kept. "
          f"Answer starts: {head!r}", flush=True)


def refine_table(t: dict, book, refine_fn, only: str = "all",
                 memo: dict | None = None, ledger_key: str | None = None,
                 ledger_path: Path | None = None) -> str:
    """Rearrange one table record in place with Gemini's medical
    rearrangement and SAVE the model's output. Returns
    "replaced" | "same" | "empty" | "invalid" | "skip"."""
    if only != "all" and not flagged(t):
        return "skip"
    md = t.get("markdown") or ""
    if not md.strip():
        return "skip"
    key = md  # same extraction -> same model answer (cached either way)
    if memo is not None and key in memo:
        new = memo[key]
    else:
        # the pages the table spans are passed for context only
        # (cache key + multi-page span note) — no images are sent
        pgs = [int(p) for p in (t.get("source_pages") or [1])] or [1]
        new = refine_fn(book, pgs, md)
        if memo is not None:
            memo[key] = new
    if not new:
        return "empty"         # model returned nothing usable
    if new.strip() == md.strip():
        return "same"          # model returned the extraction as-is
    salvaged = False
    if not valid_rearrangement(new):
        block = salvage_table(new)
        if block is None:
            _note_reject(t, new)
            return "invalid"   # not a table: keep the deterministic one
        new, salvaged = block, True
        _note_salvage(t)
    val = t.setdefault("validation", {})
    val.setdefault("pre_gemini_markdown", md)
    qa = val.setdefault("table_qa", {})
    qa["refined_by_gemini"] = True
    if salvaged:
        qa["salvaged_from_wrapper"] = True   # provenance: it came chatty
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
