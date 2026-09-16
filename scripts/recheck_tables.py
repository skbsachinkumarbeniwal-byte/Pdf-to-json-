"""Re-judge stored tables' REVIEW flags with the CURRENT determinant.

Why this exists
---------------
A table's ``validation.table_qa`` verdict is written ONCE, at extraction
(or final-refinement) time, and the export gate reads that stored
verdict. When the detector itself improves — a false-positive branch is
tightened, a new rule lands — every table extracted by the older
detector keeps a stale verdict: a false REVIEW keeps a finished book
LOCKED forever, and a table that would now be flagged keeps shipping as
``ok``. Neither is acceptable, and re-running whole chapters to refresh
a flag burns Gemini quota and shifts table fingerprints.

This script re-applies the SAME deterministic detector
(``qbank.tables.qa_suspects``) that the pipeline uses in-run, through
``qbank.refine.refresh_table_qa``, to the markdown that will actually
ship, and rewrites the stored rows only when the verdict CHANGED. It
makes no model calls.

Usage
-----
    python scripts/recheck_tables.py <outdir> <SUBJECT> [--pdf PATH]
                                     [--apply]

Without ``--apply`` it is a dry run: it prints every verdict transition
and touches nothing. With ``--apply`` each rewritten file is first
copied to ``<file>.pre-recheck`` (written once; a later re-run never
overwrites the original backup) and the JSONL is replaced atomically.

Exit code is 0 whether or not anything changed; 2 on usage errors.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qbank.config import load_books          # noqa: E402
from qbank.refine import refresh_table_qa     # noqa: E402
from qbank.tables import build_vocab          # noqa: E402
from qbank.textlayer import Book              # noqa: E402

FILES = ("questions.jsonl", "solutions.jsonl")


def _pdf_for(subject: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    entry = load_books().get(subject) or {}
    pdf = entry.get("path") or entry.get("pdf")
    if not pdf:
        raise SystemExit(f"no pdf recorded for {subject} in books.json — "
                         f"pass --pdf")
    p = Path(pdf)
    if not p.is_absolute():
        p = REPO / p
    if not p.exists():
        raise SystemExit(f"pdf not found: {p}")
    return p.resolve()


def _status(t: dict) -> str:
    return (((t.get("validation") or {}).get("table_qa") or {})
            .get("status") or "none")


def _suspects(t: dict) -> list:
    return (((t.get("validation") or {}).get("table_qa") or {})
            .get("suspect_fragments") or [])


def _rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _write_atomic(path: Path, rows: list) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                   for r in rows)
    tmp.write_text(body)
    os.replace(tmp, path)


def _backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + ".pre-recheck")
    if not bak.exists():
        shutil.copy2(path, bak)
    return bak


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir")
    ap.add_argument("subject")
    ap.add_argument("--pdf", help="book pdf (default: books.json)")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the stored rows (default: dry run)")
    args = ap.parse_args(argv)

    out_root = Path(args.outdir).resolve()
    split = out_root / "split" / args.subject
    if not split.is_dir():
        raise SystemExit(f"no split dir: {split}")
    pdf = _pdf_for(args.subject, args.pdf)
    vocab = build_vocab(Book(str(pdf)))
    print(f"vocabulary: {len(vocab[0])} words from {pdf.name}")

    transitions, touched_files, tables, rejudged, unjudgeable = [], 0, 0, 0, 0
    paths = [qf for name in FILES for qf in sorted(split.glob(f"*/{name}"))]
    for path in paths:
        rows = _rows(path)
        changed = False
        for row in rows:
            for t in row.get("tables") or []:
                if not isinstance(t, dict) or not t.get("markdown"):
                    continue
                tables += 1
                before, before_s = _status(t), list(_suspects(t))
                got = refresh_table_qa(t, t["markdown"], vocab)
                if got is None:
                    unjudgeable += 1
                    continue
                rejudged += 1
                after, after_s = _status(t), list(_suspects(t))
                if after != before:
                    changed = True
                    qa = t.setdefault("validation", {}).setdefault(
                        "table_qa", {})
                    qa["rechecked_by"] = "scripts/recheck_tables.py"
                    qa["recheck_detector"] = ("qa_suspects:wrap-evidence "
                                              "needs the joined form "
                                              "printed >=2x")
                    transitions.append((row["q_id"], t.get("table_id"),
                                        before, before_s, after, after_s))
        if changed:
            touched_files += 1
            if args.apply:
                bak = _backup(path)
                _write_atomic(path, rows)
                print(f"rewrote {path.relative_to(out_root)} "
                      f"(backup {bak.name})")
            else:
                print(f"WOULD rewrite {path.relative_to(out_root)}")

    print(f"\ntables seen {tables}, re-judged {rejudged}, "
          f"unjudgeable {unjudgeable}, verdicts changed "
          f"{len(transitions)} in {touched_files} file(s)")
    for q_id, tid, before, b_ss, after, a_ss in transitions:
        print(f"  {q_id} {tid}: {before} {b_ss} -> {after} {a_ss}")
    if not args.apply and touched_files:
        print("\ndry run — nothing written; re-run with --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
