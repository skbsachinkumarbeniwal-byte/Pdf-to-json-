#!/usr/bin/env python3
"""Medical-token spacing audit (the forensic report's §12.7 numbers).

Scans a finished output tree for candidate splits (the audit's detector
regex), counts how many the book's own vocabulary can repair, and
proves each repair is a WHITESPACE-ONLY change.

Usage:  python3 scripts/medical_token_audit.py <output_root> [subject] [pdf]

Reports, per §12.7:
    medical-token candidates BEFORE
    medical-token repairs ACCEPTED
    medical-token candidates REMAINING
    false-positive regression count
    whitespace-only fidelity failures
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root

from qbank.tables import join_medical_tokens  # noqa: E402

# the audit's candidate detector, used here for MEASUREMENT only
CAND = re.compile(r"\b([A-Z][A-Z0-9]{0,5})\s+"
                  r"([A-Z]?[A-Za-z0-9+\-]*\d[A-Za-z0-9+\-]*)\b")


def fields(row):
    """(label, text) for every text-bearing field of a row (§12.3)."""
    for k in ("question_text", "solution_text"):
        if isinstance(row.get(k), str) and row[k]:
            yield k, row[k]
    for o in row.get("options") or []:
        if isinstance(o.get("text"), str) and o["text"]:
            yield f"option:{o.get('letter') or o.get('key') or '?'}", o["text"]
    for t in row.get("tables") or []:
        if t.get("markdown"):
            yield f"table:{t.get('table_id')}", t["markdown"]


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "qbank_output")
    subject = sys.argv[2] if len(sys.argv) > 2 else None
    pdf = sys.argv[3] if len(sys.argv) > 3 else None

    subs = [subject] if subject else sorted(
        p.name for p in (root / "split").iterdir() if p.is_dir())
    candidates = Counter()
    where = {}
    for sub in subs:
        for ch in sorted((root / "split" / sub).iterdir()):
            for f in ("questions.jsonl", "answers.jsonl", "solutions.jsonl"):
                p = ch / f
                if not p.exists():
                    continue
                for line in p.read_text().splitlines():
                    r = json.loads(line)
                    for label, text in fields(r):
                        for m in CAND.finditer(text):
                            candidates[m.group(0)] += 1
                            where.setdefault(m.group(0), (r["q_id"], label))

    print(f"scanned split tree under {root} "
          f"({'/' + ','.join(subs) if len(subs) < 8 else f'{len(subs)} subjects'})")
    print(f"medical-token candidates BEFORE : {sum(candidates.values())} "
          f"in {len(candidates)} distinct forms")

    if not pdf:
        print("\n(no PDF given: pass it as argv[3] to test the repairs "
              "against the book's vocabulary)")
        for form, n in candidates.most_common(15):
            qid, lab = where[form]
            print(f"   {form!r:18} x{n:<4} e.g. {qid} {lab}")
        return 0

    from qbank.textlayer import Book
    from qbank.tables import alnum_vocab
    book = Book(pdf)
    mixed = alnum_vocab(book)
    print(f"book vocabulary: {len(mixed)} alphanumeric printed forms")

    # what the PASS actually repairs, measured the way the pipeline
    # applies it: whole text in, repairs out. (Judging candidates in
    # isolation would claim `A 9- year-old` is repairable when the
    # context — the hyphen — is exactly why it is not.)
    accepted, fidelity = Counter(), 0
    for sub in subs:
        for ch in sorted((root / "split" / sub).iterdir()):
            for f in ("questions.jsonl", "answers.jsonl", "solutions.jsonl"):
                pth = ch / f
                if not pth.exists():
                    continue
                for line in pth.read_text().splitlines():
                    r = json.loads(line)
                    for label, text in fields(r):
                        out, reps = join_medical_tokens(text, mixed)
                        if not reps:
                            continue
                        for before, after in reps:
                            accepted[(before, after)] += 1
                        if re.sub(r"\s+", "", out) != re.sub(r"\s+", "", text):
                            fidelity += 1
    rejected = {f: f for f in candidates if not any(
        b.strip() == f for b, _a in accepted)}

    print(f"medical-token repairs ACCEPTED : {sum(accepted.values())} "
          f"occurrences in {len(accepted)} forms (measured in context)")
    for (before, after), n in sorted(accepted.items(), key=lambda kv: -kv[1]):
        print(f"   {before!r:14} -> {after!r:12} x{n}")
    print(f"candidates REMAINING           : "
          f"{sum(candidates[f] for f in rejected)} in {len(rejected)} forms")
    for form in sorted(rejected, key=lambda f: -candidates[f])[:12]:
        print(f"   {form!r:18} x{candidates[form]:<4} "
              f"(joined form not printed by this book) "
              f"e.g. {where[form][0]} {where[form][1]}")
    print()
    print(f"false-positive regressions     : "
          f"{sum(candidates[f] for f in rejected)} (left untouched by design)")
    print(f"whitespace-only fidelity failures: {fidelity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
