"""
Command line:

    python3 -m qbank run --book BIO [--chapters 1,3-5] [--force]
    python3 -m qbank run --pdf path.pdf --subject BIO [--page-offset auto|N]
    python3 -m qbank export [--dest path.zip]
    python3 -m qbank status
    python3 -m qbank audit [--book BIO]
    python3 -m qbank table-audit --book BIO
    python3 -m qbank keys

Extraction is deterministic and needs no API key; the optional Gemini
table pass uses GEMINI_API_KEYS / GEMINI_API_KEY_1..20 / GEMINI_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import config, state as state_mod
from .export import build_final_zip, gate_final_zip
from .run import run_book


def _parse_chapters(spec: str) -> set[int]:
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def cmd_run(args) -> int:
    if args.book:
        books = config.load_books()
        if args.book not in books:
            print(f"book {args.book!r} not in {config.BOOKS_FILE} "
                  f"(have: {sorted(books)})", file=sys.stderr)
            return 2
        entry = books[args.book]
        subject = args.subject or entry.get("subject", args.book)
        pdf = config.resolve_book_path(entry)
        offset = (args.page_offset if args.page_offset is not None
                  else entry.get("page_offset", "auto"))
    else:
        if not args.pdf or not args.subject:
            print("either --book NAME or (--pdf PATH --subject CODE)",
                  file=sys.stderr)
            return 2
        subject = args.subject
        pdf = args.pdf
        offset = args.page_offset if args.page_offset is not None else "auto"
    if offset != "auto":
        offset = int(offset)

    filt = _parse_chapters(args.chapters) if args.chapters else None
    res = run_book(pdf, subject, page_offset=offset,
                   chapters_filter=filt, force=args.force)
    print(f"\n[{subject}] done: {res['chapters_run']} chapter(s), "
          f"{res['total_questions']} questions, "
          f"census failures: {res['census_failures'] or 'none'}")
    return 1 if res["census_failures"] else 0


def cmd_export(args) -> int:
    res = build_final_zip(config.OUTPUT_ROOT, args.book, dest=args.dest)
    if not res["ok"]:
        print(f"REFUSED: {res['why']}", file=sys.stderr)
        return 3
    print(f"export -> {res['path']}")
    print(json.dumps(res["receipt"], indent=2))
    return 0


def cmd_status(args) -> int:
    state = state_mod.load_state()
    print(json.dumps(state.get("pdf_progress", {}), indent=2))
    split = config.OUTPUT_ROOT / "split"
    for d in sorted(split.glob("*")) if split.is_dir() else []:
        if d.is_dir():
            gate = gate_final_zip(config.OUTPUT_ROOT, d.name)
            print(f"export gate {d.name}:",
                  "OPEN (zip can build)" if not gate["locked"]
                  else f"LOCKED — {gate['why']}")
    return 0


def cmd_audit(args) -> int:
    from .audit import audit_book, write_report
    res = audit_book(config.OUTPUT_ROOT, subject=args.book)
    path = write_report(config.OUTPUT_ROOT, res)
    ev = "yes" if res["page_text"] else "MISSING (numeric_drift skipped)"
    print(f"scanned {res['rows_scanned']} question rows; "
          f"page-text evidence: {ev}")
    for kind, n in sorted(res["by_kind"].items()):
        print(f"  {kind}: {n}")
    if not res["by_kind"]:
        print("  no flags")
    print(f"report -> {path}")
    return 0


def cmd_keys(args) -> int:
    from . import keypool
    pool = keypool.get_pool()
    if pool is None:
        print("no keys configured (GEMINI_API_KEYS / GEMINI_API_KEY_1..20 /"
              " GEMINI_API_KEY)")
        return 0
    print(json.dumps(pool.summary(), indent=2))
    return 0


def cmd_table_audit(args) -> int:
    """Print the final table-refinement audit report for one book."""
    from . import refine_final as final_mod
    res = final_mod.load_audit(config.OUTPUT_ROOT, args.book)
    if not res:
        print(f"no table-refinement audit for {args.book} — the final "
              "stage runs when Gemini is enabled (data/"
              f"{final_mod.AUDIT})")
        return 1
    for k in ("total_tables", "tables_unchanged", "tables_refined",
              "tables_rejected", "tables_review", "tables_no_answer",
              "spacing_repairs", "medical_spelling_repairs",
              "number_repairs", "structural_repairs",
              "structural_rejections", "presentation_only_refinements",
              "cross_page_tables_checked", "fidelity_violations",
              "rejected_hallucinations", "rejected_deletions",
              "rejected_number_changes", "render_page_waste_issues",
              "gemini_api_calls"):
        print(f"  {k}: {res.get(k)}")
    reg = res.get("before_after_regression") or {}
    print(f"  before_after_regression: "
          f"{'OK' if reg.get('ok') else 'FAILED'} "
          f"({reg.get('chapters', 0)} chapter(s), {reg.get('counts', {})})")
    corr = res.get("corrections") or []
    if corr:
        print(f"  accepted content corrections ({len(corr)}):")
        for c in corr[: args.limit]:
            print(f"    {c['table_id']} {c['cell']}: "
                  f"{c.get('before')!r} -> {c.get('after')!r} "
                  f"[{c.get('kind')}] evidence={c.get('evidence')} "
                  f"confidence={c.get('confidence')} "
                  f"reason={c.get('reason')!r}")
        if len(corr) > args.limit:
            print(f"    ... and {len(corr) - args.limit} more")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qbank", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="extract one book")
    p_run.add_argument("--book", help="key in books.json (e.g. BIO)")
    p_run.add_argument("--pdf", help="explicit pdf path")
    p_run.add_argument("--subject", help="subject code (e.g. BIO)")
    p_run.add_argument("--page-offset", default=None,
                       help="'auto' (default) or an integer")
    p_run.add_argument("--chapters", help="subset, e.g. '1,3-5'")
    p_run.add_argument("--force", action="store_true",
                       help="re-extract chapters already marked done")
    p_run.set_defaults(fn=cmd_run)

    p_exp = sub.add_parser("export", help="build the independent "
                                          "final_export_<CODE>.zip "
                                          "for ONE book")
    p_exp.add_argument("--dest", default=None)
    p_exp.add_argument("--book", required=True,
                       help="subject code (e.g. ENT): one book, one "
                            "gate, one zip")
    p_exp.set_defaults(fn=cmd_export)

    p_st = sub.add_parser("status", help="resume state + export gate")
    p_st.set_defaults(fn=cmd_status)

    p_au = sub.add_parser("audit", help="read-only content audit of the "
                                        "extracted split")
    p_au.add_argument("--book", help="subject code; default = all")
    p_au.set_defaults(fn=cmd_audit)

    p_keys = sub.add_parser("keys", help="Gemini key-pool status "
                                          "(fingerprints only)")
    p_keys.set_defaults(fn=cmd_keys)

    p_ta = sub.add_parser("table-audit",
                          help="final table-refinement audit report "
                               "(counts + accepted content corrections)")
    p_ta.add_argument("--book", required=True,
                      help="subject code (e.g. ENT)")
    p_ta.add_argument("--limit", type=int, default=40,
                      help="max corrections to print (default 40)")
    p_ta.set_defaults(fn=cmd_table_audit)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
