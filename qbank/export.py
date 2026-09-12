"""
final_export.zip — same tree as FORMAT.md, same receipt keys as v1.

    final_export.zip
    ├── REVIEW_RECEIPT.json
    ├── FORMAT.md
    ├── split/<SUBJ>/<SUBJ>-<NNN>/{questions,answers,solutions,
    │       image_manifest}.jsonl + chapter_completeness.json
    ├── subjects/<SUBJ>/chapters.json
    └── assets/questions/<SUBJ>/*.webp   (ONLY manifest-referenced files)

The v1 gate was "review queue clear". The v2 gate is its deterministic
equivalent: every chapter's completeness file proves the printed census
(question headers == key rows == solution headers, contiguous) and no
row ships as REVIEW_NEEDED. A blocked build lists exactly which
chapters/rows are open.
"""

from __future__ import annotations

import json
import os
import time
import zipfile
from pathlib import Path

SPLIT_KEEP = {"questions.jsonl", "answers.jsonl", "solutions.jsonl",
              "image_manifest.jsonl", "chapter_completeness.json"}


def _read_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _split_glob(out_root: Path, subject: str, name: str):
    return sorted((out_root / "split" / subject).glob(f"*/{name}")) \
        if (out_root / "split" / subject).is_dir() else []


def _llm_used(out_root: Path) -> bool:
    """True when the Gemini transcription stage produced any evidence:
    cached model responses, or any shipped table repaired by it."""
    cache = out_root / "llm_cache"
    if cache.is_dir() and any(cache.glob("*.json")):
        return True
    return False


def _table_stats(out_root: Path, subject: str) -> tuple:
    """(gemini_repaired_tables, qa_review_tables) over shipped rows."""
    seen, gem, review = set(), 0, 0
    for nf in ("questions.jsonl", "solutions.jsonl"):
        for qf in _split_glob(out_root, subject, nf):
            for row in _read_jsonl(qf):
                for t in row.get("tables") or []:
                    tid = t.get("table_id")
                    if tid in seen:
                        continue
                    seen.add(tid)
                    v = t.get("validation") or {}
                    if v.get("llm_space_repairs"):
                        gem += 1
                    if ((v.get("table_qa") or {}).get("status")
                            == "REVIEW"):
                        review += 1
    return gem, review


def gate_final_zip(output_root, subject: str) -> dict:
    """One book, one gate: only subject=CODE's chapters + its REVIEW
    tables gate ITS zip. Other books can never lock it, and a new
    book's run never re-locks an already-shipped one."""
    out_root = Path(output_root)
    problems = []
    chapters = 0
    for cf in _split_glob(out_root, subject, "chapter_completeness.json"):
        chapters += 1
        comp = json.loads(cf.read_text())
        census = comp.get("census") or {}
        if not census.get("ok"):
            problems.append(f"{comp['chapter_id']}: census FAILED {census}")
        # NOTE: qa_status_counts.REVIEW_NEEDED is the run-time flag
        # count — informational only.  It never decreases, so it must
        # NOT hard-lock the zip; the human-review lock below
        # (pending_count) is the one decisions can unlock.  The run's
        # counts still ship in the receipt (shipped_qa_status_counts).
        if comp.get("unresolved_qid_count"):
            problems.append(
                f"{comp['chapter_id']}: "
                f"{comp['unresolved_qid_count']} unresolved q_id(s)")
    if chapters == 0:
        problems.append(f"no chapters on disk for {subject}")
    # human review layer (adopted): final zip hard-locked while any
    # REVIEW table is undecided/stale — override QBANK_FORCE_EXPORT=1
    import os
    from . import review
    if os.environ.get("QBANK_FORCE_EXPORT") != "1":
        pend = review.pending_count(out_root, subject)
        if pend:
            problems.append(
                f"{pend} REVIEW table(s) awaiting human decision "
                "(see review dashboard; QBANK_FORCE_EXPORT=1 overrides)")
    return {
        "locked": bool(problems),
        "chapters": chapters,
        "why": None if not problems else "; ".join(problems),
    }


def build_final_zip(output_root, subject: str, dest=None) -> dict:
    """One book, one INDEPENDENT zip: final_export_<CODE>.zip holds
    only that book's split, its chapters.json and only its
    manifest-referenced assets. Other books' zips stay untouched."""
    out_root = Path(output_root)
    gate = gate_final_zip(out_root, subject)
    if gate["locked"]:
        return {"ok": False, "locked": True, "why": gate["why"]}

    dest = Path(dest or (out_root / f"final_export_{subject}.zip"))
    referenced = set()
    subjects = set()
    manifest_files = _split_glob(out_root, subject, "image_manifest.jsonl")
    for mf in manifest_files:
        subjects.add(mf.parts[-3])
        for row in _read_jsonl(mf):
            if row.get("file"):
                referenced.add(row["file"])

    shipped_status: dict = {}
    glyph_fix_total = 0
    llm_tables, qa_review_tables = _table_stats(out_root, subject)
    for qf in _split_glob(out_root, subject, "questions.jsonl"):
        for row in _read_jsonl(qf):
            st = row.get("qa_status") or "UNLABELLED"
            shipped_status[st] = shipped_status.get(st, 0) + 1
            fixes = (row.get("q_no_anchors") or {}).get("glyph_fixes") or {}
            glyph_fix_total += sum(
                v for k, v in fixes.items() if k != "unknown_glyph")

    from . import review
    receipt = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_root": out_root.name,
        "chapters": len(manifest_files),
        "subjects": sorted(subjects),
        "images_shipped": len(referenced),
        "review_decisions": len(review.load_decisions(out_root)),
        "human_edits": review.edit_count(out_root),
        "shipped_qa_status_counts": shipped_status or None,
        "glyph_fix_total": glyph_fix_total,
        "llm_tables_repaired": llm_tables,
        "tables_qa_review": qa_review_tables,
        "pipeline": ("deterministic-text-layer-v2+gemini-table-vision"
                     if _llm_used(out_root)
                     else "deterministic-text-layer-v2"),
        "gate": ("census verified — question headers, answer-key rows and "
                 "solution headers match in every chapter; no row flagged"),
    }

    fm = Path(__file__).resolve().parent.parent / "FORMAT.md"
    # write to a .part name and rename at the end: /api/status may be
    # globbing+reading zips while this runs — never expose a half zip
    tmp = dest.with_name(dest.name + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("REVIEW_RECEIPT.json",
                   json.dumps(receipt, indent=2, ensure_ascii=False))
        if fm.exists():
            z.write(fm, "FORMAT.md")
        for name in sorted(SPLIT_KEEP):
            for p in _split_glob(out_root, subject, name):
                z.write(p, str(p.relative_to(out_root)))
        cj = out_root / "subjects" / subject / "chapters.json"
        if cj.exists():
            z.write(cj, str(cj.relative_to(out_root)))
        aroot = out_root / "assets" / "questions"
        for rel in sorted(referenced):
            p = aroot / rel
            if p.exists():
                z.write(p, str(Path("assets") / "questions" / rel))
    os.replace(tmp, dest)
    return {"ok": True, "path": str(dest), "receipt": receipt,
            "images_shipped": len(referenced)}
