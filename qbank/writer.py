"""
Output writers — byte-compatible with the v1 split layer's CONTRACT:

  split/<SUBJ>/<SUBJ>-<NNN>/
      questions.jsonl  answers.jsonl  solutions.jsonl
      unresolved_qids.jsonl  orphans.jsonl  image_manifest.jsonl
      chapter_completeness.json          (written LAST = on-disk signal)
  subjects/<SUBJ>/chapters.json
  data/chapters.json
  data/image_ownership.jsonl             (global ledger, one row/claim)

Row fields are the v1 field set. What changed is provenance: every
field is TEXT_LAYER, q_no_anchors carry the printed anchors the
deterministic parser actually used, and glyph repairs are audited in
q_no_anchors.glyph_fixes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import config
from .parse import grade_and_status

SPLIT_KEEP = {"questions.jsonl", "answers.jsonl", "solutions.jsonl",
              "image_manifest.jsonl", "chapter_completeness.json"}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in rows), encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(path)


def _qid(subject: str, chapter_no: int, qn: int) -> str:
    return f"{subject}-{chapter_no:03d}-{int(qn):03d}"


def _anchors(rec: dict, qn: int) -> dict:
    fp = {}
    if rec.get("question_text"):
        fp["question_text"] = config.PROV_TEXT_LAYER
    if rec.get("options"):
        fp["options"] = config.PROV_TEXT_LAYER
    if rec.get("correct_option"):
        fp["correct_option"] = config.PROV_TEXT_LAYER
    if rec.get("solution_text"):
        fp["solution_text"] = config.PROV_TEXT_LAYER
    out = {
        "model_q_no": int(qn),
        "model_q_no_provs": [config.PROV_TEXT_LAYER],
        "model_q_no_disagree": False,
        "field_provenance": fp,
        "provenance_notes": [config.PROV_TEXT_LAYER],
    }
    if rec.get("q_header_page"):
        out["printed_question_header"] = {"page": rec["q_header_page"]}
    if rec.get("key_page"):
        out["printed_key_row"] = {"page": rec["key_page"]}
    if rec.get("s_header_page"):
        out["printed_solution_header"] = {"page": rec["s_header_page"]}
    if rec.get("source_pages"):
        out["section_position"] = {"kind": "page_set",
                                   "pages": sorted(rec["source_pages"])}
    fixes = {k: v for k, v in dict(rec.get("glyph_fixes") or {}).items() if v}
    if fixes:
        out["glyph_fixes"] = fixes
    return out


def _img_refs(files, pages_by_file):
    return [{"file": f, "source_pages": [pages_by_file.get(f)]}
            for f in files]


def build_rows(records: dict, subject: str, chapter_id: str, chapter_no: int,
               image_files_by_q: dict, pages_by_file: dict):
    """(question_rows, answer_rows, solution_rows, unresolved_rows)"""
    q_rows, a_rows, s_rows, un_rows = [], [], [], []
    for qn in sorted(records):
        rec = records[qn]
        qid = _qid(subject, chapter_no, qn)
        grade, qa_status, qa_reasons = grade_and_status(rec)
        anchors = _anchors(rec, qn)
        imgs = image_files_by_q.get(qn, {})

        if "no_question_header" in rec["flags"]:
            # cannot build a question row without the printed stem —
            # it goes to the unresolved ledger, never invented.
            if rec.get("correct_option") and rec.get("solution_text"):
                reason = "missing_question_for_solution"
            elif rec.get("solution_text"):
                reason = "solution_q_no_not_in_printed_header"
            elif rec.get("correct_option"):
                reason = "answer_q_no_not_in_printed_key"
            else:
                reason = "no_anchor_at_all"
            un_rows.append({
                "q_id": qid, "chapter_id": chapter_id, "subject": subject,
                "chapter_no": chapter_no, "q_no": qn,
                "kind": "unresolved_qid", "reason": reason,
                "q_no_anchors": anchors, "available_passes": {},
                "source_pages": rec.get("source_pages") or [],
            })
            continue

        missing = []
        if not rec["question_text"].strip():
            missing.append("question_text")
        if len(rec["options"]) < 4 or not all(
                str(rec["options"].get(l, "")).strip() for l in "ABCD"):
            missing.append("options")
        qrow = {
            "q_id": qid, "chapter_id": chapter_id, "subject": subject,
            "chapter_no": chapter_no, "q_no": qn,
            "q_id_grade": grade, "q_no_anchors": anchors,
            "question_text": rec["question_text"],
            "options": [
                {"id": l,
                 "text": rec["options"].get(l, ""),
                 "images": _img_refs(imgs.get("option", {}).get(l, []),
                                     pages_by_file)}
                for l in "ABCD"],
            "question_images": _img_refs(imgs.get("question", []),
                                         pages_by_file),
            "tables": rec.get("tables") or [],
            "source_pages": rec.get("source_pages") or [],
        }
        if rec.get("question_text"):
            qrow["question_text_prov"] = config.PROV_TEXT_LAYER
        if rec.get("options"):
            qrow["options_prov"] = config.PROV_TEXT_LAYER
        qrow["extraction_status"] = "INCOMPLETE" if missing else "COMPLETE"
        if missing:
            qrow["missing_fields"] = missing
        qrow["qa_status"] = qa_status
        if qa_reasons:
            qrow["qa_reasons"] = qa_reasons
        q_rows.append(qrow)

        a_missing = [] if rec["correct_option"] else ["correct_option"]
        arow = {
            "q_id": qid, "chapter_id": chapter_id, "subject": subject,
            "chapter_no": chapter_no, "q_no": qn,
            "correct_option": rec["correct_option"] or None,
            "correct_option_prov": (config.PROV_TEXT_LAYER
                                    if rec["correct_option"] else None),
            "q_id_grade": grade, "q_no_anchors": anchors,
            "source_pages": rec.get("source_pages") or [],
            "extraction_status": "INCOMPLETE" if a_missing else "COMPLETE",
            "qa_status": qa_status,
        }
        if a_missing:
            arow["missing_fields"] = a_missing
        if qa_reasons:
            arow["qa_reasons"] = qa_reasons
        a_rows.append(arow)

        s_missing = [] if rec["solution_text"].strip() else ["solution_text"]
        srow = {
            "q_id": qid, "chapter_id": chapter_id, "subject": subject,
            "chapter_no": chapter_no, "q_no": qn,
            "solution_text": rec["solution_text"],
            "tables": rec.get("tables") or [],
            "solution_images": _img_refs(imgs.get("solution", []),
                                         pages_by_file),
            "solution_prov": (config.PROV_TEXT_LAYER
                              if rec["solution_text"] else None),
            "q_id_grade": grade, "q_no_anchors": anchors,
            "source_pages": rec.get("source_pages") or [],
            "extraction_status": "INCOMPLETE" if s_missing else "COMPLETE",
            "qa_status": qa_status,
        }
        if s_missing:
            srow["missing_fields"] = s_missing
        if qa_reasons:
            srow["qa_reasons"] = qa_reasons
        s_rows.append(srow)
    return q_rows, a_rows, s_rows, un_rows


def _with_final_refine(table_stats: dict | None,
                       final_refine_stats: dict | None) -> dict:
    """Per-chapter table stats + the final-refinement pass's own audit
    (accepted/rejected/review counts and the before/after regression
    verdict), nested under tables.final_refine."""
    out = dict(table_stats or {})
    if final_refine_stats is not None:
        out["final_refine"] = final_refine_stats
    return out


def write_chapter_split(*, output_root: Path, subject: str, chapter_id: str,
                        chapter_no: int, q_rows, a_rows, s_rows, un_rows,
                        orphan_rows, manifest_rows, scan_summary: dict,
                        glyph_audit: dict, image_report_summary: dict,
                        table_stats: dict | None = None,
                        final_refine_stats: dict | None = None) -> dict:
    ch_dir = Path(output_root) / "split" / subject / chapter_id
    ch_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(ch_dir / "questions.jsonl", q_rows)
    _atomic_write_jsonl(ch_dir / "answers.jsonl", a_rows)
    _atomic_write_jsonl(ch_dir / "solutions.jsonl", s_rows)
    _atomic_write_jsonl(ch_dir / "unresolved_qids.jsonl", un_rows)
    _atomic_write_jsonl(ch_dir / "orphans.jsonl", orphan_rows)
    _atomic_write_jsonl(ch_dir / "image_manifest.jsonl", manifest_rows)

    qa_counts: dict = {}
    for r in q_rows:
        qa_counts[r["qa_status"]] = qa_counts.get(r["qa_status"], 0) + 1
    grade_counts = {}
    for r in q_rows:
        grade_counts[r["q_id_grade"]] = grade_counts.get(r["q_id_grade"], 0) + 1

    completeness = {
        "chapter_id": chapter_id, "subject": subject,
        "chapter_no": chapter_no, "ts": _now(),

        "question_records": len(q_rows),
        "answer_records": len(a_rows),
        "solution_records": len(s_rows),
        "image_manifest_records": len(manifest_rows),

        "incomplete_questions": sum(
            1 for r in q_rows if r["extraction_status"] == "INCOMPLETE"),
        "incomplete_answers": sum(
            1 for r in a_rows if r["extraction_status"] == "INCOMPLETE"),
        "incomplete_solutions": sum(
            1 for r in s_rows if r["extraction_status"] == "INCOMPLETE"),

        "unresolved_qid_count": len(un_rows),
        "unresolved_qid_q_nos": sorted(r["q_no"] for r in un_rows),
        "orphan_count": len(orphan_rows),
        "unresolved_image_count": image_report_summary.get("orphans", 0),

        "q_id_grade_counts": grade_counts,
        "extraction_status_counts": {
            "COMPLETE": sum(1 for r in q_rows + a_rows + s_rows
                            if r["extraction_status"] == "COMPLETE"),
            "INCOMPLETE": sum(1 for r in q_rows + a_rows + s_rows
                              if r["extraction_status"] == "INCOMPLETE"),
        },
        "qa_status_counts": qa_counts,
        "pass_provenance_summary": {config.PROV_TEXT_LAYER: len(q_rows)},

        # v2 additions (audit of the deterministic run)
        "census": scan_summary,
        "glyph_fix_counts": glyph_audit,
        "images": image_report_summary,
        "tables": _with_final_refine(table_stats, final_refine_stats),
        "phase2_pending_anchors": {},
    }
    _atomic_write_json(ch_dir / "chapter_completeness.json", completeness)
    return completeness


def write_chapters_json(path: Path, chapters_out: list[dict]) -> None:
    """Dedup by chapter_id, last entry wins (same rule as v1)."""
    uniq = {}
    for c in chapters_out:
        uniq[c["chapter_id"]] = c
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, list(uniq.values()))


def append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
