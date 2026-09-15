"""Post-run audit scan — READ-ONLY, never modifies extraction output.

Flags content-level failure classes that the extraction-time envelope
cannot see (the char-identity envelope owns spacing; this owns
CONTENT):

  numeric_drift       every number printed in the output must exist in
                      the text layer of the pages the row was taken
                      from (each source page +/-1). Catches a digit the
                      model invented or mutated. Evidence comes from
                      data/page_text.jsonl, dumped during the run; when
                      it is absent the check is SKIPPED, never faked.
  duplicate_question  two rows whose normalised stems share >=80% of
                      their 8-token shingles — a question extracted
                      twice under different ids.
  thin_options        a question row with fewer than 2 non-empty
                      options.
  bad_answer          answer key points at a letter that is not one of
                      the question's options.

Output: data/audit_report.jsonl (rewritten per scan) + a stdout
summary. Severity is advisory — nothing here gates the export; the
human decides, in the dashboard, whether a flag is real.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_NUM_RE = re.compile(r"\d[\d,\.]*")


def num_tokens(text: str) -> set:
    """Numeric tokens, normalised (comma grouping stripped, trailing
    separators dropped): '1,000' == '1000', '5.' == '5'."""
    out = set()
    for m in _NUM_RE.findall(text or ""):
        v = m.replace(",", "").rstrip(".,")
        if v:
            out.add(v)
    return out


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ",
                  re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower())).strip()


def _shingles(t: str, n: int = 8) -> set:
    toks = t.split()
    if len(toks) < n:
        return {t} if t else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def _evidence_pages(rows) -> set:
    pages = set()
    for r in rows:
        for p in r.get("source_pages") or []:
            pages.update((p - 1, p, p + 1))
    return pages


def _load_page_text(output_root: Path) -> dict | None:
    f = output_root / "data" / "page_text.jsonl"
    if not f.exists():
        return None
    out = {}
    for line in f.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[int(r["page"])] = r.get("text", "")
    return out


def load_page_text(output_root: Path) -> dict | None:
    """Public view of the run's page-text evidence dump (page -> raw
    text layer). None when the dump is absent — callers must then
    treat page-text evidence as unavailable, never as empty."""
    return _load_page_text(Path(output_root))


def audit_book(output_root: Path, subject: str | None = None) -> dict:
    """Scan one subject's split tree (subject=None => all subjects).
    Returns {"flags": [...], "by_kind": {...}, "rows_scanned": n,
    "page_text": bool}."""
    output_root = Path(output_root)
    split = output_root / "split"
    page_text = _load_page_text(output_root)
    flags: list[dict] = []
    stems: list[tuple] = []          # (q_id, chapter_id, shingle set)
    n_rows = 0

    for subj_dir in sorted(split.iterdir()) if split.exists() else []:
        if not subj_dir.is_dir():
            continue
        if subject and subj_dir.name != subject:
            continue
        for ch_dir in sorted(subj_dir.iterdir()):
            if not ch_dir.is_dir():
                continue
            questions = {r["q_id"]: r for r in
                         _iter_jsonl(ch_dir / "questions.jsonl")}
            solutions = {r.get("q_id"): r for r in
                         _iter_jsonl(ch_dir / "solutions.jsonl")}
            answers = {r.get("q_id"): r for r in
                       _iter_jsonl(ch_dir / "answers.jsonl")}
            for qid, q in sorted(questions.items()):
                n_rows += 1
                pages = _evidence_pages(
                    [q] + list(q.get("tables") or []))

                def drift(text, where):
                    if page_text is None:
                        return
                    ev = set()
                    for p in pages:
                        ev |= num_tokens(page_text.get(p, ""))
                    for v in sorted(num_tokens(text) - ev):
                        flags.append({
                            "kind": "numeric_drift", "severity": "REVIEW",
                            "subject": subj_dir.name,
                            "chapter_id": q.get("chapter_id"),
                            "q_id": qid, "where": where, "value": v,
                            "pages": sorted(pages)})

                drift(q.get("question_text", ""), "question_text")
                for o in q.get("options") or []:
                    drift(o.get("text", ""), f"option_{o.get('id')}")
                for t in q.get("tables") or []:
                    drift(t.get("markdown", ""), f"table_{t.get('table_id')}")
                sol = solutions.get(qid)
                if sol:
                    sp = set()
                    for p in sol.get("source_pages") or []:
                        sp.update((p - 1, p, p + 1))
                    if page_text is not None:
                        ev = set()
                        for p in sp:
                            ev |= num_tokens(page_text.get(p, ""))
                        for v in sorted(num_tokens(sol.get("solution_text", "")) - ev):
                            flags.append({
                                "kind": "numeric_drift", "severity": "REVIEW",
                                "subject": subj_dir.name,
                                "chapter_id": q.get("chapter_id"),
                                "q_id": qid, "where": "solution_text",
                                "value": v, "pages": sorted(sp)})

                filled = [o for o in (q.get("options") or [])
                          if str(o.get("text", "")).strip()]
                if len(filled) < 2:
                    flags.append({"kind": "thin_options", "severity": "HIGH",
                                  "subject": subj_dir.name,
                                  "chapter_id": q.get("chapter_id"),
                                  "q_id": qid,
                                  "detail": f"{len(filled)} non-empty options"})

                ans = answers.get(qid)
                if ans is not None:
                    letters = {o.get("id") for o in filled}
                    co = ans.get("correct_option")
                    if not co or co not in letters:
                        flags.append({"kind": "bad_answer", "severity": "HIGH",
                                      "subject": subj_dir.name,
                                      "chapter_id": q.get("chapter_id"),
                                      "q_id": qid,
                                      "detail": f"correct_option={co!r}"})

                # A repeated STEM is not a duplicate question: books
                # reuse "Match the following:" / "The given life cycle
                # belongs to which of the following organisms?" many
                # times with DIFFERENT options and images. The row's
                # whole content joins the fingerprint, so only a row
                # repeated with its answers (and its pictures) is a
                # duplicate.
                full = " ".join([q.get("question_text") or "",
                                 " ".join(str(o.get("text") or "")
                                          for o in (q.get("options") or [])),
                                 " ".join(str(i) for i in
                                          (q.get("question_images") or [])),
                                 str(sorted((t.get("table_id") or "")
                                            for t in (q.get("tables") or [])))])
                stems.append((qid, q.get("chapter_id"),
                              _shingles(_norm_text(q.get("question_text", ""))),
                              _shingles(_norm_text(full))))

    # ---- duplicate detection across the whole scanned set ------------
    index: dict[str, list] = {}
    for i, (_q, _c, sh, _full) in enumerate(stems):
        for s in sh:
            index.setdefault(s, []).append(i)
    seen_pairs, flagged = set(), set()
    for members in index.values():
        if len(members) < 2 or len(members) > 50:
            continue                    # boilerplate shingle: no signal
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                key = (min(i, j), max(i, j))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                sa, sb = stems[i][2], stems[j][2]
                inter = len(sa & sb)
                share = inter / min(len(sa), len(sb)) if sa and sb else 0
                # the options/images must match too — same stem,
                # different choices is the book's own repetition
                fa, fb = stems[i][3], stems[j][3]
                fshare = (len(fa & fb) / min(len(fa), len(fb))
                          if fa and fb else 0)
                if share >= 0.8 and fshare >= 0.8 and j not in flagged:
                    flagged.add(j)
                    flags.append({"kind": "duplicate_question",
                                  "severity": "HIGH",
                                  "subject": stems[j][1].split("-")[0]
                                  if stems[j][1] else None,
                                  "chapter_id": stems[j][1],
                                  "q_id": stems[j][0],
                                  "detail": f"~{share:.0%} shingle overlap "
                                            f"with {stems[i][0]}"})

    by_kind: dict[str, int] = {}
    for f in flags:
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
    return {"flags": flags, "by_kind": by_kind,
            "rows_scanned": n_rows, "page_text": page_text is not None}


def write_report(output_root: Path, res: dict) -> Path:
    out = Path(output_root) / "data" / "audit_report.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(f, ensure_ascii=False)
                             for f in res["flags"])
                   + ("\n" if res["flags"] else ""))
    tmp.replace(out)
    return out
