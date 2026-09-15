#!/usr/bin/env python3
"""
Independent verification of a finished qbank_output/ tree.

Re-reads the SOURCE PDF through a SEPARATE, minimal extraction path
(its own span reader, its own sorting, no qbank parser code except the
TOC page ranges) and proves, per chapter:

  1. CONTENT FIDELITY  every question_text / option / solution_text /
     table markdown is a substring of the chapter's own zone text in
     VISUAL reading order (baselines top-down, left-right inside a
     baseline) after whitespace/glyph normalisation. Nothing invented,
     nothing reordered, nothing corrupted.
  2. COVERAGE          >=97% of the zone's content characters are
     accounted for by the rows (the remainder is chapter titles,
     headers and markers, which are stripped).
  3. ANSWERS           answers.jsonl equals an independent re-parse of
     the answer-key zone.
  4. IMAGES            embedded-image placements + table renders ==
     claimed + orphaned + skipped. Nothing silently lost.

HOW THE TABLE-CELL CHECK WORKS.  A markdown cell must be printed either
as an ordered concatenation of the zone's baseline lines (the cell was
joined from the lines that carried it) or as a contiguous substring of a
zone's text (the refinement re-cut a merged printed cell into rows), and
the table's own declared source_pages are checked the same way.  The
separators the typesetter ran together (";", ",") and the presentation
the refinement may add ("<br>", bullets) are erased on BOTH sides, so
every printed LETTER and DIGIT of a cell must still appear, in order.

WHAT THIS DOES NOT CHECK.  norm() erases ALL whitespace, so this script
cannot see a word-boundary corruption ("incompletely immunized" printing
as "in completely immunized" normalises identically).  That is by design
— re-wrapped prose must still verify — and that class is owned upstream
by the deterministic restore path in qbank/refine.py
(restore_split_words / segmentation_weakens, with segmentation_change a
fatal in refine_final.py) and audited by tools/segmentation_audit.py.
Characterised in tests/test_verify_extraction.py.

Exit code 0 = all checks passed.

Usage:  python3 scripts/verify_extraction.py <pdf> <subject> [output_root]
"""

from __future__ import annotations

import glob
import json
import re
import sys
import unicodedata
from collections import Counter

import pymupdf

GLYPH_EQUIV = str.maketrans({
    "\u00b0": "", "\u2192": "", "\u0394": "", "\u25a0": "",
    '"': "", "'": "", "\u2019": "", "\u2018": "", "-": "",
    "\u2013": "", "\u2014": "", "|": "",
})


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(GLYPH_EQUIV)
    return re.sub(r"\s+", "", s)


# Presentation the table pipeline is allowed to introduce (and that the
# printed page therefore does not show): "<br>" line breaks, bullet
# markers, and the list separators ";" / "," between items that the
# typesetter simply ran together. The table-cell check erases them on
# BOTH sides — it still proves every LETTER and DIGIT of the cell is
# printed, in order, which is what "content is in the zones" means.
_PRESENTATION = re.compile(r"<br\s*/?>|<[A-Za-z/][^>]{0,12}>|[;,\u2022\u00b7]")


def norm_cell(s: str) -> str:
    """norm() for a markdown table cell (presentation erased)."""
    return norm(_PRESENTATION.sub("", s or ""))


RE_Q_HDR_LINE = re.compile(r"^Question\s+\d+\s*:\s*$", re.I)
RE_S_HDR = re.compile(r"^Solution\s+to\s+Question\s+(\d+)\s*:\s*$", re.I)
RE_KEY_LINE = re.compile(r"^Answer\s*Key\s*$", re.I)
RE_DETAILED = re.compile(r"^Detailed\s+Explanations?\s*$", re.I)
RE_MARKER = re.compile(r"^[a-d]\)")
RE_KEYROW = re.compile(r"^\d{1,3}\s+[a-dA-D]$")
RE_INT = re.compile(r"^\d{1,4}$")


def visual_lines(page, printed: int):
    """Minimal independent line reader: (y0, x0, x1, text) per text
    line, footers dropped. A footer is the PRINTED PAGE NUMBER standing
    alone near the bottom AND horizontally centred on the page — the
    same two signals the pipeline uses, because the naive "number low
    on the page" test silently deleted page 14's answer-key row
    `14  d` (that row's number is not centred and has its option letter
    beside it). A verifier that repeats a parser bug cannot catch it,
    so this rule is derived from page geometry, not copied."""
    out = []
    h, w = page.rect.height, page.rect.width
    lines = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for ln in b["lines"]:
            text = "".join(s["text"] for s in ln["spans"]).strip()
            if not text:
                continue
            lines.append((tuple(ln["bbox"]), text))
    for bb, text in lines:
        keep = True
        if text == str(printed) and bb[3] > h * 0.88:
            centred = abs((bb[0] + bb[2]) / 2 - w / 2) <= w * 0.05
            beside = [ob for ob, _t in lines
                      if not (ob[3] <= bb[1] - 2 or ob[1] >= bb[3] + 2)
                      and ob != bb]
            keep = not (centred and not beside)
        if keep:
            out.append((bb[1], bb[0], bb[2], text))
    return out


GAP = 22.0     # (unused; kept for reference)


def ruled_boxes(page) -> list:
    """Independent re-implementation of the parser's ruled-table
    detection: clusters of long horizontal rules (+verticals)."""
    h, v = [], []
    for it in page.get_drawings():
        for item in it["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                seg = (min(p1.x, p2.x), min(p1.y, p2.y),
                       max(p1.x, p2.x), max(p1.y, p2.y))
            elif item[0] == "re":
                r = item[1]
                seg = (r.x0, r.y0, r.x1, r.y1)
            else:
                continue
            if seg[3] - seg[1] < 2.5 and seg[2] - seg[0] > 60:
                h.append(seg)
            elif seg[2] - seg[0] < 2.5 and seg[3] - seg[1] > 8:
                v.append(seg)
    TOL = 3.0
    segs = sorted([(s, "h") for s in h] + [(s, "v") for s in v],
                  key=lambda sk: (sk[0][1], sk[0][0]))
    boxes = []

    def _hits(seg):
        return [bx for bx in boxes
                if (seg[0] <= bx[2] + TOL and seg[2] >= bx[0] - TOL
                    and seg[1] <= bx[3] + TOL and seg[3] >= bx[1] - TOL)]

    for seg, kind in segs:
        hits = _hits(seg)
        if not hits:
            boxes.append([seg[0], seg[1], seg[2], seg[3],
                          1 if kind == "h" else 0,
                          1 if kind == "v" else 0])
            continue
        base = hits[0]
        for other in hits[1:]:
            base[0] = min(base[0], other[0]); base[1] = min(base[1], other[1])
            base[2] = max(base[2], other[2]); base[3] = max(base[3], other[3])
            base[4] += other[4]; base[5] += other[5]
            boxes.remove(other)
        base[0] = min(base[0], seg[0]); base[1] = min(base[1], seg[1])
        base[2] = max(base[2], seg[2]); base[3] = max(base[3], seg[3])
        base[4 if kind == "h" else 5] += 1
    return [(b[0], b[1], b[2], b[3]) for b in boxes
            if b[4] >= 3 or (b[4] >= 2 and b[5] >= 2)]


def zone_texts(doc, first: int, last: int, offset: int):
    """(q_full, q_prose, k_lines, s_full, s_prose).

    Zones split at the 'Answer Key' baseline and the first 'Solution to
    Question' baseline, in visual order. Baselines inside a RULED
    table box are dropped from the *_prose zones (the parser moves
    them into the tables field). *_full keeps everything — table
    markdown is checked against the full zone."""
    q_all, k_lines, s_all = [], [], []
    q_dl, s_dl = [], []        # individual dict-lines per zone (cell checks)
    mode = "Q"
    for pno in range(first - 1, last):
        lines = visual_lines(doc[pno], pno + 1 - offset)
        boxes = ruled_boxes(doc[pno])
        buckets: dict = {}
        for y0, x0, x1, text in lines:
            buckets.setdefault(round(y0 / 3.0), []).append((x0, y0, text))
        for bkey in sorted(buckets):
            row = sorted(buckets[bkey])
            joined = " ".join(t for _x0, _y0, t in row)
            ycen = sum(y for _x, y, _t in row) / len(row)
            in_table = any(b[1] - 3 <= ycen <= b[3] + 3 for b in boxes)
            first_t = row[0][2]
            if mode == "Q" and RE_KEY_LINE.match(first_t):
                mode = "K"
                continue
            if mode == "K" and RE_S_HDR.match(first_t):
                mode = "S"
            if mode == "Q":
                q_all.append((joined, in_table))
                q_dl.extend(t for _x, _y, t in row)
            elif mode == "K":
                k_lines.append(joined)
            else:
                s_all.append((joined, in_table))
                s_dl.extend(t for _x, _y, t in row)

    def strip_runs(pairs):
        return [t for t, in_table in pairs if not in_table]

    def content(lines):
        kept = []
        for t in lines:
            if (RE_Q_HDR_LINE.match(t) or RE_S_HDR.match(t)
                    or RE_DETAILED.match(t) or RE_KEYROW.match(t)
                    or RE_KEY_LINE.match(t)):
                continue
            kept.append(RE_MARKER.sub("", t))
        return kept

    q_texts = [t for t, _m in q_all]
    s_texts = [t for t, _m in s_all]
    return (norm(" ".join(content(q_texts))),
            norm(" ".join(content(strip_runs(q_all)))),
            k_lines,
            norm(" ".join(content(s_texts))),
            norm(" ".join(content(strip_runs(s_all)))),
            [norm(RE_MARKER.sub("", t)) for t in q_dl],
            [norm(RE_MARKER.sub("", t)) for t in s_dl])


def _cell_in_lines(cell_norm: str, line_norms) -> bool:
    """A reconstructed table cell must equal an ORDERED concatenation
    of zone dict-lines (the cell was joined from its own lines;
    joining only removes/keeps spaces, which norm() erases). Non-
    greedy: every possible consumption frontier is tracked, so an
    earlier line that merely PREFIX-matches the cell (e.g. "T1" vs
    cell "T1a") cannot sabotage the exact single-line match."""
    if not cell_norm:
        return True
    n = len(cell_norm)
    reach = {0}
    for ln in line_norms:
        if not ln:
            continue
        nxt = set()
        for pos in reach:
            if pos < n and cell_norm.startswith(ln, pos):
                p2 = pos + len(ln)
                if p2 == n:
                    return True
                nxt.add(p2)
        reach |= nxt
    return False


def _page_line_sets(doc, pages, offset: int):
    """(line norms, joined text) for the file pages a table declares.
    A table's cells must be reconstructible from the pages it names —
    that is a stricter claim than the chapter zone (which classifies
    lines into question/solution/details buckets and can drop the ones
    a table was assembled from)."""
    lines, texts = [], []
    for pg in pages:
        fp = int(pg) + int(offset)
        if fp < 1 or fp > doc.page_count:
            continue
        page = doc[fp - 1]
        ls = []
        for blk in page.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                txt = "".join(sp["text"] for sp in ln["spans"])
                if txt.strip():
                    ls.append(norm_cell(txt))
        lines.append(ls)
        texts.append(norm_cell(page.get_text()))
    return lines, texts


def _table_ok(md: str, line_sets, zone_texts=()) -> bool:
    """Every non-separator markdown row's non-empty cell must be
    PRINTED in the chapter: either as an ordered concatenation of the
    zone's baseline lines (the extraction case — a cell joined from the
    lines that printed it), or as a contiguous substring of a zone's
    whole text (the refinement case — a cell the model re-cut, e.g. one
    merged cell split into three, still has to exist on the page
    letter-for-letter). Presentation is erased on both sides."""
    for row in md.splitlines():
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if all(set(c) <= set("- ") for c in cells):
            continue                      # |---|---| separator
        for c in cells:
            nc = norm_cell(c)
            if not nc:
                continue
            if any(_cell_in_lines(nc, ls) for ls in line_sets):
                continue
            if any(nc in zt for zt in zone_texts):
                continue
            return False
    return True


def main() -> int:
    pdf, subject = sys.argv[1], sys.argv[2]
    out_root = sys.argv[3] if len(sys.argv) > 3 else "qbank_output"
    doc = pymupdf.open(pdf)
    failures = []
    stats = Counter()
    worst_q = worst_s = 1.0

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from qbank.textlayer import Book
    from qbank.toc import assign_file_ranges, detect_page_offset, parse_toc
    book = Book(pdf)
    book.set_offset(detect_page_offset(book))
    toc = parse_toc(book)
    assign_file_ranges(toc, book.page_offset, book.total_pages)
    ranges = {c.chapter_no: (c.file_start, c.file_end) for c in toc}
    offset = book.page_offset
    book.close()

    chapters = [c for c in json.load(open(f"{out_root}/data/chapters.json"))
                if c["subject"] == subject]

    for ch in chapters:
        cid, no = ch["chapter_id"], ch["chapter_no"]
        first, last = ranges[no]
        (nqz, nqp, k_lines, nsz, nsp,
         q_lns, s_lns) = zone_texts(doc, first, last, offset)

        rows = {f.split("/")[-1]: [json.loads(l) for l in open(f)]
                for f in glob.glob(f"{out_root}/split/{subject}/{cid}/*.jsonl")}
        qrows, srows, arows = (rows["questions.jsonl"],
                               rows["solutions.jsonl"], rows["answers.jsonl"])

        # 1+2. fidelity and coverage (tables counted once — the shared
        # list rides on both question and solution rows)
        covered = 0
        for r in qrows:
            for val in [r["question_text"]] + [o["text"] for o in r["options"]]:
                nv = norm(val)
                if not nv:
                    continue
                if nv not in nqp and nv not in nsp:
                    failures.append(f"{r['q_id']}: text not found in zones")
                covered += len(nv)
            for t in r.get("tables") or []:
                md = t.get("markdown", "")
                if not md:
                    continue
                # the cell check erases presentation on BOTH sides (the
                # page's own commas/semicolons included: the typesetter
                # runs list items together without separators)
                plines, ptexts = _page_line_sets(doc, t.get("source_pages")
                                                 or [], offset)
                if not _table_ok(md, [[norm_cell(x) for x in q_lns],
                                      [norm_cell(x) for x in s_lns]]
                                 + plines,
                                 (norm_cell(nqz), norm_cell(nsz))
                                 + tuple(ptexts)):
                    failures.append(f"{r['q_id']}: table cells not "
                                    f"verifiable in zones ({t.get('table_id')})")
                covered += len(norm(md))
        for r in srows:
            nv = norm(r["solution_text"])
            if nv:
                if nv not in nsp and nv not in nqp:
                    failures.append(f"{r['q_id']}: solution text not in zones")
                covered += len(nv)
        cov_q = covered / max(1, len(nqz) + len(nsz))
        worst_q = min(worst_q, cov_q)
        if cov_q < 0.97:
            failures.append(f"{cid}: content coverage {cov_q:.3f} < 0.97")

        # 3. answers — independent key re-parse (rows come as 'N' + letter
        # on one baseline in these books; accept both split and joined)
        pairs = {}
        prev = None
        for t in k_lines:
            t = t.strip()
            if RE_INT.fullmatch(t):
                prev = int(t)
            elif re.fullmatch(r"[a-dA-D]", t) and prev is not None:
                pairs.setdefault(prev, t.upper())
                prev = None
            else:
                m = re.fullmatch(r"(\d{1,3})\s+([a-dA-D])", t)
                if m:
                    pairs.setdefault(int(m.group(1)), m.group(2).upper())
                    prev = None
                else:
                    prev = None
        for r in arows:
            if pairs.get(r["q_no"]) != r["correct_option"]:
                failures.append(
                    f"{r['q_id']}: answer {r['correct_option']} != "
                    f"independent key {pairs.get(r['q_no'])}")

        # 4. images — placements + table renders fully accounted for
        placed = sum(len(doc[pno].get_image_info())
                     for pno in range(first - 1, last))
        comp = json.load(open(
            f"{out_root}/split/{subject}/{cid}/chapter_completeness.json"))
        imgs = comp["images"]
        accounted = (imgs["claimed"] - imgs.get("table_renders", 0)
                     + imgs["orphans"] + imgs["skipped"]
                     + imgs.get("merged_placements", 0))
        if placed != accounted:
            failures.append(
                f"{cid}: {placed} placements vs {accounted} accounted "
                f"({imgs})")

        # 5. TABLE/IMAGE SEPARATION — a region classified TABLE and
        #    successfully structured must never also ship as an image
        #    asset (duplicate content). Manifest rows carry table_id
        #    only for table clip renders; genuine figures have none.
        structured = {t["table_id"]
                      for r in list(qrows) + list(srows)
                      for t in r.get("tables") or []
                      if (t.get("markdown") or "").strip()}
        man_rows = rows.get("image_manifest.jsonl") or []
        dup = sorted({m.get("table_id") for m in man_rows
                      if m.get("table_id") in structured})
        if dup:
            failures.append(
                f"{cid}: structured tables also shipped as image "
                f"assets: {dup}")
        stats["questions"] += len(qrows)
        stats["images"] += placed
        stats["tables"] += imgs.get("table_renders", 0)

    print(f"verified {len(chapters)} chapters, {stats['questions']} questions, "
          f"{stats['images']} embedded images + {stats['tables']} table renders")
    print(f"worst content coverage: {worst_q:.4f}")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures[:40]:
            print(" -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
