"""Final table refinement — presentation + medically safe extraction repair.

The LAST stage that may touch a table, inside run_chapter, after the
in-extraction Gemini rearrangement and before the split files are
written. It reuses the existing records, table geometry, QA flags,
review queue and ledger conventions — nothing here re-extracts or
re-merges anything:

    MODES (QBANK_FINAL_REFINE)
        all (default) -> EVERY table is sent for Gemini refinement;
                         a table that needs no correction comes back
                         as NO_CHANGE and ships byte-identical
        flagged       -> only QA-flagged tables are sent
        off           -> the stage is disabled entirely
    A deterministic pre-check (this module, zero API) still runs on
    every table and its reasons are recorded on the ledger row for
    auditing — it no longer gates the call. The Gemini visual
    refinement receives the crop image of the printed table + the
    current pipe-markdown + metadata + optional context text.
    DETERMINISTIC FIDELITY VALIDATOR (before vs after, zero API)
        same rows x columns, every cell maps 1:1, and the cell's
        character stream is identical after whitespace / <br> /
        bullet-marker normalisation — OR a number restored with
        source page-text evidence (data/page_text.jsonl)
        ACCEPT -> save; pre-final markdown kept for provenance
        REVIEW -> keep current markdown, flag the human queue
                  (L3: medical knowledge suggests, PDF unclear)
        REJECT -> keep current markdown, loud log + audit row
                  (hallucinated addition / deletion / substitution /
                   changed number without source evidence / structural)

Medical knowledge is a VALIDATION signal only. A repair is accepted
only when this module deterministically proves the content unchanged
(character-identity), which is exactly the shape of the allowed
repairs: "layere d" -> "layered", "osteocal cin" -> "osteocalcin",
"retractionnot" -> "retraction not", "50 – 300" -> "50–300". Any
change that adds, deletes or substitutes a character of content is
rejected — the source PDF stays the authority.

No image asset is ever created by this stage: crops are sent to
Gemini inline, the structured markdown remains the primary
representation, and the existing asset-dedup / table-render machinery
is untouched.

Ledger + audit:
  data/table_refinement.jsonl     one row per table processed
  data/table_refinement_audit.json  book-level report (write_audit)
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

from .audit import load_page_text, num_tokens, page_num_evidence
from .refine import (flagged as qa_flagged, restore_split_words,
                     salvage_table, segmentation_weakens)
from .tables import qa_suspects

LEDGER = "table_refinement.jsonl"
AUDIT = "table_refinement_audit.json"

# pre-check thresholds (deterministic; the reasons are recorded on
# the ledger row for auditing — in `all` mode they no longer gate
# the Gemini call, the safety comes from the fidelity validator)
LONG_CELL_CHARS = 70
LIST_SEPARATORS = 2          # "A; B; C" = 2 separators = list-like

# bullet markers as list bullets: • · * + always, -/–/— ONLY when the
# item does not start with a digit ("50 – 300" is a RANGE, never a
# bullet — a dash glued between digits must survive the normalisation)
_BULLET_CHAR = re.compile(r"(?:^|\s)(?:\u2022|\u00b7|[*+])\s+")
_BULLET_DASH = re.compile(r"(?<![\d.])(?:^|\s)(?:[-\u2013\u2014])\s+(?!\d)")
_BR = re.compile(r"</?[Bb][Rr]\s*/?>")
# list separators "; " / ", " become line breaks (bullets) in a
# presentation refinement — normalised symmetrically on BOTH sides so
# the conversion is judged on the item content, never on the marker.
# The separator and the whitespace AROUND it are both presentation:
# "antibody,colostrum" and "antibody, colostrum" are the same cell
# (the book's text layer drops that space all over the tables), and
# before this the missing space made the validator call a correct
# spacing repair a `deletion` and throw it away.
_SEP = re.compile(r"[;,]\s*")
# The book's OTHER list separator: the typesetter ran a period straight
# into the next item ("…apiospermum).Madurella mycetomatis", "…, B.
# cereus.Clostridium"). It is presentation exactly like "; " / ", ", so
# it is folded on BOTH sides — a refinement that turns it into <br>
# bullets is restructuring, not deleting a printed character. Guarded to
# the run-in shape ("), ." + Capital), so abbreviation periods ("S.
# aureus"), decimals, and a "." the model invents at a word's end
# ("E. jeanselmei") all stay content.
_SEP_DOT = re.compile(r"(?<=[)\w])\.(?=\s*[A-Z])")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _new_mid_word_breaks(b: str, a: str) -> int:
    """Count the <br> tags in `a` that land where the SOURCE `b` had
    NO space between two letters — a true mid-word break ("recep
    <br>tor"). A <br> that replaces a space already in the source
    ("dome shaped" -> "dome<br>shaped") is a normal line break and is
    not counted. Requires both cells to share the same content
    character stream (the caller only uses this on character-identical
    cells)."""
    def content_positions(s: str):
        out, i, n = [], 0, len(s)
        while i < n:
            m = _BR.match(s[i:])
            if m:
                i += m.end()
                continue
            if s[i].isspace():
                i += 1
                continue
            out.append(i)
            i += 1
        return out
    pb, pa = content_positions(b), content_positions(a)
    if not pb or len(pb) != len(pa):
        return 0
    count, i, n_before, na = 0, 0, 0, len(a)
    while i < na:
        m = _BR.match(a[i:])
        if m:
            if 0 < n_before < len(pb):
                # source characters around the corresponding boundary:
                # gap == "" means the two letters were ADJACENT in the
                # source (a real mid-word break); any gap is whitespace
                gap = b[pb[n_before - 1] + 1:pb[n_before]]
                if gap == "":
                    count += 1
            i += m.end()
            continue
        if not a[i].isspace():
            n_before += 1
        i += 1
    return count

# rejected answers are printed once, like the in-extraction pass
REJECT_SAMPLES = 3
_reject_samples_left = REJECT_SAMPLES


def reset_samples() -> None:
    """Fresh sample budget per run (process-global, dashboard is
    long-lived — same pattern as refine.reset_samples)."""
    global _reject_samples_left
    _reject_samples_left = REJECT_SAMPLES


def _note_reject(t: dict, why: str) -> None:
    global _reject_samples_left
    if _reject_samples_left <= 0:
        return
    _reject_samples_left -= 1
    print(f"[refine-final] {(t.get('table_id') or '?')}: REJECTED — {why} "
          f"(current table kept)", flush=True)


# ------------------------------------------------------------- parsing

def parse_md(md: str) -> list | None:
    """Rows (list of cell strings) of one even pipe-markdown table,
    separator row excluded; None when it is not a table."""
    lines = [l for l in (md or "").strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    rows = []
    for i, l in enumerate(lines):
        s = l.strip()
        if i == 1 and re.fullmatch(r"[\s|:\-]+", s):
            continue
        if not (s.startswith("|") and s.endswith("|")):
            return None
        rows.append([c.strip() for c in s[1:-1].split("|")])
    if not rows:
        return None
    return rows


# Typographic variants of the SAME character. The source PDF is set
# with curly quotes and thin spaces; a model that answers with the
# straight equivalents is not changing content, and the validator used
# to call it `content_substitution` and throw a CORRECT fix away (the
# live case: the printed cell "the host ’s immune system" — the
# typesetter wrapped the line before "’s" — came back as "the host's
# immune system" and was rejected). Applied SYMMETRICALLY inside
# display_text, so it can never hide a difference that exists on one
# side only.
_CHAR_EQUIV = str.maketrans({
    "\u2019": "'", "\u2018": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"',
    "\u00a0": " ", "\u2007": " ", "\u202f": " ", "\u2009": " ",
})


def display_text(text: str) -> str:
    """Presentation-normalised view of a cell: <br> -> space, list
    separators "; " / ", " -> space, leading bullet markers dropped,
    typographic quote/space variants folded to their ASCII form,
    whitespace collapsed. Used for token comparisons and for stripping
    presentation before the character identity check (applied
    SYMMETRICALLY to both sides, so it can never hide a content change
    on one side only)."""
    t = (text or "").translate(_CHAR_EQUIV)
    t = _BR.sub(" ", t)
    t = _SEP.sub(" ", t)
    t = _SEP_DOT.sub(" ", t)
    t = _BULLET_CHAR.sub(" ", t)
    t = _BULLET_DASH.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def fidelity_chars(text: str) -> str:
    """The cell's character stream: presentation stripped (as in
    display_text), ALL whitespace removed. Two cells are
    content-equivalent iff these are equal. Case is NOT folded — in
    medical text it is content ("IgG" -> "igg" is a corruption, not a
    spacing fix)."""
    return re.sub(r"\s+", "", display_text(text))


def _nums_ordered(text: str) -> list:
    return [m.group(0).replace(",", "")
            for m in _NUM.finditer(text or "")]


# --------------------------------------------------------- pre-check

def precheck(t: dict, vocab=None) -> list:
    """Deterministic suspicion signals, recorded on the ledger row
    for auditing. In `all` mode (the default) they no longer gate
    the Gemini call — every table is sent; the safety still comes
    from the fidelity validator below."""
    reasons = []
    v = t.get("validation") or {}
    qa = v.get("table_qa") or {}
    md = t.get("markdown") or ""
    if qa.get("status") == "REVIEW" or qa.get("suspect_fragments"):
        reasons.append("qa_flagged")
    if any(str(w).startswith("suspect_lost_space")
           for w in v.get("warnings") or []):
        reasons.append("lost_space")
    rows = parse_md(md)
    if rows:
        cells = [c for r in rows for c in r]
        if max((len(c) for c in cells), default=0) > LONG_CELL_CHARS:
            reasons.append("long_cell")
        # list-like content packed into one cell ("A; B; C" = 2
        # separators) — a bullet candidate for the refinement
        if any(len(re.findall(r"[;]\s|\u2022", c)) >= LIST_SEPARATORS
               for c in cells):
            reasons.append("list_like_cell")
    if re.search(r"\d\s+[-\u2013\u2014]\s+\d", md):
        reasons.append("range_spacing")
    if ("  " in md or re.search(r"\S\s+[.,;:!?\)]", md)
            or re.search(r"[\(]\s{2,}", md)):
        reasons.append("spacing_artifact")
    if vocab is not None and rows:
        susp = qa_suspects(rows, vocab[0], vocab[1])
        if susp:
            reasons.append("suspect_fragment")
    return reasons


# -------------------------------------------------------- render QA

def render_qa(md: str) -> list:
    """Deterministic page-waste / readability findings on one
    table's markdown (run before AND after refinement; findings are
    advisory — they are never fixed by deleting information)."""
    findings = []
    rows = parse_md(md)
    if rows is None:
        return ["invalid_markdown"]
    ncols = len(rows[0])
    widths = []
    for c in range(ncols):
        widths.append(max((len(r[c]) for r in rows if c < len(r)),
                          default=0))
    if any(w <= 3 for w in widths) and any(w >= 30 for w in widths):
        findings.append("extremely_narrow_column")
    cells = [c for r in rows for c in r]
    if any(len(c) > 240 for c in cells):
        findings.append("unreadably_wide_cell")
    if any(c != c.strip() or re.search(r"[ \t]{2,}", c) for c in cells):
        findings.append("stray_whitespace")
    if any(re.search(r"\S\s+[.,;:!?\)]$", c) for c in cells):
        findings.append("punctuation_spacing")
    empty = sum(1 for c in cells if not c)
    if cells and empty > len(cells) / 3:
        findings.append("mostly_empty_cells")
    return findings


# ----------------------------------------------- fidelity validator

def _norm_evidence(mch: dict | None) -> str | None:
    if not mch:
        return None
    ev = str(mch.get("evidence") or "").strip().lower()
    if not ev:
        return None
    if ev in ("visual", "image", "crop"):
        return "visual"
    if ev in ("visual+medical", "visual+medical knowledge",
              "image+medical", "visual+vocab"):
        return "visual+medical"
    if ev in ("medical", "medical_only", "knowledge", "medical knowledge"):
        return "medical"
    if ev in ("presentation", "formatting", "layout"):
        return "presentation"
    if ev in ("spacing", "whitespace", "wrap"):
        return "spacing"
    return ev


def _confidence(mch: dict | None):
    if not mch:
        return None
    c = mch.get("confidence")
    if isinstance(c, (int, float)) and not isinstance(c, bool):
        return round(float(c), 3)
    return None


def _classify_token_change(tb: list, ta: list, vocab) -> str:
    """Character-identical but token stream changed: which repair?
    word_restore  a merged join forms an established book word
                  ("osteocal"+"cin" -> "osteocalcin", "layere"+"d"
                  -> "layered") — the medically obvious spelling
                  corruption class
    spacing       un-glue / re-split / plain re-wrap
                  ("retractionnot" -> "retraction not")"""
    from .refine import _split_vocab
    words, _ = _split_vocab(vocab)     # bare Counter or (words, pairs)
    if words is None:
        return "spacing"
    import difflib
    sm = difflib.SequenceMatcher(a=[x.lower() for x in tb],
                                 b=[x.lower() for x in ta], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        bb, ba = tb[i1:i2], ta[j1:j2]
        if len(bb) >= 2 and len(ba) == 1 and words.get(ba[0].lower(), 0) >= 2:
            return "word_restore"
    return "spacing"


def _letters_chars(text: str) -> str:
    """Character stream WITHOUT the numeric tokens (letters, symbols,
    punctuation) — presentation-normalised, case preserved. A number
    repair may never move a letter; a letter repair may never hide a
    number change; a case change ("IgG" -> "igg") is a substitution,
    never a reorder."""
    return re.sub(r"\s+", "", re.sub(_NUM, " ", display_text(text)))


def _mark_keys(text: str) -> Counter:
    """Multiset of (letter the mark hangs on, mark) for every
    verifier-visible mark (".", "-", "/") of a cell.

    The same rule the rearrange envelope uses (refine._punct_keys): a
    separator the typesetter ran into the text may be DROPPED, but a
    mark must not MOVE to another word. Counting alone cannot see a
    move — "…apiospermum). Madurella … grisea E jeanselmei" rewritten as
    "… grisea<br>• E. jeanselmei" takes one period out and puts one in —
    so the move was classified `reorder` (a REVIEW) and locked the
    export gate on a table whose text was perfectly faithful.

    One letter of context is deliberate: the pass re-segments words
    ("antibodyp-ANCA" -> "antibody p-ANCA"), which changes a longer
    context but never the letter the mark sits on."""
    out: Counter = Counter()
    tail = ""
    for ch in display_text(text):
        if ch.isalnum():
            tail = ch
        elif ch.isspace():
            continue
        elif ch in "./-":
            out[(tail, ch)] += 1
        else:
            tail = ""
    return out


def _marks_moved(before: str, after: str) -> list:
    """Marks `after` carries that `before` does not, i.e. marks that
    appeared on a different word (readable form for the report)."""
    return [f"{ch} on {tail!r}" if tail else ch
            for (tail, ch), n in (_mark_keys(after)
                                  - _mark_keys(before)).items()
            for _ in range(n)]


def _cell_change(bc: str, ac: str, words, page_nums, mch: dict | None) -> dict | None:
    """Compare ONE cell pair. None = untouched; otherwise a change
    record, optionally with a 'fatal' fidelity violation.

    Fatal classes (whole table rejected):
      changed_number         number added/changed/dropped without
                             source page-text evidence
      hallucinated_addition  letters/symbols added
      deletion               letters/symbols removed
      content_substitution   letters/symbols swapped
    Non-fatal:
      presentation / spacing / word_restore / number_repair  (accepted)
      reorder            (-> REVIEW: same content, different order)"""
    b, a = bc.strip(), ac.strip()
    if b == a:
        return None
    base = {"reason": (mch.get("reason") if mch else None),
            "evidence": _norm_evidence(mch),
            "confidence": _confidence(mch)}
    fb, fa = fidelity_chars(b), fidelity_chars(a)
    if fb == fa:
        # content-identical: presentation or a spacing-level repair
        if _new_mid_word_breaks(b, a) > 0:
            # a NEW line break inside a word ("recep<br>tor"): the
            # model's layout broke the source's words. This is not an
            # ambiguous medical reading, it is damage — refused as
            # fatal (live: 032-T02 shipped a REVIEW for exactly this
            # and the human queue would have had nothing to decide).
            return {**base, "kind": "mid_word_break",
                    "fatal": "mid_word_break"}
        tb, ta = display_text(b).split(), display_text(a).split()
        if tb == ta:
            kind = "presentation"
        elif Counter(tb) == Counter(ta):
            kind = "reorder"        # -> REVIEW at table level
        elif (weak := segmentation_weakens(display_text(b), display_text(a),
                                           words))[0]:
            # same letters, WORSE word boundaries than the book prints
            # ("incompletely" -> "in completely"): content corruption,
            # not a spacing fix — see refine.segmentation_weakens
            return {**base, "kind": "segmentation_change",
                    "fatal": "segmentation_change",
                    "detail": {"boundaries": weak[1]}}
        else:
            kind = _classify_token_change(tb, ta, words)
            if base["evidence"] is None:
                base["evidence"] = "not_reported"
            if base["confidence"] is None:
                # character identity was PROVEN by this validator
                base["confidence"] = 0.9
        return {**base, "kind": kind}
    # ---- character stream differs ----------------------------------
    nb, na = _nums_ordered(b), _nums_ordered(a)
    cnb, cna = Counter(nb), Counter(na)
    num_repair = False
    nadded, nremoved = [], []
    if cnb != cna:
        nadded, nremoved = list((cna - cnb).elements()), \
            list((cnb - cna).elements())
        supported = (page_nums is not None and nadded
                     and all(n in page_nums for n in nadded)
                     and all(n not in page_nums for n in nremoved))
        if not supported:
            return {**base, "kind": "changed_number",
                    "fatal": "changed_number",
                    "detail": {"added": nadded, "removed": nremoved}}
        num_repair = True
    elif nb != na:
        return {**base, "kind": "reorder"}      # number order swapped
    # the letter/symbol stream must be untouched (or re-ordered)
    lb, la = _letters_chars(b), _letters_chars(a)
    if lb != la:
        if Counter(lb) == Counter(la):
            moved = _marks_moved(b, a)
            if moved:
                # same characters, but a mark ended up on another word:
                # not a reorder, and not presentation either — content
                return {**base, "kind": "mark_moved",
                        "fatal": "mark_moved",
                        "detail": {"marks": moved[:6]}}
            return {**base, "kind": "reorder"}
        clb, cla = Counter(lb), Counter(la)
        added_l, removed_l = list((cla - clb).elements()), \
            list((clb - cla).elements())
        if added_l and not removed_l:
            fatal = "hallucinated_addition"
        elif removed_l and not added_l:
            fatal = "deletion"
        else:
            fatal = "content_substitution"
        detail = {"added": added_l[:8], "removed": removed_l[:8]}
        if num_repair:
            detail["numbers"] = {"added": nadded, "removed": nremoved}
        return {**base, "kind": fatal, "fatal": fatal, "detail": detail}
    if num_repair:
        return {**base, "kind": "number_repair",
                "evidence": "page_text", "confidence": 0.95,
                "reason": base["reason"] or
                "number restored from source page text",
                "detail": {"numbers": {"added": nadded,
                                       "removed": nremoved}}}
    return {**base, "kind": "reorder"}


def fidelity_compare(before_md: str, after_md: str, page_nums=None,
                     vocab=None, model_action: str | None = None,
                     model_changes=None) -> dict:
    """Deterministic BEFORE vs AFTER judgement.

    verdicts:
      ACCEPT  structure identical + every cell character-identical
              after whitespace/<br>/bullet normalisation (numbers
              may differ only with page-text evidence);
      REVIEW  model asks for a human (action=REVIEW), a content
              repair was reported as medical-knowledge-only (L3),
              a reported repair's confidence is < 0.5, or words were
              re-ordered inside a cell;
      REJECT  invalid markdown, row/column structural change,
              hallucinated addition, deletion, content substitution
              or a number change without source-page evidence.
    """
    words = vocab          # full (words, pairs) when available: the
    #                        boundary rule needs the pair evidence too
    out = {"verdict": "ACCEPT", "reject_reasons": [],
           "review_reasons": [], "cells_changed": [], "cells_checked": 0}
    br = parse_md(before_md)
    ar = parse_md(after_md)
    if br is None or ar is None:
        out["verdict"] = "REJECT"
        out["reject_reasons"] = ["invalid_markdown"]
        return out
    if len(br) != len(ar) or [len(r) for r in br] != [len(r) for r in ar]:
        out["verdict"] = "REJECT"
        out["reject_reasons"] = [
            f"structural_change:{len(br)}x{len(br[0]) if br else 0}->"
            f"{len(ar)}x{len(ar[0]) if ar else 0}"]
        return out
    mch_by_id = {}
    mch_list = []
    for c in (model_changes or []):
        if not isinstance(c, dict):
            continue
        mch_list.append(c)
        if c.get("cell"):
            mch_by_id[str(c["cell"]).upper().replace(" ", "")] = c
    used: set = set()

    def _match(cell_id: str, before: str, after: str) -> dict | None:
        """The model's change record for this cell: exact cell ref
        first, then content match (models miscount rows surprisingly
        often — a (before, after) pair is the reliable key)."""
        m = mch_by_id.get(cell_id)
        if m is not None and id(m) not in used:
            used.add(id(m))
            return m
        for k, c in enumerate(mch_list):
            if id(c) in used:
                continue
            if (str(c.get("before", "")).strip() == before.strip()
                    and str(c.get("after", "")).strip() == after.strip()):
                used.add(id(c))
                return c
        for k, c in enumerate(mch_list):
            if id(c) in used:
                continue
            if str(c.get("before", "")).strip() == before.strip():
                used.add(id(c))
                return c
        return None
    for i, (brow, arow) in enumerate(zip(br, ar)):
        for j, (bc, ac) in enumerate(zip(brow, arow)):
            out["cells_checked"] += 1
            ch = _cell_change(bc, ac, words, page_nums,
                              _match(f"R{i + 1}C{j + 1}", bc, ac))
            if ch is None:
                continue
            ch = {"cell": f"R{i + 1}C{j + 1}", "before": bc,
                  "after": ac, **ch}
            out["cells_changed"].append(ch)
            if ch.get("fatal"):
                out["verdict"] = "REJECT"
                out["reject_reasons"].append(
                    f"{ch['fatal']}:R{i + 1}C{j + 1}")
    if out["verdict"] == "REJECT":
        return out
    if (model_action or "").strip().upper() == "REVIEW":
        out["review_reasons"].append("model_action_review")
    for ch in out["cells_changed"]:
        if ch["kind"] in ("reorder", "mid_word_break"):
            out["review_reasons"].append(f"{ch['kind']}:{ch['cell']}")
        elif ch["kind"] not in ("presentation", "number_repair"):
            if ch.get("evidence") == "medical":
                # L3: medical knowledge suggests, PDF evidence unclear
                out["review_reasons"].append(
                    f"medical_only_evidence:{ch['cell']}")
            elif (isinstance(ch.get("confidence"), (int, float))
                  and ch["confidence"] < 0.5):
                out["review_reasons"].append(
                    f"low_confidence:{ch['cell']}")
    if out["review_reasons"]:
        out["verdict"] = "REVIEW"
    return out


# --------------------------------------- regression snapshots

def snapshot_records(records: dict) -> dict:
    """Everything a before/after comparison must prove unchanged:
    question text, options, answers, solution text, source pages,
    table ids/pages/cross-page continuity (and table markdowns,
    which may differ ONLY for accepted refinements)."""
    snap = {}
    for qn, rec in records.items():
        snap[str(qn)] = {
            "question_text": rec.get("question_text", ""),
            "options": {k: v for k, v in (rec.get("options") or {}).items()},
            "correct_option": rec.get("correct_option", ""),
            "solution_text": rec.get("solution_text", ""),
            "source_pages": sorted(rec.get("source_pages") or []),
            "tables": [{
                "table_id": t.get("table_id"),
                "source_pages": list(t.get("source_pages") or []),
                "merged_continuation": bool(t.get("merged_continuation")),
                "header_deduplicated": bool(t.get("header_deduplicated")),
                "markdown": t.get("markdown", ""),
            } for t in rec.get("tables") or []],
        }
    return snap


def verify_regression(before: dict, after: dict, changed_ids: set) -> dict:
    """Before/after verification over whole-chapter snapshots.
    changed_ids: the table_ids an accepted refinement was allowed to
    touch — every other byte of data must be identical."""
    problems = []
    for qn in sorted(set(before) | set(after)):
        b, a = before.get(qn), after.get(qn)
        if b is None or a is None:
            problems.append(f"q{qn} record {'missing-after' if a is None else 'missing-before'}")
            continue
        for f in ("question_text", "options", "correct_option",
                  "solution_text", "source_pages"):
            if b[f] != a[f]:
                problems.append(f"q{qn}:{f} changed")
        bt, at = b["tables"], a["tables"]
        if [t["table_id"] for t in bt] != [t["table_id"] for t in at]:
            problems.append(f"q{qn}: table id list changed")
            continue
        for tb, ta in zip(bt, at):
            for f in ("source_pages", "merged_continuation",
                      "header_deduplicated"):
                if tb[f] != ta[f]:
                    problems.append(
                        f"table {tb['table_id']}:{f} changed")
            if tb["markdown"] != ta["markdown"] \
                    and tb["table_id"] not in changed_ids:
                problems.append(
                    f"table {tb['table_id']}: markdown changed "
                    f"without an accepted refinement")
    counts = {
        "questions": len(after),
        "answers": sum(1 for r in after.values() if r["correct_option"]),
        "solutions": sum(1 for r in after.values()
                         if r["solution_text"].strip()),
        "tables": sum(len(r["tables"]) for r in after.values()),
        "cross_page_tables": sum(
            1 for r in after.values()
            for t in r["tables"] if t["merged_continuation"]),
    }
    return {"ok": not problems, "problems": problems[:10],
            "counts": counts}


# ------------------------------------------------- the stage itself

def _page_number_evidence(page_text: dict | None, pages: list) -> set | None:
    """Numeric tokens printed on the table's source pages +/-1 — the
    deterministic 'source clearly supports' evidence for a number
    repair. None when the page-text dump is absent (then no number
    repair can be accepted — safe direction)."""
    if not page_text:
        return None
    ev: set = set()
    for p in pages:
        for pp in (p - 1, p, p + 1):
            txt = page_text.get(pp, page_text.get(str(pp), ""))
            ev |= page_num_evidence(txt)
    return ev


def _ledger_append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def final_refine_table(t: dict, book, fn, only: str = "all",
                       memo: dict | None = None, page_text: dict | None = None,
                       vocab=None, ledger_key: str | None = None,
                       ledger_path: Path | None = None,
                       regions: list | None = None,
                       context: str | None = None) -> str:
    """Run the final refinement stage on ONE table record in place.
    fn: refine_final factory output (book, regions, t, md, context)
        -> {table_id, action, refined_table, changes} | None.
    only: "all" -> EVERY table is sent for refinement; "flagged"
          -> only QA-flagged tables are sent.
    Returns "skip" (no markdown) | "no_change" | "accepted" |
    "rejected" | "review" | "empty" | "invalid"."""
    md = (t.get("markdown") or "").strip()
    if not md:
        return "skip"
    val = t.setdefault("validation", {})
    qa = val.setdefault("table_qa", {})
    reasons = precheck(t, vocab)
    if only == "flagged":
        call = qa_flagged(t)
    elif only == "all":
        # all mode: EVERY table is sent for refinement — the
        # pre-check no longer gates the call (its reasons are still
        # recorded on the ledger row); a table that needs no
        # correction simply comes back as NO_CHANGE
        call = True
    else:                      # unknown mode: fail safe, no model
        call = False
    ans = None
    if call:
        key = md
        ans = memo.get(key) if memo is not None else None
        if ans is None:
            try:
                ans = fn(book, regions or [], t, md, context)
            except Exception as e:        # noqa: BLE001 - never break a run
                from . import llm as _llm
                _llm.report_error(
                    "final refinement pass failed",
                    f"{type(e).__name__}: {e}",
                    "deterministic table kept")
                ans = None
        if memo is not None:
            memo[key] = ans

    action = "not_called"
    new_md = None
    model_changes = None
    review_reasons = []
    reject_reasons = []
    cells_changed = []
    cells_checked = 0
    status = "skip"
    if call:
        if ans is None:
            status = "empty"
            action = "no_answer"
        else:
            action = str(ans.get("action") or "").strip().upper()
            model_changes = ans.get("changes")
            rt = ans.get("refined_table")
            new_md = rt.strip() if isinstance(rt, str) and rt.strip() \
                else None
            if action == "NO_CHANGE":
                status = "no_change"
            elif action == "REVIEW":
                # L3 territory, self-declared: the model itself is not
                # sure — a human decides, the table is flagged
                status = "review"
                review_reasons = ["model_action_review"]
            elif new_md is None:
                status = "invalid"
                reject_reasons = ["missing_refined_table"]
            elif new_md == md:
                status = "no_change"      # model echoed the table back
            else:
                if parse_md(new_md) is None:
                    # empty cells are LEGITIMATE in medical tables, so
                    # the gate is this module's own even-table parser,
                    # not refine.valid_rearrangement (no-empty-cells)
                    block = salvage_table(new_md)
                    if block is None or parse_md(block) is None:
                        status = "invalid"
                        reject_reasons = ["answer_not_a_table"]
                    else:
                        new_md = block
                if status != "invalid":
                    fwords = vocab[0] if vocab else None
                    new_md, nseg = restore_split_words(md, new_md, fwords)
                    if nseg:
                        val["segmentation_repairs"] = \
                            int(val.get("segmentation_repairs", 0)) + nseg
                        print(f"[refine-final] {(t.get('table_id') or '?')}: "
                              f"{nseg} word boundary repair(s) applied "
                              f"(book-vocabulary evidence)", flush=True)
                    page_nums = _page_number_evidence(
                        page_text,
                        [int(p) for p in (t.get("source_pages") or [1])])
                    cmp = fidelity_compare(md, new_md, page_nums, vocab,
                                           action, model_changes)
                    cells_changed = cmp["cells_changed"]
                    cells_checked = cmp["cells_checked"]
                    if cmp["verdict"] == "REJECT":
                        status = "rejected"
                        reject_reasons = list(cmp["reject_reasons"])
                    elif cmp["verdict"] == "REVIEW":
                        status = "review"
                        review_reasons = list(cmp["review_reasons"])
                    else:
                        status = "accepted"
    # ---- apply / flag / log ----------------------------------------
    if status == "accepted":
        val["pre_final_markdown"] = md
        val["final_refine"] = {
            "action": "REFINED",
            "gemini": True,
            "cells_checked": cells_checked,
            "changes": cells_changed,
        }
        qa["refined_final_by_gemini"] = True
        t["markdown"] = new_md
        # the REVIEW flag was judged on the pre-refinement markdown —
        # re-judge it on what actually ships (a stale REVIEW locks the
        # export gate even though the table is now repaired)
        try:
            from .refine import refresh_table_qa
            refresh_table_qa(t, new_md, vocab)
        except Exception:                      # noqa: BLE001
            pass                               # never break a run for QA
    elif status == "review":
        qa["status"] = "REVIEW"
        qa["refinement_review_reason"] = "; ".join(review_reasons)
        val["refinement_review"] = {
            "suggestion": new_md, "reasons": review_reasons,
            "changes": cells_changed or
            ([{k: c.get(k) for k in ("cell", "before", "after", "reason",
                                     "evidence", "confidence")}
              for c in (model_changes or []) if isinstance(c, dict)]),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    if status == "rejected":
        _note_reject(t, "; ".join(reject_reasons) or "fidelity")
    if ledger_path is not None and ledger_key:
        kinds = Counter(c["kind"] for c in cells_changed)
        _ledger_append(ledger_path, {
            "key": ledger_key, "table_id": t.get("table_id"),
            "status": status, "action": action,
            "precheck": reasons, "gemini_call": call,
            "cross_page": bool(t.get("merged_continuation")),
            "header_deduplicated": bool(t.get("header_deduplicated")),
            "reject_reasons": reject_reasons,
            "review_reasons": review_reasons,
            "changes": [
                {k: c.get(k) for k in
                 ("cell", "before", "after", "kind", "reason",
                  "evidence", "confidence")}
                for c in cells_changed],
            "render_qa_before": render_qa(md),
            "render_qa_after": render_qa(t.get("markdown") or md),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    return status


# --------------------------------------------------------- audit

def _iter_ledger(path: Path, subject: str) -> dict:
    """key -> latest row (a re-run rewrites a table's state)."""
    out = {}
    if not path.exists():
        return out
    prefix = subject + "|"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if str(row.get("key", "")).startswith(prefix):
            out[row["key"]] = row
    return out


def write_audit(output_root, subject: str, api_calls: int | None = None,
                regression: list | None = None) -> dict:
    """Aggregate the book's table-refinement ledger into the final
    audit report (data/table_refinement_audit.json)."""
    output_root = Path(output_root)
    rows = _iter_ledger(output_root / "data" / LEDGER, subject)
    by_status: Counter = Counter(r["status"] for r in rows.values())
    kinds: Counter = Counter()
    corrections = []
    fidelity_violations = 0
    rejected_hallucinations = rejected_deletions = 0
    rejected_number_changes = structural_rejections = 0
    cross_page_checked = 0
    render_issues = 0
    rejections = []          # every refused suggestion, with its reason
    for r in sorted(rows.values(), key=lambda x: x["key"]):
        if r.get("cross_page"):
            cross_page_checked += 1
        if r["status"] == "accepted":
            for c in r.get("changes") or []:
                kinds[c.get("kind") or "?"] += 1
                if c.get("kind") != "presentation":
                    corrections.append({
                        "table_id": r.get("table_id"),
                        "cell": c.get("cell"),
                        "before": c.get("before"),
                        "after": c.get("after"),
                        "kind": c.get("kind"),
                        "reason": c.get("reason"),
                        "evidence": c.get("evidence"),
                        "confidence": c.get("confidence"),
                    })
        if r["status"] in ("rejected", "invalid", "review"):
            # A refused suggestion is still INFORMATION: the reviewer can
            # see exactly what Gemini wanted to change and why the
            # validator said no. Without this the refusal was invisible
            # (the run printed one line and the detail was nowhere).
            for c in r.get("changes") or []:
                rejections.append({
                    "table_id": r.get("table_id"),
                    "cell": c.get("cell"),
                    "before": c.get("before"),
                    "after": c.get("after"),
                    "kind": c.get("kind"),
                    "reason": c.get("reason"),
                    "evidence": c.get("evidence"),
                    "confidence": c.get("confidence"),
                    "verdict": r["status"],
                    "reject_reasons": list(r.get("reject_reasons") or []),
                })
            if r["status"] == "review":
                rejections.append({
                    "table_id": r.get("table_id"), "cell": None,
                    "before": None, "after": None, "kind": "review",
                    "reason": "; ".join(r.get("review_reasons") or []),
                    "evidence": None, "confidence": None,
                    "verdict": "review",
                    "reject_reasons": []})
        for why in r.get("reject_reasons") or []:
            fidelity_violations += 1
            head = why.split(":", 1)[0]
            if head == "hallucinated_addition":
                rejected_hallucinations += 1
            elif head == "deletion":
                rejected_deletions += 1
            elif head == "changed_number":
                rejected_number_changes += 1
            elif head == "structural_change":
                structural_rejections += 1
        if r.get("render_qa_after"):
            render_issues += 1
    reg_rows = regression or []
    reg_ok = all(bool(x and x.get("regression_ok")) for x in reg_rows) \
        if reg_rows else True
    reg_counts: Counter = Counter()
    for x in reg_rows:
        c = (x or {}).get("counts") or {}
        for k, v in c.items():
            reg_counts[k] += v
    report = {
        "subject": subject,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_tables": len(rows),
        "tables_unchanged": by_status.get("no_change", 0)
        + by_status.get("skip", 0),
        "tables_refined": by_status.get("accepted", 0),
        "tables_rejected": by_status.get("rejected", 0)
        + by_status.get("invalid", 0),
        "tables_review": by_status.get("review", 0),
        "tables_no_answer": by_status.get("empty", 0),
        "spacing_repairs": kinds.get("spacing", 0),
        "medical_spelling_repairs": kinds.get("word_restore", 0),
        "number_repairs": kinds.get("number_repair", 0),
        "structural_repairs": 0,     # structural edits are rejected, never
                                     # accepted — kept for report shape
        "structural_rejections": structural_rejections,
        "presentation_only_refinements": sum(
            1 for r in rows.values()
            if r["status"] == "accepted"
            and all(c.get("kind") == "presentation"
                    for c in r.get("changes") or [])
            and r.get("changes")),
        "cross_page_tables_checked": cross_page_checked,
        "fidelity_violations": fidelity_violations,
        "rejected_hallucinations": rejected_hallucinations,
        "rejected_deletions": rejected_deletions,
        "rejected_number_changes": rejected_number_changes,
        "render_page_waste_issues": render_issues,
        "gemini_api_calls": api_calls if api_calls is not None else None,
        "before_after_regression": {
            "ok": reg_ok, "chapters": len(reg_rows),
            "counts": dict(reg_counts)},
        "corrections": corrections,
        "rejected_changes": rejections,
    }
    out = output_root / "data" / AUDIT
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(out)
    return report


def load_audit(output_root, subject: str) -> dict | None:
    f = Path(output_root) / "data" / AUDIT
    if not f.exists():
        return None
    try:
        r = json.loads(f.read_text())
    except ValueError:
        return None
    return r if r.get("subject") == subject else None
