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
import os
import re
import time
from collections import Counter
from pathlib import Path

from . import config

LEDGER = "refine_ledger.jsonl"


def md_table_shape(md: str) -> int | None:
    """Column count of an even pipe-markdown table (separator row
    excluded), else None. Needs >=2 rows and >=2 columns, every row the
    same width.

    EMPTY CELLS ARE FINE. They are all over these tables (the empty
    top-left header corner, footnote rows that span one column, "not
    applicable" gaps) and the old rule refused any table containing one
    — while telling the operator "answer rejected — not an even
    pipe-markdown table", which is not what was wrong. On the real book
    that silently threw away GOOD Gemini answers (e.g. 009-T01: the
    model returned the whole sensitivity/specificity table with its
    footnote row, 4 columns x 8 rows, and the pass refused to use it).
    Only the COLUMN COUNT has to agree, never the cell contents —
    content safety is the content envelope's job (same_content)."""
    lines = [l for l in (md or "").strip().splitlines() if l.strip()]
    if len(lines) >= 2 and re.fullmatch(r"[\s|:-]+", lines[1] or " "):
        lines.pop(1)
    if len(lines) < 2:
        return None
    widths = set()
    for l in lines:
        cells = l.strip().strip("|").split("|")
        if len(cells) < 2:
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


# ------------------------------------------------------- content envelope
# The rearrange pass is the ONE place where a model's markdown ships as
# the table, and until now the only gate was "it parses as an even pipe
# table" — so a model that rewrote a cell, dropped a row, or invented a
# word shipped it. Gemini is asked to REARRANGE (headers, cell
# placement, row order, spacing); the envelope below makes that
# literal: the rearrangement may move text around, but the content —
# every letter, digit and number — has to be the same as the
# deterministic extraction. Whitespace, hyphens, bullets and other
# pure separator characters are free (those ARE the layout artifacts
# the pass exists to clean); anything else is refused loudly.

_SEPARATOR_CHARS = " \t\n\r\f\v-\u2010\u2011\u2012\u2013\u2014_*`\u2022\u00b7|"

_NUMTOK = re.compile(r"\d+(?:[.,]\d+)*")


def content_signature(md: str, casefold: bool = False) -> tuple:
    """(letter/digit stream, multiset of numeric tokens) of a pipe table.

    Pipes and separator rows are dropped (they are markdown syntax, not
    content), then every character that is not a letter/digit is
    removed — so line breaks, bullets, hyphens, list separators and
    spacing are all presentation. Row/column ORDER is deliberately not
    part of the signature: reordering is what the pass is for."""
    rows = []
    lines = [l for l in (md or "").strip().splitlines() if l.strip()]
    for i, l in enumerate(lines):
        core = l.strip().strip("|")
        if i == 1 and re.fullmatch(r"[\s|:\-]+", l.strip()):
            continue                       # the |---|---| separator row
        rows.append(core)
    body = " ".join(rows)
    stream = re.sub(r"[^0-9A-Za-z]", "", body)
    if casefold:
        stream = stream.lower()
    # a MULTISET (sorted) of characters, not a sequence: moving a cell
    # to another row/column is exactly what the pass may do, so position
    # carries no meaning here — presence and count do.
    return "".join(sorted(stream)), sorted(_NUMTOK.findall(body))


def signature_diff(before: str, after: str) -> dict:
    """What changed between two signatures: missing/added characters and
    numbers, so a rejection can say exactly what the model did."""
    sb, nb = content_signature(before)
    sa, na = content_signature(after)
    # case-insensitive comparison has to fold BEFORE sorting (folding
    # a sorted stream would leave the two orders different)
    out = {"added": sorted(Counter(sa) - Counter(sb)),
           "removed": sorted(Counter(sb) - Counter(sa)),
           "numbers_added": sorted(Counter(na) - Counter(nb)),
           "numbers_removed": sorted(Counter(nb) - Counter(na)),
           "case_only": (content_signature(before, True)
                         == content_signature(after, True))}
    out["summary"] = "; ".join(
        f"{k}={v}" for k, v in (("added", "".join(out["added"])[:40]),
                                ("removed", "".join(out["removed"])[:40]),
                                ("numbers_added", ",".join(out["numbers_added"][:5])),
                                ("numbers_removed", ",".join(out["numbers_removed"][:5])))
        if v)
    return out


# ------------------------------------------------- word segmentation
# The letter/digit signature above is deliberately blind to WHITESPACE
# — but word boundaries are content in a medical text. Live corruption
# it let through (016-T01, printed "incompletely"): Gemini rearranged
# the cell and returned "in completely immunized", the signature saw
# the same letters and shipped it, and the meaning changed. The
# decision "is 'in completely' or 'incompletely' the printed form?"
# cannot be answered from the model's word (it can always glue or
# split) but it CAN be answered from the book itself: the vocabulary
# counts every word the PDF's text layer prints, so the segmentation
# the book actually uses is the one with the STRONGER evidence.
#
# Rule (document-internal, no external knowledge): compare the two
# segmentations of the same letter run by their weakest link
# (min(count+1)); the stronger one wins, ties keep the original. That
# heals artifacts in both directions — gluing ("sensitivitytesting"
# -> "sensitivity testing") and bogus splitting ("in completely" ->
# "incompletely") — and it never invents text, because the letter
# stream may not change.

_PHRASE_WORD = re.compile(r"[A-Za-z]{2,}")

# the project's function-word list (tables._FUNC): a welded FUNCTION word
# is the classic stolen mid-line fragment ("he matogenous"), a welded
# CONTENT+CONTENT pair ("middle ear") is the invented glue we refuse
from .tables import _FUNC                      # noqa: E402


def _split_vocab(vocab):
    """(words, pairs) from whatever the caller has: tables.build_vocab's
    (words, pairs) tuple, a bare words Counter, or None."""
    if vocab is None:
        return None, None
    if isinstance(vocab, tuple):
        return (vocab[0], vocab[1] if len(vocab) > 1 else None)
    return vocab, None


def _supported(words, token: str) -> int:
    """Evidence for one word as printed (smoothed: 0 must not win)."""
    return words.get(token.lower(), 0) + 1


def _segmentation_support(words, parts) -> int:
    """Weakest-link support of a segmentation: a phrase is only as well
    evidenced as its rarest word."""
    return min((_supported(words, p) for p in parts), default=0)


def _is_shard(token: str) -> bool:
    """A 1-2 letter piece (or a function word) welded into a longer
    word is the text layer's stolen-fragment signature — the join that
    mends it must not be refused ("he matogenous" -> "hematogenous").
    Three letters is already a real word far too often ("ear", "eye")
    for that allowance, and such a part only needs to be a PRINTED word
    (>=2x) to keep the refusal (see the caller)."""
    return len(token) <= 2 or token.lower() in _FUNC


def _phrases(text: str):
    """[(start, end, [tokens])] — runs of words separated by whitespace
    ONLY. A cell boundary, comma or slash ends a phrase, so a decision
    is never made across punctuation the model did not touch."""
    spans, cur = [], []
    for m in _PHRASE_WORD.finditer(text or ""):
        if cur and text[cur[-1].end():m.start()].strip() == "":
            cur.append(m)
            continue
        if cur:
            spans.append(cur)
        cur = [m]
    if cur:
        spans.append(cur)
    return [(ms[0].start(), ms[-1].end(), [m.group(0) for m in ms])
            for ms in spans]


def _partition(single: list, multi: list):
    """[(single_token, [parts])] — `single` and `multi` hold the same
    letters in the same order, `multi` has more tokens: each single
    token is the concatenation of consecutive multi tokens."""
    groups, i = [], 0
    for tok in single:
        need, parts = len(tok), []
        while need > 0 and i < len(multi):
            part = multi[i]
            i += 1
            parts.append(part)
            need -= len(part)
        groups.append((tok, parts))
    return groups


_MARKUP = re.compile(r"<br\s*/?>|<[A-Za-z/][^>]{0,12}>")


def word_runs(text: str) -> list:
    """The words of a text in reading order (letter runs, >=2 chars).
    Markup is stripped first: a "<br>" is presentation and its letters
    ("br") must never enter a word comparison (measured: it showed up
    as a phantom split, "wo -> br two", on the real book)."""
    return _PHRASE_WORD.findall(_MARKUP.sub(" ", text or ""))


def restore_split_words(before: str, after: str, words) -> tuple:
    """(answer, repairs) — put back every word the ANSWER split apart
    that the extraction prints whole.

    The extraction is the authority for spelling; rearranging may move
    text but not re-cut it. Live case (016-T01): the extraction prints
    "incompletely immunized", Gemini's rearrangement returned "in
    completely immunized" — identical letters, so the character
    envelope saw nothing, and the meaning changed. Repairing is safer
    than rejecting: the model's real fixes in the same answer (glued
    "phos phate" -> "phosphate") are then kept instead of thrown away
    with the corruption.

    Only a run of NEW words whose concatenation is a word the
    extraction prints WHOLE (and that the answer does not) is put back
    — the letters are identical by construction, so nothing is
    invented, and a repair the model made in the other direction (a
    split extraction joined back) is never undone."""
    words, _ = _split_vocab(words)
    if not words or not before or not after:
        return after, 0
    src = {w.lower() for w in word_runs(before)}
    if not src:
        return after, 0
    out, n, last = [], 0, 0
    for start, end, toks in _phrases(after):
        fixed, k = _restore_phrase(toks, src, words)
        if not k:
            continue
        out.append(after[last:start])
        out.append(" ".join(fixed))
        last, n = end, n + k
    if not n:
        return after, 0
    out.append(after[last:])
    return "".join(out), n


def _restore_phrase(tokens: list, src: set, words) -> tuple:
    """Put back the whole words of ONE phrase (see restore_split_words)."""
    out, n, i = [], 0, 0
    while i < len(tokens):
        hit = None
        for k in (3, 2):                    # longest run first
            if i + k > len(tokens):
                continue
            cand = "".join(tokens[i:i + k])
            # the joins must not swallow a printed word ("Weak" + "+"?)
            if (cand.lower() in src and words.get(cand.lower(), 0) >= 2
                    and all(words.get(t.lower(), 0) >= 1
                            for t in tokens[i:i + k])):
                hit = (k, cand)
                break
        if hit:
            out.append(hit[1])
            i += hit[0]
            n += 1
            continue
        out.append(tokens[i])
        i += 1
    return out, n


def segmentation_weakens(before: str, after: str, words) -> tuple:
    """(weakened?, summary) — True when the answer broke ONE printed
    word into a function-word + word the book does not print that way.

    This is deliberately the only class it judges: the live corruption
    (016-T01) turned the printed "incompletely immunized" into "in
    completely immunized" — identical letters, different meaning, and
    the character-identity envelope could not see it.

    Anything else is left alone, and that restraint is measured, not
    guessed:
      * joins ("he matogenous" -> "hematogenous", "Pox virus" ->
        "Poxvirus", "pred nisolone" -> "prednisolone") are REPAIRS of
        the text layer's stolen fragments; a vocabulary that counts the
        broken lines themselves cannot forbid them (it "knows" the
        shard "matogenous" — printed twice, both times from the very
        line that is wrong);
      * a many-token boundary rewrite (a wrapped name list re-cut cell
        by cell, 028-T02) partitions into groups this heuristic cannot
        judge honestly — refusing it threw away a page of correct
        spacing repairs, and the blob "jeanselmeiPhialop" counts as an
        "established word" precisely because it is printed twice.
    The restrictive classes above are already covered: the character
    envelope refuses letter changes, and restore_split_words() puts the
    extraction's own words back."""
    words, pairs = _split_vocab(words)
    if not words:
        return False, ""
    import difflib
    wb, wa = word_runs(before), word_runs(after)
    if [w.lower() for w in wb] == [w.lower() for w in wa]:
        return False, ""
    bad = []
    sm = difflib.SequenceMatcher(a=[w.lower() for w in wb],
                                 b=[w.lower() for w in wa], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal" or (i2 - i1) != 1 or (j2 - j1) != 2:
            continue                      # only a clean 1 -> 2 split
        tok, (p1, p2) = wb[i1], (wa[j1], wa[j1 + 1])
        if words.get(tok.lower(), 0) < 2:
            continue                      # not an established word
        if tok[1:] != tok[1:].lower():
            continue                      # "jeanselmeiPhialop": a glue
            #                                 artifact, not a word
        bad.append(f"{tok} -> {p1} {p2}")
    if not bad:
        return False, ""
    return True, "; ".join(bad[:3])


def envelope_mode() -> str:
    """`strict` (default): the rearrangement may move/re-space, never
    change a letter or a digit. `off`: the old behaviour — any even
    table shape is accepted, so the model may also ADD text (e.g. an
    invented header name). Strict is the default because an invented
    header is a hallucination in a medical study app; turn it off with
    QBANK_REARRANGE_ENVELOPE=off only if you accept that."""
    return (os.environ.get("QBANK_REARRANGE_ENVELOPE")
            or "strict").strip().lower()


def same_content(before: str, after: str, words=None) -> tuple:
    """(True, diff) when the model only moved/re-spaced the table;
    (False, diff) when a letter or digit changed — or when `words` (the
    book vocabulary) shows the answer split/joined a word the printed
    book separates the other way. `off` short-circuits this to True
    (see envelope_mode)."""
    if envelope_mode() in ("off", "0", "none"):
        return True, {}
    sb, nb = content_signature(before)
    sa, na = content_signature(after)
    if (sb, nb) == (sa, na):
        weak, how = segmentation_weakens(before, after, words) \
            if words is not None else (False, "")
        if weak:
            diff = signature_diff(before, after)
            diff["segmentation"] = how
            diff["summary"] = (f"{diff.get('summary') or ''}"
                               f"{'; ' if diff.get('summary') else ''}"
                               f"segmentation={how}")
            return False, diff
        return True, {}
    diff = signature_diff(before, after)
    if diff["case_only"] and nb == na:
        return True, diff                    # capitalisation only
    return False, diff


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
    global _content_samples_left
    _reject_samples_left = REJECT_SAMPLES
    _salvage_samples_left = SALVAGE_SAMPLES
    _content_samples_left = CONTENT_REJECT_SAMPLES


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
    print(f"[gemini] {(t.get('table_id') or '?')}: answer rejected — it is "
          f"not a pipe table with one consistent column count, so the "
          f"deterministic extraction is kept. Answer starts: {head!r}",
          flush=True)


CONTENT_REJECT_SAMPLES = 3
_content_samples_left = CONTENT_REJECT_SAMPLES


def _note_content_reject(t: dict, diff: dict) -> None:
    """A rearrangement that changed printed characters is refused — and
    the log says exactly what it tried to change (never silently)."""
    global _content_samples_left
    if _content_samples_left <= 0:
        return
    _content_samples_left -= 1
    print(f"[gemini] {(t.get('table_id') or '?')}: rearrangement REJECTED "
          f"— it changed printed content ({diff.get('summary') or 'differs'}"
          f") — the deterministic extraction is kept", flush=True)


def _md_matrix(markdown: str) -> list:
    """Cell matrix of a pipe table (separator rows and blanks dropped).
    Kept local so the QA refresh never depends on the model layer."""
    rows = []
    for line in (markdown or "").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if set(line) <= set("|-: "):           # |---|---| separator
            continue
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def refresh_table_qa(t: dict, markdown: str, vocab) -> bool | None:
    """Re-judge this table's REVIEW flag on the markdown that will ship.

    The flag is computed at extraction time, on the DETERMINISTIC
    markdown. Once a refinement is ACCEPTED the old verdict can be
    stale, and a stale REVIEW does not just look wrong — it locks the
    export gate for good. Live case (010-T02): the accepted
    rearrangement shipped "drug-sensitivity testing" and "BacT
    Mycobacteria growth indicator tube (MGIT)" while table_qa still
    said REVIEW for 'sensitivitytesting' and 'BacTMycobacteria', so the
    human queue never opened and the zip stayed locked.

    The re-judgement is the SAME deterministic, vocabulary-based
    detector (tables.qa_suspects) applied to the shipped cells — not a
    rubber stamp: a refinement that glues something scores REVIEW
    again. Returns True when the flag cleared, False when it stays,
    None when there is no vocabulary to judge with.
    """
    if not vocab or not (markdown or "").strip():
        return None
    cells = _md_matrix(markdown)
    if not cells:
        return None
    from .tables import qa_suspects
    found = sorted(set(qa_suspects(cells, vocab[0], vocab[1])))
    val = t.setdefault("validation", {})
    qa = val.setdefault("table_qa", {})
    before = list(qa.get("suspect_fragments") or [])
    qa["suspect_fragments"] = found[:12]
    qa["status"] = "REVIEW" if found else "ok"
    qa["rechecked_after_refinement"] = True
    if before and not found:
        qa["suspect_fragments_before"] = before[:12]
        qa["review_cleared_by_refinement"] = True
        print(f"[gemini] {(t.get('table_id') or '?')}: the accepted "
              f"refinement cleared the table's REVIEW flag "
              f"(was: {', '.join(before[:4])})", flush=True)
    warns = val.get("warnings")
    if isinstance(warns, list):        # drop lost-space warnings whose
        low = markdown.lower()         # token is no longer in the text
        val["warnings"] = [
            w for w in warns
            if not (str(w).startswith("suspect_lost_space:")
                    and str(w).split(":", 1)[1].lower() not in low)]
    return not found


def _vocab_for(book):
    """The book vocabulary, or None when it cannot be built (tests pass
    book=None; a failed build must never break a run)."""
    try:
        from .tables import build_vocab
        return build_vocab(book) if book is not None else None
    except Exception:                          # noqa: BLE001
        return None


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
    # WORD BOUNDARIES FIRST: the model's answer may re-cut a word the
    # extraction prints whole ("in completely" for "incompletely").
    # That is repaired here, BEFORE the envelope, so a good
    # rearrangement is not thrown away for one artifact it can fix.
    vocab = _vocab_for(book)
    new, nseg = restore_split_words(md, new, vocab)
    if nseg:
        val0 = t.setdefault("validation", {})
        val0["segmentation_repairs"] = nseg
        print(f"[gemini] {(t.get('table_id') or '?')}: {nseg} word "
              f"boundary repair(s) in the rearrangement (book-vocabulary "
              f"evidence)", flush=True)
    # CONTENT ENVELOPE: reordering and re-spacing are the model's job,
    # changing a letter or a number is not. Until this check existed a
    # hallucinated cell shipped as the table (see the module docstring:
    # the extraction is the only authority for characters).
    ok_content, diff = same_content(md, new, vocab)
    if not ok_content:
        _note_content_reject(t, diff)
        val = t.setdefault("validation", {})
        val["gemini_rearrange"] = {
            "status": "REJECTED_CONTENT_CHANGED",
            "reason": "the rearrangement changed printed characters "
                      "(only rearrangement/spacing is allowed)",
            "diff": {k: v for k, v in diff.items() if k != "case_only"},
            "kept": "deterministic_extraction",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        return "content_changed"
    val = t.setdefault("validation", {})
    val.setdefault("pre_gemini_markdown", md)
    qa = val.setdefault("table_qa", {})
    qa["refined_by_gemini"] = True
    if diff:                       # accepted, but the model re-cased it
        val["gemini_rearrange"] = {
            "status": "ACCEPTED_CASE_ONLY",
            "note": "capitalisation changed; letters/digits identical",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if salvaged:
        qa["salvaged_from_wrapper"] = True   # provenance: it came chatty
    t["markdown"] = new.strip()
    # the extraction-time REVIEW flag was judged on the OLD markdown:
    # re-judge it now, or a repaired table keeps demanding a human
    refresh_table_qa(t, new.strip(), _vocab_for(book))
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
