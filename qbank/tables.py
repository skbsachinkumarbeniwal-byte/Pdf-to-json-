"""
Ruled-table reconstruction: geometry -> cells -> logical tables.

Pipeline (this module replaces "flattened text -> regex -> markdown"):

    ruled box (vector rules)
      -> column edges (vertical rules) and row bands (horizontal rules)
      -> lines assigned to (band, column) cells by bbox geometry
      -> cross-line word reconstruction INSIDE a cell, decided by
         layout evidence only:
           * previous line fills the column's measured fill edge
             (max x1 of the column = the typesetter's text extent)
             AND next line starts lowercase  -> join WITHOUT space
             ("medial su" + "rface" = "medial surface",
              "C2,C" + "3" = "C2,C3",  "Grad" + "e 1" = "Grade 1")
           * trailing hyphen                 -> join, hyphen kept
           * anything else                   -> join WITH a space
             ("middle" + "ear" = "middle ear")
      -> camel-boundary space repair for text-layer lost spaces
         ("antihelixSome" -> "antihelix Some"; guarded so pH, IgG,
         mOsm, B12 are never split)
      -> glyph repair (shared frozen rule table)
      -> cross-page continuation merge (same column geometry on the
         next page + repeated header, or last-box-on-page ->
         first-box-on-next-page flow) into ONE logical table
      -> pipe-markdown + validation

Never a blanket whitespace/regex normalisation: every join is decided
per line pair from coordinates and counted for audit.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from . import glyphs
from .llm import merge_llm

# camel boundary: >=2 lowercase, then an uppercase that starts a
# lowercase run. Never splits pH / IgG / mOsm / B12 / VLDL.
_CAMEL = re.compile(r"(?<=[a-z]{2})(?=[A-Z][a-z])")

# a token ending in ONE capital after a lowercase run: the capital opens a
# new item ("griseaE" -> "grisea E"). The capital must be followed by a
# non-letter, or established lowercase-prefix acronyms ("cccDNA",
# "dsDNA", "ssRNA" — capital RUNS) would be torn apart.
_CAMEL_TAIL = re.compile(r"(?<=[a-z]{3})(?=[A-Z](?![A-Za-z]))")
_LONG_TOKEN = re.compile(r"[A-Za-z]{12,}")
_ALPHA = re.compile(r"[A-Za-z]+")
# printed token that MIXES letters and digits (CD4, STAT3, UL97, NOD1,
# SA14-14-2). The leading letter is required so pure numbers stay out:
# a bare "58" is a number, not a medical token.
_ALNUM = re.compile(r"[A-Za-z][A-Za-z0-9+\-]*")


def build_vocab(book) -> tuple:
    """Book-wide evidence: lowercase word counts + adjacent-word pair
    counts from the raw layer (visual rows). The space repairs below
    only fire when the book itself prints the other form elsewhere —
    document-internal evidence, no external knowledge, no blanket
    whitespace regex.

    Memoised on the book object: the scan touches every page, and the
    refinement stages re-ask for the same vocabulary after extraction
    already built it once."""
    cached = getattr(book, "_qbank_vocab", None)
    if cached is not None:
        return cached
    from .textlayer import word_rows
    words: Counter = Counter()
    pairs: Counter = Counter()
    # ALPHANUMERIC tokens are a SECOND vocabulary. _ALPHA is letters-only,
    # so before this Counter existed a printed `CD4` left no evidence
    # anywhere and the medical-token repair had nothing to stand on.
    mixed: Counter = Counter()
    items: Counter = Counter()                 # camel-boundary sub-runs
    for pg in range(1, book.total_pages + 1):
        for wr in word_rows(book.page(pg)):
            # strip edge punctuation first: "count," / "none." are the
            # only occurrence of many real words in these books
            toks = [m.group(0) for w in wr
                    if (m := _ALPHA.fullmatch(
                        w.text.strip(".,;:!?()[]{}\"'"))) is not None]
            for t in toks:
                words[t.lower()] += 1
            for a, b in zip(toks, toks[1:]):
                pairs[(a.lower(), b.lower())] += 1
            for w in wr:
                m = _ALNUM.fullmatch(w.text.strip(".,;:!?()[]{}\"'"))
                if m is not None and any(ch.isdigit() for ch in m.group(0)):
                    mixed[m.group(0).lower()] += 1
                # CAMEL SUB-ITEMS: this book glues list items and species
                # names together ("richardsiaeBipolaris", "coliCryptospo",
                # "aCladophialophora"). A capital inside a printed token
                # therefore marks a printed ITEM START, and the sub-run
                # (`Bipolaris`) is a whole printed item even though the
                # vocabulary has no standalone occurrence of it.
                a = w.text.strip(".,;:!?()[]{}\"'")
                if re.fullmatch(r"[A-Za-z]{4,}", a) and re.search(r"[a-z][A-Z]", a):
                    pieces = re.split(r"(?<=[a-z])(?=[A-Z])", a)
                    if len(pieces) > 1:
                        for piece in pieces:
                            if len(piece) >= 3:
                                items[piece.lower()] += 1
    try:
        book._qbank_vocab = (words, pairs)      # reuse across phases
        book._qbank_mixed = mixed
        book._qbank_items = items
    except Exception:                           # noqa: BLE001
        pass                                    # read-only book: recompute
    return words, pairs


# terms visually verified against the PDF render as single legitimate
# words that the raw vocabulary prints only once (qa must not flag them)
_LEGIT = frozenset({"antihelix"})

# spacing corrections confirmed by visual inspection of the PDF
# render (the raw layer has zero internal evidence for these): applied
# to table cells only, exact-match, counted for audit
_VERIFIED_MERGES = {
    "osteocal cin": "osteocalcin",
    "bonedestruct ion": "bone destruction",
    "fossaororbital": "fossa or orbital",
    "fossaor": "fossa or",
    "theinfrat emporal": "the infratemporal",
    "oroptic": "or optic",
    "involvem ent": "involvement",
    "regionwithintracranialextradural":
        "region with intracranial extradural",
    "suprastruct ures": "suprastructures",
    "adja cent": "adjacent",
    "groo ve": "groove",
    "do me": "dome",
    "destr oying": "destroying",
    "pterygo palatine": "pterygopalatine",
    "swi m": "swim",
    "im paired": "impaired",
    "be yond": "beyond",
    # ENT 017-T01: wrap splits of words the book prints joined
    # ("restless" p282/463, "throughout" p280/445; split spacing
    # printed nowhere) — both parts are common words, so the
    # evidence rules must stay silent to protect "brain stem"
    "rest less": "restless",
    "through out": "throughout",
}

_FUNC = frozenset({"the", "not", "of", "a", "an", "in", "on", "at", "is",
                   "or", "and", "to", "for", "with", "per", "by", "has",
                   "but", "can", "be", "are", "was", "were", "it", "as",
                   "into", "from", "than", "may", "no", "so", "if"})


# Function words that may drive a LOST-SPACE split inside a token. The
# four words below are refused because they are also Latin/anatomical
# ENDINGS: the measurement over both books' vocabularies (every once-
# printed token split at a glue boundary with both halves established
# and the spaced pair printed twice or more) listed "arteryin",
# "layersof", "partof", "thecell", "themiddle", "proximalto",
# "alongwith", "atleast", "thesame" — all true artifacts — plus exactly
# ONE false positive, "inferioris" ("Depressor labii inferioris", a real
# Latin term) from "is". "or"/"as"/"it" are refused for the same class
# (Latin -or, -as, -it: levator, fetalis/recurrentis species names) and
# buy no repair in either book.
_LATIN_TAILS = frozenset({"is", "or", "as", "it"})
_GLUE_WORDS = frozenset(w for w in _FUNC if w not in _LATIN_TAILS)


def _split_glue_token(tok: str, words, pairs):
    """A function word glued to a common word, printed at most ONCE.

    "partof", "alongwith", "thesame", "layersof", "proximalto" are
    lost spaces the typesetter left in the print itself: the glued form
    appears exactly once in the whole book (it IS the artifact), while
    both halves are established words and the SPACED pair is a
    recurring print. Nothing else qualifies — a form the book prints
    twice is a word of the book, not a misprint.

    Returns the re-spaced token, or None. Whitespace is the only thing
    that ever changes."""
    lo = tok.lower()
    if not tok.isalpha() or len(lo) < 6 or words.get(lo, 0) > 1:
        return None
    for i in range(2, len(lo) - 1):
        h, t = lo[:i], lo[i:]
        if (h in _GLUE_WORDS or t in _GLUE_WORDS) \
                and words.get(h, 0) >= 2 and words.get(t, 0) >= 2 \
                and pairs.get((h, t), 0) >= 2:
            return f"{tok[:i]} {tok[i:]}"
    return None


def split_glue_words(text: str, words, pairs) -> str:
    """PROSE-side of _split_glue_token: re-space the glued function
    words a paragraph contains. The prose path never had a token-level
    repair, so a printed "leucocytosis alongwith microscopy" shipped
    as-is (MIC-014-020). Whitespace only."""
    if not text or not words:
        return text
    return re.sub(r"[A-Za-z]{6,}",
                  lambda m: _split_glue_token(m.group(0), words, pairs)
                  or m.group(0), text)


_ORD_TAIL = r"(?:st|nd|rd|th)"


def spacing_fix(text: str, mixed=None, vocab=None) -> str:
    """Deterministic spacing fixes for the glue shapes the text layer
    leaves behind (reviewer-directed; purely shape-based, no vocab):

      "1strib"      -> "1st rib"       ordinal glued onto a word
      "the1st"      -> "the 1st"
      "vertebraD4"  -> "vertebra D4"   vertebral notation (C/D/T/L/S +
      "T3nerve"     -> "T3 nerve"       1-2 digits) never mixes with a
                                        word
      "nerve(T3)"   -> "nerve (T3)"    one space before "(" and
      "(T3)is"      -> "(T3) is"       after ")"
      "following:Anterior" -> "following: Anterior"
                                       one space after ":" too
      "membrane.It"   -> "membrane. It"
      "( parasellar )"-> "(parasellar)"
      "the   patient" -> "the patient"
      "(EAC) ."     -> "(EAC)."  /  "below :" -> "below:"

    Safe: bare "D4"/"T3" untouched; "HbA1c", "C2H5OH", "vitamin B12"
    are never split (A/B are not in the notation set; an uppercase
    letter after digits is never separated)."""
    out = text
    # ordinal glued onto the following word ("1strib", "12thrib")
    out = re.sub(rf"(?<=\d){_ORD_TAIL}(?=[A-Za-z])",
                 lambda m: m.group(0) + " ", out)
    # word glued onto an ordinal ("the1st", "Chapter4th")
    out = re.sub(rf"(?<=[A-Za-z])(?=\d{{1,2}}{_ORD_TAIL}(?![0-9A-Za-z]))",
                 " ", out)
    # word + vertebral notation ("vertebraD4", "inT3")
    out = re.sub(r"(?<=[A-Za-z])(?=[CDTLS]\d{1,2}(?![0-9A-Za-z]))",
                 " ", out)
    # vertebral notation + lowercase word ("D4vertebra", "T3nerve")
    out = re.sub(r"(?<=[CDTLS]\d)(?=[a-z])", " ", out)
    # no space BEFORE sentence punctuation: reflow artifacts like
    # "(EAC) ." or "given below :" — the source prints "(EAC)." and
    # "below:" (verified against the ENT text layer)
    out = re.sub(r"(?<=[^\s]) +([.,;:?!])", r"\1", out)
    # colon: exactly one space after it — but never inside digit
    # ratios/times ("1:1000" adrenaline, "10:30") or URLs ("http://")
    out = re.sub(r"(?<!\d): *(?=[^\s:/])", ": ", out)
    # NOTE: no comma/semicolon space insertion here — the ENT source
    # itself prints compact notations like "(C2,C3)", so inserting a
    # space is NOT source-backed (source-fidelity rule). Table cells
    # keep the evidence-based comma split in _repair_token.
    # sentence collision ("membrane.It") — a lowercase letter before
    # the stop and an uppercase start after: unambiguous new sentence;
    # initials ("J.K.Rowling") and versions ("v1.2Beta") untouched
    out = re.sub(r"(?<=[a-z])[.?!](?=[A-Z])",
                 lambda m: m.group(0) + " ", out)
    # brackets: exactly one space before "(" and after ")"
    out = re.sub(r"(?<=[^\s(]) *\(", " (", out)
    # space after ")" only before word characters — never before
    # punctuation ("(EAC)." stays glued, source-true)
    out = re.sub(r"\) *(?=[A-Za-z0-9(])", ") ", out)
    # no padding immediately INSIDE brackets ("( parasellar )")
    out = re.sub(r"\(\s+", "(", out)
    out = re.sub(r"\s+\)", ")", out)
    # accidental multiple spaces collapse to one (prose only in
    # practice: table cells arrive here as single-space tokens)
    out = re.sub(r" {2,}", " ", out)
    # lost-space glue in PROSE ("alongwith" -> "along with"): needs the
    # book's own word + pair counts, so it only runs when the caller
    # hands them over
    if vocab:
        out = split_glue_words(out, vocab[0], vocab[1])
    # medical-token joins LAST: the repairs above only move spaces
    # around, and this one needs the book's alphanumeric vocabulary
    if mixed:
        out, _joins = join_medical_tokens(out, mixed)
    return out


def _repair_token(tok: str, words: Counter, pairs: Counter,
                  whole: set | None = None) -> tuple:
    """Publisher misprints inside ONE token, repaired only with
    book-internal evidence:

      "damage,fetal"       -> "damage, fetal"  (comma glued: both parts
                                are printed words, glued form is not)
      "Increasedpulmonary" -> "Increased pulmonary" (two common words,
                                glued form never/rarely printed, spaced
                                phrase printed elsewhere)

    `whole` is the set of forms that must NOT be split: tokens the join
    pass built from the printed geometry, and camel sub-items the book
    prints (see camel_items). Splitting those undoes the print."""
    if whole and tok.lower() in whole:
        return tok, 0
    hit = _split_glue_token(tok, words, pairs)
    if hit:
        return hit, 1
    m = re.fullmatch(r"([A-Za-z]{3,}),([A-Za-z]{3,})", tok)
    if (m and words.get(m.group(1).lower(), 0) >= 1
            and words.get(m.group(2).lower(), 0) >= 1
            and words.get((m.group(1) + m.group(2)).lower(), 0) == 0):
        return f"{m.group(1)}, {m.group(2)}", 1
    # glued pair whose spaced form the book also prints ("andhas"):
    # both parts must be common words and the pair must recur
    if (tok.isalpha() and 6 <= len(tok) < 8
            and words.get(tok.lower(), 0) == 0):
        for i in range(3, len(tok) - 2):
            h, t = tok[:i], tok[i:]
            if (words.get(h.lower(), 0) >= 5 and words.get(t.lower(), 0) >= 5
                    and pairs.get((h.lower(), t.lower()), 0) >= 2):
                return f"{h} {t}", 1
    if tok.isalpha() and len(tok) >= 8 and words.get(tok.lower(), 0) == 0:
        for i in range(3, len(tok) - 2):
            h, t = tok[:i], tok[i:]
            if (words.get(h.lower(), 0) >= 2 and words.get(t.lower(), 0) >= 2
                    and pairs.get((h.lower(), t.lower()), 0) >= 1):
                return f"{h} {t}", 1
    return tok, 0


_DIGIT_FRAG = re.compile(r"\d")


def _repair_tokens(parts: list, words: Counter, pairs: Counter,
                   whole: set | None = None) -> tuple:
    """Token-stream repair for one cell line (see _repair_token), plus:

      "o fcancer" -> "of cancer"  (short non-word fragment whose head
                                   completes a common word)
      "Theprobability" -> "The probability" / "notdepend" -> "not depend"
                   (function prefix + common remainder; glued form
                    never printed as a real word elsewhere)
      "following:Anterior" -> "following: Anterior"  (colon glue)
      "3 00" -> "300", "50 – 300" -> "50–300"  (number fragments)
      "tiss ues" -> "tissues"  (wrapped fragments whose join is a
                   book word; neither fragment is one)
      "retractionnot" -> "retraction not"  (misprint printed <=2x whose
                   parts are both common book words)"""
    res, nfix = [], 0
    i = 0
    while i < len(parts):
        tok = parts[i]
        # a form the print proves whole (a glued seam or a camel sub-item)
        # is never split: "Bipolaris" is a printed ITEM that only happens
        # to be glued to the species before it, and splitting it into
        # "Bipolar is" corrupted a correct word.
        m = re.fullmatch(r"[A-Za-z]+", tok or "")
        if whole and m and tok.lower() in whole:
            res.append(tok)
            i += 1
            continue
        m = re.fullmatch(r"([A-Za-z]{3,})([:;])([A-Za-z]{3,})", tok)
        if (m and words.get(tok.lower(), 0) == 0
                and words.get(m.group(1).lower(), 0) >= 2
                and (words.get(m.group(3).lower(), 0) >= 1
                     or len(m.group(3)) >= 8)):
            res.append(f"{m.group(1)}{m.group(2)} {m.group(3)}")
            nfix += 1
            i += 1
            continue
        if (tok.isdigit() and i + 1 < len(parts)
                and parts[i + 1].isdigit() and parts[i + 1][:1] == "0"):
            res.append(tok + parts[i + 1])
            nfix += 1
            i += 2
            continue
        if (i + 1 < len(parts) and tok.isalpha() and 1 <= len(tok) <= 2
                and words.get(tok.lower(), 0) <= 3
                and words.get(parts[i + 1].lower(), 0) <= 1):
            nxt = parts[i + 1]
            hit = None
            for k in range(1, min(3, len(nxt) - 2)):
                if (words.get((tok + nxt[:k]).lower(), 0) >= 2
                        and words.get(nxt[k:].lower(), 0) >= 2):
                    hit = k
                    break
            if hit:
                res.append(tok + nxt[:hit])
                res.append(nxt[hit:])
                nfix += 1
                i += 2
                continue
        core, punct = tok, ""
        while core and core[-1] in ",.;:!?)]":
            punct = core[-1] + punct
            core = core[:-1]
        fixed, nf = _repair_token(core, words, pairs, whole)
        # function word glued onto a common word, printed <=2x
        # ("oftouch" -> "of touch", "tomotor" -> "to motor",
        # "ofinternal" -> "of internal"): both survivors are strongly
        # evidenced in the book, the glued form is not. 2-letter
        # function prefixes ("of", "to") are below the generic peel's
        # 3-letter cut, hence this dedicated rule.
        _SUFFIX = ("able", "ible", "ance", "ence", "tion", "ment",
                   "ness", "ous", "ive", "ful", "less", "ity", "ies")
        # two function words glued ("tothe" -> "to the"): neither
        # survives the >=3-letter cuts of the generic peels
        if (nf == 0 and core.isalpha() and len(core) >= 4
                and words.get(core.lower(), 0) <= 2):
            for f in _FUNC:
                t2 = core[len(f):]
                if (core[:len(f)].lower() == f and len(f) >= 2
                        and len(t2) >= 2 and t2.lower() in _FUNC):
                    fixed, nf = f"{core[:len(f)]} {t2}", 1
                    break
        if (nf == 0 and core.isalpha() and len(core) >= 6
                and words.get(core.lower(), 0) <= 2):
            for f in _FUNC:
                t2 = core[len(f):]
                # t2 must not be a productive suffix word ("notable"
                # = not+able is a real word; "tomotor" is not)
                if (core[:len(f)].lower() == f and len(t2) >= 4
                        and t2[0].islower() and t2.lower() not in _SUFFIX
                        and words.get(t2.lower(), 0) >= 5):
                    fixed, nf = f"{core[:len(f)]} {t2}", 1
                    break
            if nf == 0:
                # SUFFIX splits must use a _GLUE_WORD: with plain _FUNC
                # this loop rewrote printed terms — "inferioris" ->
                # "inferior is" (Depressor labii inferioris), and it
                # would have corrupted "inhibitor" -> "inhibit or",
                # "vaginalis"/"recurrentis"/"fetalis" (species names)
                # the moment such a token reached a cell.
                for f in sorted(_GLUE_WORDS):
                    h = core[:-len(f)]
                    if (core[-len(f):].lower() == f and len(h) >= 5
                            and words.get(h.lower(), 0) >= 5):
                        fixed, nf = f"{h} {core[-len(f):]}", 1
                        break

        # prefix + unknown remainder ("intosuprastructures"): the
        # prefix is a common book word, the remainder is printed
        # nowhere on its own (so not a real short word being glued).
        # Runs BEFORE generic peeling so "into" wins over a random cut.
        if (nf == 0 and core.isalpha() and len(core) >= 10
                and words.get(core.lower(), 0) <= 1):
            for p in ("into", "onto", "within", "without", "after",
                      "before", "over", "under"):
                t2 = core[len(p):]
                if (core[:len(p)].lower() == p
                        and words.get(p, 0) >= 10 and len(t2) >= 6
                        and t2[0].islower()
                        and words.get(t2.lower(), 0) == 0):
                    fixed, nf = f"{core[:len(p)]} {t2}", 1
                    break
        if (nf == 0 and core.isalpha() and len(core) >= 8
                and words.get(core.lower(), 0) <= 1):
            for j in range(3, len(core) - 1):
                h, t2 = core[:j], core[j:]
                # a lowercase token the book actually prints, splittable
                # into two common lowercase words, is a rare real word
                # ("everywhere"), not a glued artifact — never peel it
                if (core[0].islower() and words.get(core.lower(), 0) >= 1
                        and h[0].islower() and t2[0].islower()
                        and words.get(h.lower(), 0) >= 2
                        and words.get(t2.lower(), 0) >= 2):
                    continue
                thr = 1 if h.lower() in _FUNC else 2
                # recursive peel: the head may itself be a glued blob
                # (w==0) that later passes split further
                h_blob = len(h) >= 8 and words.get(h.lower(), 0) == 0
                h_ok = (words.get(h.lower(), 0) >= 2 or h.lower() in _FUNC
                        or h_blob)
                if h_ok and ((len(t2) >= 4 and t2[0].islower()
                              and words.get(t2.lower(), 0) >= thr
                              and (not h_blob or len(t2) >= 6
                                   or words.get(t2.lower(), 0) >= 5))
                             or (t2.lower() in _FUNC and len(t2) >= 2
                                 and words.get(h.lower(), 0) >= 2)):
                    fixed, nf = f"{h} {t2}", 1
                    break
        # misprint printed exactly twice: both parts common book words
        # (>=5), or a glued head (w==0, peeled by a later pass) plus a
        # common tail ("Causesof" + "referred")
        if (nf == 0 and core.isalpha() and len(core) >= 8
                and words.get(core.lower(), 0) == 2):
            for j in range(3, len(core) - 1):
                h, t2 = core[:j], core[j:]
                ch, ct = words.get(h.lower(), 0), words.get(t2.lower(), 0)
                h_ok = ch >= 5 or (len(h) >= 8 and ch == 0)
                if (ct >= 5 and h_ok
                        and (t2[0].islower() or t2.lower() in _FUNC)
                        and len(t2) >= (6 if ch == 0 else 2)
                        and (h[0].islower() or h.lower() in _FUNC
                             or ch >= 5 or ch == 0)):
                    fixed, nf = f"{h} {t2}", 1
                    break
        # content word + glued function tail ("negligibleor"): the
        # head is a real word the book prints, the glued form is not
        if (nf == 0 and core.isalpha() and len(core) >= 8
                and words.get(core.lower(), 0) <= 1):
            for f in _FUNC:
                if core[-len(f):].lower() == f and len(core) - len(f) >= 6:
                    h = core[:-len(f)]
                    hw = words.get(h.lower(), 0)
                    prod = h.lower().endswith(
                        ("ible", "able", "ence", "ance", "tion",
                         "ment", "ness", "ous", "ive"))
                    if h[0].islower() and hw <= 2 and (hw >= 1 or prod):
                        fixed, nf = f"{h} {core[-len(f):]}", 1
                        break
        # wrapped fragments whose join is a book word: neither part is
        # a standalone word ("tiss ues", "su rface", "Sp henoid",
        # "or bit"). "brain stem" is safe: both parts are real words.
        if nf == 0 and i + 1 < len(parts):
            raw_next = parts[i + 1]
            nxt, npunct = raw_next, ""
            while nxt and nxt[-1] in ",.;:!?)]":
                npunct = nxt[-1] + npunct
                nxt = nxt[:-1]
            # A CAMEL BOUNDARY IS PRINTED EVIDENCE OF A WORD START and
            # the vocab join must not undo the camel split that already
            # ran. Without this, `belli Micr` (from `Isospora belliMicr`
            # in the organism list) was re-glued to `belliMicr` because
            # the book "prints" that string TWICE — both times as the
            # same line-wrap artifact on p497 and p563. A broken line
            # printed twice is not a printed word.
            camel = (core[-1:].islower() and len(nxt) >= 2
                     and nxt[0].isupper() and nxt[1].islower())
            if core.isalpha() and nxt.isalpha() and len(nxt) >= 1 and not camel:
                jn = (core + nxt).lower()
                wa, wb = words.get(core.lower(), 0), words.get(nxt.lower(), 0)
                jj = words.get(jn, 0)
                if (jj >= 2 and wa <= 2 and wb <= 2) or \
                   (jj >= 10 and wb <= 3 and wa <= 6 and nxt[0].islower()) or \
                   (core.lower() in _FUNC and jj >= 5 and wb <= 1) or \
                   (jj >= (10 if len(nxt) == 1 else 2) and jj > wa and jj > wb
                    and jn.startswith(core.lower())
                    and jn.endswith(nxt.lower())
                    and len(core) >= 3 and len(nxt) >= 1) or \
                   (len(core) == 1 and core.lower() not in "ai"
                    and nxt[0].islower() and jj >= 50
                    and jj > 10 * wa and jj > 10 * wb
                    and pairs.get((core.lower(), nxt.lower()), 0) == 0):
                    fixed, punct = fixed + nxt, npunct + punct
                    nfix += 1
                    i += 1
        nfix += nf
        res.append(fixed + punct)
        i += 1
    # number range: "50 – 300" -> "50–300" (dash between digits)
    out, j = [], 0
    while j < len(res):
        t = res[j]
        if (t in ("–", "-", "-") and j > 0 and j + 1 < len(res)
                and res[j - 1].isdigit() and res[j + 1].isdigit()):
            out[-1] = out[-1] + t + res[j + 1]
            nfix += 1
            j += 2
            continue
        out.append(t)
        j += 1
    # ordinals / vertebral notations / brackets: deterministic spacing
    for k, t in enumerate(out):
        ft = spacing_fix(t)
        if ft != t:
            out[k], nfix = ft, nfix + 1
    return out, nfix


# ------------------------------------------------------------------ rules

def _segments(book, pg: int):
    """Long horizontal / vertical rule segments on a page."""
    p = book.doc[pg - 1]
    h, v = [], []
    for it in p.get_drawings():
        for item in it["items"]:
            if item[0] == "l":
                a, b = item[1], item[2]
                seg = (min(a.x, b.x), min(a.y, b.y),
                       max(a.x, b.x), max(a.y, b.y))
            elif item[0] == "re":
                r = item[1]
                seg = (r.x0, r.y0, r.x1, r.y1)
            else:
                continue
            if seg[3] - seg[1] < 2.5 and seg[2] - seg[0] > 30:
                h.append(seg)
            elif seg[2] - seg[0] < 2.5 and seg[3] - seg[1] > 8:
                v.append(seg)
    return h, v


def grid(book, pg: int, box):
    """(column edges, row rule ys) of one ruled box, from its own rules.

    Empty row list means "no reliable row rules" — the caller falls
    back to one markdown row per baseline (v2.0 behaviour)."""
    h, v = _segments(book, pg)
    tol = 4.0
    cols = sorted({round(s[0], 1) for s in v
                   if s[1] >= box[1] - tol and s[3] <= box[3] + tol
                   and box[0] - tol <= s[0] <= box[2] + tol})
    if len(cols) < 2 or cols[0] > box[0] + 12 or cols[-1] < box[2] - 12:
        cols = [round(box[0], 1), round(box[2], 1)]
    rows = sorted({round(s[1], 1) for s in h
                   if s[0] >= box[0] - tol and s[2] <= box[2] + tol
                   and box[1] - tol <= s[1] <= box[3] + tol})
    if len(rows) < 2 or rows[0] > box[1] + 12 or rows[-1] < box[3] - 12:
        rows = []
    return cols, rows


def _box_lines(book, pg: int, box):
    pd = book.page(pg)
    out = []
    for l in pd.lines:
        ycen = (l.y0 + l.y1) / 2
        if (box[0] - 2 <= l.x0 <= box[2] + 2
                and box[1] - 3 <= ycen <= box[3] + 3):
            out.append(l)
    return out


# ------------------------------------------------------------------ box

# --- punctuation spacing -------------------------------------------------
# The corrected ED8 text layer glues/splits punctuation the same way it
# glues words: the printed cell `...at thesame time,e.g- ...` reaches the
# extractor as `... at the same time,e.g- ...`, and a cell wrapped by the
# typesetter after "host" + "’s immune system" rejoins as "host ’s
# immune system". Both are pure SPACING defects: no letter, digit or
# symbol changes. They used to be left in the markdown, and then the
# Gemini refinement proposed exactly this fix and the fidelity validator
# had to REJECT it (a space is a character to the validator), so the
# repair never reached the output.
#
# The rules below are the same ones the refinement prompt asks the model
# to apply, moved into the DETERMINISTIC stage where they are counted
# (chapter_completeness.tables.punct_space_fixes), provable and free:
#   * no space before a closing punctuation / quote:  "host ’s" -> "host’s"
#   * no space before , ; : ) %                       "1 , 2"   -> "1, 2"
#   * one space after , ; : when a LETTER is on BOTH sides
#     ("time,e.g-" -> "time, e.g-") — the letter guard keeps
#     numbers and codes intact: "O157:H7", "1,000" and
#     "A-4,B-3" are left exactly as printed
# Digits are never touched on the left of the rule ("1,000" and "C2,C3"
# stay exactly as printed: the letter guard is what keeps numeric
# thousands and cervical-level codes intact), and no letters/digits/
# hyphens are ever added, removed or reordered.
_NO_SPACE_BEFORE = re.compile(r"(?<=[^\s])\s+([,;:%\)\]\u2019'\u201d])")
_SPACE_AFTER = re.compile(r"(?<=[A-Za-z])([,;:])(?=[A-Za-z])")


def punct_spacing(text: str) -> tuple:
    """(repaired text, number of repairs) — spacing only, never content."""
    if not text:
        return text, 0
    n = 0
    out = text
    for _ in range(4):                      # chained artifacts
        fixed, k = _NO_SPACE_BEFORE.subn(r"\1", out)
        fixed2, k2 = _SPACE_AFTER.subn(r"\1 ", fixed)
        if not (k + k2):
            break
        n += k + k2
        out = fixed2
    return out, n


@dataclass
class BoxTable:
    page: int
    box: tuple
    cols: list                    # column edge xs
    rows: list                    # cell matrix (list of rows of str)
    header: tuple                 # first row (for continuation tests)
    line_joins: int = 0
    camel_fixes: int = 0
    vocab_fixes: int = 0
    punct_fixes: int = 0          # space around , ; : ( ) quotes repaired
    med_token_fixes: int = 0      # split medical tokens rejoined (C D4->CD4)
    llm_fixes: int = 0
    verify_calls: int = 0
    verify_clear: bool = False
    warnings: list = field(default_factory=list)


def alnum_vocab(book):
    """Counter of printed ALPHANUMERIC tokens (lowercased) — the evidence
    pool for join_medical_tokens. Shares build_vocab's single page scan;
    a book whose word vocab was never built gets it built here."""
    cached = getattr(book, "_qbank_mixed", None)
    if cached is not None:
        return cached
    try:
        build_vocab(book)
    except Exception:                           # noqa: BLE001
        return Counter()
    return getattr(book, "_qbank_mixed", None) or Counter()


def camel_items(book):
    """Counter of printed ITEM forms that only ever appear glued inside a
    longer token or wrapped across a line ("Bipolaris" only exists as
    `richardsiaeBipolaris`; "Phaeohyphomycosi" + "s"). Splitting one of
    these is undoing the print's own structure, so the vocab repair must
    leave them whole."""
    cached = getattr(book, "_qbank_items", None)
    if cached is not None:
        return cached
    try:
        build_vocab(book)
    except Exception:                           # noqa: BLE001
        return Counter()
    return getattr(book, "_qbank_items", None) or Counter()


# --- medical-token joins ----------------------------------------------
# The ED8 text layer sometimes leaves a space INSIDE an alphanumeric
# token: the printed `CD4+` arrives as `C D4+`, `STAT3` as `STA T3`,
# `UL97` as `U L97`. This is not a presentation choice — a reader
# searching the output for `CD4` misses the row — so it is repaired
# deterministically, on every text-bearing field (stem, options,
# solution, table cells), exactly as the forensic audit required.
#
# The candidate shape is deliberately narrow: the SECOND piece must
# contain a digit. That single requirement is what keeps ordinary
# English out of the pass — `brain stem`, `T cell`, `B cell`,
# `bone marrow`, `red blood cell` can never match, so their joined forms
# are never even proposed.
# Second piece: starts with an UPPERCASE letter or a digit and contains
# a digit. Requiring the uppercase/digit start is what refuses `C d4`
# (a lowercase tail is not the split shape the extractor produces), and
# requiring a digit is what refuses ordinary English — the two guards
# together are why `brain stem` / `T cell` / `bone marrow` can never be
# candidates.
_MED_JOIN = re.compile(
    r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{0,5})\s+"
    r"((?:[A-Z][A-Za-z0-9+\-]*|[0-9][A-Za-z0-9+\-]*)\d?"
    r"[A-Za-z0-9+\-]*)(?![A-Za-z0-9])")
# ...and a WRAPPED NUMBER (`OX 1 9` in a narrow table column: the
# printed `OX 19` broke after the first digit). Three pieces, all merged
# at once and gated on the full form, so only a printed `OX19` joins.
_MED_JOIN_NUM = re.compile(
    r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{0,5})\s+(\d{1,3})\s+(\d{1,3})(?![0-9])")
# ...and a three-piece LETTER split (`N O D1`, the shape the audit
# named) would be unreachable through the plain rule, because the
# intermediate `OD1` has no evidence of its own. This pattern merges all
# three at once, and is still gated on the FULL form: only a printed
# `NOD1` joins.
_MED_JOIN3 = re.compile(
    r"(?<![A-Za-z0-9])([A-Z])\s+([A-Z])\s+"
    r"((?:[A-Z][A-Za-z0-9+\-]*|[0-9][A-Za-z0-9+\-]*)\d?"
    r"[A-Za-z0-9+\-]*)(?![A-Za-z0-9])")


def join_medical_tokens(text: str, mixed=None) -> tuple:
    """(repaired text, [(before, after), ...]) — joins a split medical
    token ONLY where the book itself prints the joined form.

    The forensic audit was explicit that the detector regex is a
    CANDIDATE GENERATOR and must never be run blindly: joining every
    uppercase/number pair would rewrite `A 5-year-old` into
    `A5-year-old`. Evidence comes from `mixed`, the book's own
    alphanumeric vocabulary, so the pass can only restore a form the
    book prints somewhere.

    Fidelity contract: only WHITESPACE may change. The result with all
    whitespace removed is identical to the input's, character for
    character — nothing is added, dropped, substituted or spell-fixed.
    """
    if not text or mixed is None or not mixed:
        return text, []
    before = re.sub(r"\s+", "", text)
    out = text
    repairs: list = []
    for _ in range(6):                  # fixed point: N O D1 -> NOD1
        def _sub(m):
            joined = m.group(1) + m.group(2)
            if not any(ch.isdigit() for ch in joined):
                return m.group(0)           # first guard: digits required
            if joined.lower() in mixed:     # second guard: book evidence
                repairs.append((m.group(0), joined))
                return joined
            return m.group(0)

        def _sub3(m):
            joined = m.group(1) + m.group(2) + m.group(3)
            if not any(ch.isdigit() for ch in joined):
                return m.group(0)
            if joined.lower() in mixed:
                repairs.append((m.group(0), joined))
                return joined
            return m.group(0)

        new = _MED_JOIN_NUM.sub(_sub3, _MED_JOIN3.sub(_sub3,
                                                        _MED_JOIN.sub(_sub, out)))
        if new == out:
            break
        out = new
    if repairs and re.sub(r"\s+", "", out) != before:
        raise AssertionError(                       # safety net
            "join_medical_tokens changed content, not just spacing: "
            f"{before[:60]!r} -> {re.sub(r'\\s+', '', out)[:60]!r}")
    return out, repairs


def _join_decision(prev, nxt, fill_x1, fill_reaches_edge, vocab=None) -> str:
    """'glue' (no space), 'hyphen' (no space, keep '-') or 'space'.

    A mid-word split only happens when the typesetter ran out of room:
    the line must fill the column (fill edge) AND the column's fill
    edge must itself reach near the column's right rule. In a column
    of short lines nothing ever had to split mid-word, so every wrap
    there is a deliberate word boundary."""
    t = prev.text.rstrip()
    n = nxt.text.lstrip()
    if not t or not n:
        return "space"
    if t.endswith("-"):
        return "hyphen"
    if t[-1] in ".;:!?":
        return "space"
    if vocab is not None:
        # wrapped fragment the flush heuristic cannot decide: glue when
        # the book's own vocabulary says the concatenation is a real
        # word at least as common as each part alone, with at least
        # one part too rare to be a deliberate standalone word
        # ("fl"+"ow", "Atri"+"al", "inc"+"reasing", "ductu"+"s").
        words = vocab[0]
        a, b = t.split()[-1], n.split()[0]
        ca, cb = words.get(a.lower(), 0), words.get(b.lower(), 0)
        combo = words.get((a + b).lower(), 0)
        # same invariant as _repair_tokens: a capital letter inside a run
        # is a word start, so a "printed" glue across that boundary is a
        # broken line, not a word
        # ... and a SINGLE CAPITAL is an initial, i.e. a new list item,
        # never a wrap fragment: this book's organism/species lists are
        # printed as `... grisea` / `E. jeanselmei`, and gluing there
        # produced `Madurella griseaE jeanselmei` (the same broken line
        # twice made `griseae` look printed).
        single_initial = len(b) == 1 and b.isupper()
        if (a.isalpha() and b.isalpha() and combo >= 2
                and not single_initial
                and not (a[-1:].islower() and len(b) >= 2
                         and b[0].isupper() and b[1].islower())):
            if combo >= ca and combo >= cb and min(ca, cb) <= 2:
                return "glue"
            if (len(a) == 1 or len(b) == 1) and min(ca, cb) <= 2:
                return "glue"
    # Tolerance MEASURED on this book (615 pages, every ruled box):
    # line pairs whose gap to the column's fill edge falls in 2.5-5.0 pt
    # are 66 for 66 MID-WORD WRAPS ("sulfur-containi|ng",
    # "pres|ent", "he|matogenous", "Poxviru|s", "belliMicr|osporidia"),
    # while the next bucket (5-8 pt, 10 pairs) is already mixed — it
    # holds real word boundaries such as "cells|monoclonally". Those are
    # left to the vocabulary path, which needs the joined form to be a
    # printed word, so 5.0 pt is the widest cut that is still evidence.
    flush = fill_reaches_edge and prev.x1 >= fill_x1 - 5.0
    if not flush:
        return "space"
    if t[-1] == "," and n[0].isdigit():
        return "glue"          # "...," + "3"
    if len(t) >= 2 and t[-2] == "," and n[0].isdigit():
        return "glue"          # "C2,C" + "3"  (comma-list continuation)
    if t[-1].isalnum() and n[0].islower():
        return "glue"          # "su" + "rface", "Grad" + "e 1"
    return "space"


def long_space_suspects(text: str, words) -> list:
    """>=12-letter tokens that are NOT established book words — the
    glued multi-word blob signature. Established long terms
    (Mucoperichondrial, arteriosclerosis, suprastructures) are
    legitimate and stay unflagged; without a vocabulary there is no
    evidence either way, so nothing is flagged."""
    if words is None:
        return []
    return [t for t in _LONG_TOKEN.findall(text)
            if words.get(t.lower(), 0) < 2]


def qa_suspects(matrix: list, words, pairs=None) -> list:
    """Suspect word fragments in a cell matrix, judged with the book's
    own vocabulary. A flag must point at a plausible MALFORMATION,
    never at legitimate multi-word terminology ("brain stem",
    "In Complete palsy") or proper names (Freer, Killian):

      (a) adjacent PAIR whose glued join beats both parts in print
          count — the signature of a collision the book itself
          repeats ("do me" when "dome" outscores "do"/"me"); a join
          that is rarer than either part is a misprint of the PAIR,
          so the spaced pair is correct text, not a suspect;
      (b) adjacent PAIR of two rare tokens whose join is a book word
          ("tiss"+"ues", "osteocal"+"cin") — a mid-word wrap;
      (c) rare lowercase token that is itself a book word FRAGMENT
          (head or tail of some printed word, or a two-way split
          into book words) — "rface", "henoid", "ncha";
      (d) >=14-letter token printed nowhere (any case): a glued
          multi-word blob ("Intracranialintraduraltumorwith...").
    Capitalised unknown tokens are assumed proper nouns; established
    words (>=2x) are never suspects."""
    w = words
    heads = {t[:k] for t in w for k in range(2, len(t))}
    tails = {t[-k:] for t in w for k in range(2, len(t))}

    def _split2(tok):
        # a collision is only suspect when a function word is glued
        # in ("andhas", "oroptic"); content+content joins like
        # "antihelix" (anti+helix) are legitimate medical terms.
        # PAIR EVIDENCE is required too: the two parts must be printed
        # next to each other somewhere in the book, otherwise the
        # "glue" is just a real word that happens to contain a function
        # word ("independent" = in+dependent, "ingredient" =
        # in+gredient) — two false REVIEW flags on the real book came
        # from exactly that, and a REVIEW flag locks the export gate.
        for k in range(1, len(tok)):
            if (tok[:k].lower() in _FUNC or tok[k:].lower() in _FUNC) \
                    and w.get(tok[:k], 0) >= 2 and w.get(tok[k:], 0) >= 2:
                if pairs is None:
                    return True
                if pairs.get((tok[:k].lower(), tok[k:].lower()), 0) >= 1:
                    return True
        return False

    def _func_glue(tok):
        # chained glue: closed-class prefix stuck onto a run that is
        # itself two book words ("but"+"mobile"+"on") — every part
        # must be established print, else it is not evidence
        for k in range(2, len(tok) - 4):
            if tok[:k] not in _FUNC:
                continue
            rest = tok[k:]
            for j in range(3, len(rest) - 2):
                if w.get(rest[:j], 0) >= 2 and w.get(rest[j:], 0) >= 2:
                    return True
        return False

    qa: list = []
    for r in matrix:
        for c in r:
            # a HYPHEN is a printed separator, not a lost space:
            # tokenising "re-assortment" as "re"+"assortment" made the
            # fragment "assortment" look like the tail of a wrapped word
            # (the book prints "reassortment" elsewhere) and raised a
            # REVIEW on the real book's 025-T01, which locks the gate
            toks = re.findall(r"[A-Za-z]{2,}(?:-[A-Za-z]{2,})*", str(c))
            for a, b in zip(toks, toks[1:]):
                al, bl = a.lower(), b.lower()
                if al in _FUNC and bl in _FUNC:
                    continue        # "in"+"to" is not corruption
                if al in _LEGIT or bl in _LEGIT:
                    continue
                if a[0].isupper() or b[0].isupper():
                    continue        # proper-name pairs (Freer incision)
                if pairs is not None and pairs.get((al, bl), 0) >= 2:
                    continue        # the book prints the spaced pair
                j = w.get(al + bl, 0)
                wa, wb = w.get(al, 0), w.get(bl, 0)
                # (a) only when the join outscores BOTH parts — the
                # signature of a collision the book repeats; legitimate
                # spaced terms ("brain stem") never satisfy this
                if j >= 2 and j > wa and j > wb:
                    qa += [a, b]
            for k, t in enumerate(toks):
                lo = t.lower()
                if w.get(lo, 0) >= 2 or t[0].isupper() or lo in _FUNC \
                        or lo in _LEGIT:
                    continue
                if len(lo) >= 16 and w.get(lo, 0) == 0 and "-" not in lo:
                    # (d) a glued multi-word blob. A HYPHEN is printed
                    # structure, not glue: "cell-independent" is 16
                    # characters and perfectly legitimate.
                    qa.append(t)
                elif len(lo) >= 3 and _split2(lo):
                    qa.append(t)                    # (c) two-way split
                elif len(lo) >= 6 and _func_glue(lo):
                    qa.append(t)                    # (c2) chained glue
                elif len(lo) >= 3 and (lo in heads or lo in tails):
                    # a fragment only with its complement partner:
                    # "rface" after "su" (join "surface" is printed);
                    # lone rare words like "drum" stay unflagged
                    for p in (toks[k - 1] if k else None,
                              toks[k + 1] if k + 1 < len(toks) else None):
                        # a function-word neighbour whose glue is a
                        # misprint blob ("negligible"+"or") is not
                        # complement evidence. The glued form must be
                        # ESTABLISHED (>=2 prints) to read as a wrap: a
                        # form printed exactly ONCE is the artifact
                        # itself, not a word — "Sensorylanguage" (p366)
                        # and "Bilateraljugulodigastric" (p590) each
                        # print once, so "language"/"jugulodigastric"
                        # are ordinary words whose separating space the
                        # typesetter dropped, and flagging them raised
                        # two REVIEWs that locked the export gate.
                        if p and p.lower() not in _FUNC \
                                and (w.get((p + t).lower(), 0) >= 2
                                     or w.get((t + p).lower(), 0) >= 2):
                            qa.append(t)
                            break
    return qa


def build_box(book, pg: int, box, counts, vocab=None, llm=None,
              verify=None) -> BoxTable:
    """Cell matrix of ONE ruled box with in-cell line reconstruction."""
    cols, row_ys = grid(book, pg, box)
    lines = _box_lines(book, pg, box)
    ncols = len(cols) - 1
    bt = BoxTable(page=pg, box=tuple(box), cols=list(cols),
                  rows=[], header=())
    if not lines:
        return bt

    def col_of(x0):
        c = 0
        for i in range(ncols):
            if x0 >= cols[i] - 4:
                c = i
        return c

    def band_of(ycen):
        for i in range(len(row_ys) - 1):
            if row_ys[i] - 3 <= ycen <= row_ys[i + 1] + 3:
                return ("r", i)
        # outside row rules or no row rules: per-baseline fallback
        return ("y", round(ycen / 3.0))

    # the book's alphanumeric vocabulary (CD4, STAT3, ...): evidence for
    # the medical-token joins below. Memoised on the book, so this is a
    # dict lookup in every box after the first.
    mixed = alnum_vocab(book) if book is not None else None
    items = camel_items(book) if book is not None else None

    # measured fill edge per column (typesetter's text extent)
    fill = [0.0] * ncols
    for l in lines:
        fill[col_of(l.x0)] = max(fill[col_of(l.x0)], l.x1)
    # a column can only force mid-word splits when its fill edge
    # actually reaches near the right rule (12pt ~ cell padding + slack)
    fill_reaches = [fill[c] >= cols[c + 1] - 12 for c in range(ncols)]

    # per column: visual-order lines -> cells (band groups, joined)
    percol = {c: [] for c in range(ncols)}
    for l in sorted(lines, key=lambda l: (round(l.y0 / 3.0), l.x0)):
        percol[col_of(l.x0)].append(l)

    cells = {}          # (bandkey, col) -> text
    band_order = []     # unique bandkeys in visual order
    seam = set()        # tokens built by a glue join (must stay whole)
    for c, lns in percol.items():
        cur_band, cur, prev_line = None, None, None
        for l in lns:
            bk = band_of((l.y0 + l.y1) / 2)
            txt = l.text.strip()
            if not txt:
                continue
            if cur is None or bk != cur_band:
                cur_band, cur, prev_line = bk, txt, l
                cells[(bk, c)] = cur
                if bk not in band_order:
                    band_order.append(bk)
                continue
            how = _join_decision(prev_line, l, fill[c], fill_reaches[c],
                                 vocab)
            if how == "space":
                cur = cur + " " + txt
            else:
                # the seam form is PRINTED WHOLE: the geometry says the
                # typesetter ran out of room and continued the word, so
                # the vocab repair must not cut it back apart later
                # (`aquaspersa`, `Phaeohyphomycosis` were both re-split
                # into `aquaspers a` / `Phaeohypho mycosis`)
                cur = cur + txt
                bt.line_joins += 1
                seam.add(cur.split()[-1].lower())
            cells[(bk, c)] = cur
            prev_line = l
    band_order.sort(key=lambda bk: (bk[0] != "r", bk[1]))

    # matrix + camel repair + glyph repair + validation warnings
    matrix = []
    for bk in band_order:
        row = []
        for c in range(ncols):
            t = cells.get((bk, c), "")
            if t:
                before_toks = set(t.split())
                fixed, n = _CAMEL.subn(" ", t)
                if n:
                    bt.camel_fixes += n
                    t = fixed
                # ...and a token that ENDS in one capital after a
                # lowercase run is the same print evidence: the capital
                # starts a new item ("Madurella griseaE jeanselmei" is
                # "… grisea" + "E jeanselmei"). Book-wide this pattern
                # matches exactly one token, and the established
                # lowercase-prefix acronyms ("cccDNA", "dsDNA", "ssRNA")
                # keep a CAPITAL RUN after the prefix, so they are never
                # touched. The split is registered in `seam` before the
                # vocab repair, so it cannot be re-glued either.
                if _CAMEL_TAIL.search(t):
                    t, n2 = _CAMEL_TAIL.subn(" ", t)
                    bt.camel_fixes += n2
                # a camel split is print evidence in itself: the pieces
                # are printed ITEMS, so protect them from the vocab
                # repair too ("aquaspersaCladophialophora" -> "aquaspersa
                # Cladophialophora", where the repair then cut
                # "aquaspersa" into "aquaspers a")
                seam.update(w.lower() for w in t.split()
                            if w not in before_toks)
                if vocab is not None:
                    # fixed point: chained glues ("theinfrat...") peel                if vocab is not None:
                    # fixed point: chained glues ("theinfrat...") peel
                    # one repair per pass
                    whole = None
                    if seam or items:
                        whole = set(seam)
                        whole |= {w for w in items}
                    for _ in range(6):
                        fixed_parts, nf = _repair_tokens(
                            t.split(" "), vocab[0], vocab[1], whole)
                        bt.vocab_fixes += nf
                        t = " ".join(fixed_parts)
                        if not nf:
                            break
                for k, v in _VERIFIED_MERGES.items():
                    if k in t:
                        t = re.sub(rf"\b{re.escape(k)}\b", v, t)
                        bt.vocab_fixes += 1
                t, npf = punct_spacing(t)
                bt.punct_fixes += npf
                if mixed:
                    t, joins = join_medical_tokens(t, mixed)
                    if joins:
                        bt.med_token_fixes += len(joins)
                t = glyphs.repair(t, counts)
                for tok in long_space_suspects(
                        t, vocab[0] if vocab is not None else None):
                    bt.warnings.append(f"suspect_lost_space:{tok[:20]}")
            row.append(t)
        if any(row):
            matrix.append(row)
    bt.rows = matrix
    bt.header = tuple(matrix[0]) if matrix else ()
    if llm is not None and matrix:
        lm = llm(book, pg, box)
        if lm:
            merged, n = merge_llm(matrix, lm, vocab)
            matrix = merged
            bt.llm_fixes = n
            bt.rows = matrix
            bt.header = tuple(matrix[0])
        # second pass: when the QA scan still sees suspect fragments,
        # send the same box back to the model naming them; the answer
        # goes through the identical fidelity envelope
        if verify is not None and vocab is not None:
            susp = qa_suspects(matrix, vocab[0], vocab[1])
            if susp:
                bt.verify_calls += 1
                lm2 = verify(book, pg, box, sorted(set(susp))[:8])
                if lm2:
                    merged2, n2 = merge_llm(matrix, lm2, vocab)
                    # model re-read the box and found nothing to fix:
                    # the remaining flags are false positives
                    bt.verify_clear = n2 == 0
                    matrix = merged2
                    bt.llm_fixes += n2
                    bt.rows = matrix
                    bt.header = tuple(matrix[0])
    return bt


# ------------------------------------------------------- logical tables

@dataclass
class LogicalTable:
    table_id: str
    markdown: str
    chunks: list                       # [(page, box)] in reading order
    header_deduplicated: bool
    line_joins: int
    camel_fixes: int
    vocab_fixes: int
    punct_fixes: int
    llm_fixes: int
    cross_page: bool
    warnings: list
    qa_tokens: list = field(default_factory=list)

    @property
    def source_pages(self) -> list:
        return [pg for pg, _ in self.chunks]

    def as_record(self) -> dict:
        return {
            "type": "table",
            "markdown": self.markdown,
            "table_id": self.table_id,
            "source_pages": self.source_pages,
            "merged_continuation": self.cross_page,
            "header_deduplicated": self.header_deduplicated,
            "extraction": ("ruled_grid_geometry+gemini_transcription"
                           if self.llm_fixes else "ruled_grid_geometry"),
            "validation": {
                "status": "warnings" if self.warnings else "ok",
                "line_joins": self.line_joins,
                "camel_space_fixes": self.camel_fixes,
                "vocab_space_fixes": self.vocab_fixes,
                "punct_space_fixes": self.punct_fixes,
                "llm_space_repairs": self.llm_fixes,
                "table_qa": {
                    "status": "REVIEW" if self.qa_tokens else "ok",
                    "suspect_fragments": self.qa_tokens[:12],
                },
                "warnings": sorted(set(self.warnings))[:8],
            },
        }


def _markdown(matrix) -> str:
    md = []
    for i, row in enumerate(matrix):
        md.append("| " + " | ".join(row) + " |")
        if i == 0:
            md.append("|" + "---|" * len(row))
    return "\n".join(md)


def _rowkey(row) -> str:
    """Whitespace/punct-insensitive fingerprint of one table row."""
    return re.sub(r"[^a-z0-9]", "", "".join(row).lower())


class ChapterTables:
    """Logical-table registry for one chapter.

    Boxes are chained into logical tables; a block that touches any box
    of a chain gets THAT logical table (one record, one id) plus one
    render region per contributing box."""

    def __init__(self, book, chapter_no: int, first_page: int,
                 last_page: int, vocab=None, llm=None, verify=None):
        self.book = book
        self.chapter_no = chapter_no
        self.vocab = vocab
        self.llm = llm
        self.verify = verify
        self.boxes = []                 # ordered [(pg, box)]
        for pg in range(first_page, min(last_page, book.total_pages) + 1):
            for bx in sorted(book.page(pg).table_boxes, key=lambda b: b[1]):
                self.boxes.append((pg, tuple(bx)))
        self._bt_cache: dict = {}
        self._own_counts = Counter()
        self._chain_of: dict = {}       # (pg, box) -> chain index
        self._chains = self._build_chains()
        self._lt_cache: dict = {}
        self._counter = 0

    def _bt(self, pg, box, counts=None) -> BoxTable:
        key = (pg, box)
        if key not in self._bt_cache:
            self._bt_cache[key] = build_box(
                self.book, pg, box,
                counts if counts is not None else self._own_counts,
                self.vocab, self.llm, self.verify)
        return self._bt_cache[key]

    def _continues(self, i) -> str | None:
        """Does box i+1 continue box i? -> 'dedup' | 'keep' | None."""
        (pg, box), (npg, nbox) = self.boxes[i], self.boxes[i + 1]
        if npg != pg + 1:
            return None
        bt, nbt = self._bt(pg, box), self._bt(npg, nbox)
        if len(bt.cols) != len(nbt.cols):
            return None
        if any(abs(a - b) > 6 for a, b in zip(bt.cols, nbt.cols)):
            return None
        if bt.header and bt.header == nbt.header:
            # repeated header AND repeated first data row: the next
            # page prints a variant COPY of the same table (the book
            # does this for adjacent solutions), not a continuation
            if (len(bt.rows) > 1 and len(nbt.rows) > 1
                    and _rowkey(bt.rows[1]) == _rowkey(nbt.rows[1])):
                return None
            return "dedup"                       # repeated header
        # no repeated header: document-flow continuation — previous box
        # is the page's last, next box the next page's first, starting
        # in the top quarter, previous box ending in the bottom 40%.
        ph = self.book.page(npg).height
        last_on_pg = all(p != pg or b[1] <= box[1] for p, b in self.boxes)
        first_on_npg = all(p != npg or b[1] >= nbox[1]
                           for p, b in self.boxes)
        if (last_on_pg and first_on_npg and nbox[1] <= ph * 0.25
                and box[3] >= ph * 0.6):
            return "keep"
        return None

    def _build_chains(self) -> list:
        chains, cur = [], []
        for i, (pg, box) in enumerate(self.boxes):
            if not cur:
                cur = [(pg, box, None)]
                continue
            mode = self._continues(i - 1)
            if mode:
                cur.append((pg, box, mode))
            else:
                chains.append(cur)
                cur = [(pg, box, None)]
        if cur:
            chains.append(cur)
        for ci, ch in enumerate(chains):
            for pg, box, _m in ch:
                self._chain_of[(pg, box)] = ci
        return chains

    def lookup(self, pg: int, box, counts=None) -> LogicalTable | None:
        key = (pg, tuple(box))
        ci = self._chain_of.get(key)
        if ci is None:
            return None
        if ci in self._lt_cache:
            return self._lt_cache[ci]
        self._counter += 1
        tid = f"{self.chapter_no:03d}-T{self._counter:02d}"
        chunks = self._chains[ci]
        matrix, joins, camels, vfix, pfx, lfix, warns = [], 0, 0, 0, 0, 0, []
        dedup = False
        bts = []
        for k, (cpg, cbox, mode) in enumerate(chunks):
            bt = self._bt(cpg, cbox, counts)
            bts.append(bt)
            joins += bt.line_joins
            camels += bt.camel_fixes
            vfix += bt.vocab_fixes
            pfx += bt.punct_fixes
            lfix += bt.llm_fixes
            warns += bt.warnings
            rows = bt.rows
            if k and mode == "dedup" and rows and rows[0] == matrix[0]:
                rows = rows[1:]
                dedup = True
            if k == 0:
                matrix = list(rows)
            else:
                matrix += rows
        qa: list = []
        if self.vocab is not None and matrix:
            qa = qa_suspects(matrix, self.vocab[0], self.vocab[1])
            # a REVIEW flag is only meaningful when the visual second
            # pass agrees something is wrong: every box that was
            # re-read and came back clean downgrades the flag
            if qa and any(b.verify_calls for b in bts) and all(
                    b.verify_clear for b in bts if b.verify_calls):
                qa = []
        lt = LogicalTable(
            table_id=tid, markdown=_markdown(matrix) if matrix else "",
            chunks=[(p, b) for p, b, _ in chunks],
            header_deduplicated=dedup, line_joins=joins,
            camel_fixes=camels, vocab_fixes=vfix, punct_fixes=pfx,
            llm_fixes=lfix,
            cross_page=len(chunks) > 1, warnings=warns,
            qa_tokens=sorted(set(qa))[:12])
        self._lt_cache[ci] = lt
        return lt

    def materialised(self) -> list:
        """All logical tables actually referenced so far."""
        return list(self._lt_cache.values())

    def stats(self) -> dict:
        lts = list(self._lt_cache.values())
        return {
            "logical_tables": len(lts),
            "ruled_boxes": len(self.boxes),
            "cross_page_merges": sum(1 for t in lts if t.cross_page),
            "header_dedups": sum(1 for t in lts if t.header_deduplicated),
            "line_joins": sum(t.line_joins for t in lts),
            "camel_space_fixes": sum(t.camel_fixes for t in lts),
            "vocab_space_fixes": sum(t.vocab_fixes for t in lts),
            "punct_space_fixes": sum(t.punct_fixes for t in lts),
            "llm_space_repairs": sum(t.llm_fixes for t in lts),
            "llm_verify_calls": sum(
                bt.verify_calls for bt in self._bt_cache.values()),
            "tables_qa_review": sum(1 for t in lts if t.qa_tokens),
            "tables_with_warnings": sum(1 for t in lts if t.warnings),
        }
