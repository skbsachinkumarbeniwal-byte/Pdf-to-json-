"""Medical-token spacing repair (audit §12.6 required regression tests).

The forensic audit of the main-branch output found 90 occurrences of a
space inserted INSIDE an alphanumeric medical token — `C D4` for CD4,
`STA T3` for STAT3, `U L97` for UL97 — in question text, options,
solution text AND table cells. Its patch specification required:

  * the detector regex is a CANDIDATE GENERATOR, never a blind re.sub;
  * a candidate joins only when the joined form is established by the
    book's own printed vocabulary;
  * legitimate phrases (`brain stem`, `T cell`, `B cell`, ...) keep
    their spaces;
  * fidelity: whitespace-only change, character content identical;
  * explicit positive and negative tests.

These tests use a hand-made evidence pool so they assert the RULES
rather than the contents of one book.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qbank.tables import join_medical_tokens  # noqa: E402

# the forms the audit confirmed against the printed page
PRINTED = Counter({
    "cd4": 23, "cd40": 9, "cd3": 6, "cd8": 4, "stat3": 3, "cd10": 3,
    "ul97": 2, "ul54": 2, "cd16": 2, "cd56": 2, "il4": 2, "il5": 2,
    "nod1": 1, "b12": 4, "cd8+": 5, "cd4+": 8,
    "ox19": 1, "ox2": 1,
})


# ------------------------------------------------------------- positives

@pytest.mark.parametrize("broken, fixed", [
    ("C D4", "CD4"),
    ("C D40", "CD40"),
    ("C D8", "CD8"),
    ("C D3", "CD3"),
    ("C D10", "CD10"),
    ("U L97", "UL97"),
    ("U L54", "UL54"),
    ("STA T3", "STAT3"),
    ("I L4", "IL4"),
    ("I L5", "IL5"),
    ("C D16", "CD16"),
    ("C D56", "CD56"),
    ("B 12", "B12"),
])
def test_confirmed_splits_are_joined(broken, fixed):
    out, repairs = join_medical_tokens(broken, PRINTED)
    assert out == fixed
    assert repairs == [(broken, fixed)]


def test_join_inside_a_sentence():
    out, reps = join_medical_tokens(
        "CD4+ T cells and C D8+ cells were counted in 23 cases.", PRINTED)
    assert out == "CD4+ T cells and CD8+ cells were counted in 23 cases."
    assert reps == [("C D8+", "CD8+")]


def test_wrapped_number_in_a_narrow_column():
    """`OX 1 9` is the printed `OX 19` broken after the first digit by a
    narrow table column. Evidence for `ox19` comes from the book's own
    prose (which prints it), not from guesswork."""
    out, reps = join_medical_tokens("| DISEASE | OX 1 9 | OX 2 |", PRINTED)
    assert out == "| DISEASE | OX19 | OX2 |"
    assert ("OX 1 9", "OX19") in reps
    assert ("OX 2", "OX2") in reps


def test_wrapped_number_needs_the_printed_full_form():
    """`A 9- year-old` must NOT become `A9- year-old`: the hyphen means
    the pieces are not a wrapped number, and `a9-` is not printed."""
    out, reps = join_medical_tokens("A 9- year-old boy", PRINTED)
    assert out == "A 9- year-old boy"
    assert reps == []


def test_multi_piece_token_joins_at_fixed_point():
    """`N O D1` needs two passes — the audit called this out explicitly."""
    out, reps = join_medical_tokens("N O D1 receptor", PRINTED)
    assert out == "NOD1 receptor"
    assert ("N O D1", "NOD1") in reps


def test_repeated_and_multiple_tokens():
    out, reps = join_medical_tokens("C D4, C D4 and C D8", PRINTED)
    assert out == "CD4, CD4 and CD8"
    assert len(reps) == 3


@pytest.mark.parametrize("where", [
    "the C D4 count rose",              # question text
    "(b) low C D40",                    # option text
    "C D8+ cells predominate",          # solution text
    "| marker | C D56 |",               # table cell
])
def test_every_field_shape_is_repaired(where):
    out, reps = join_medical_tokens(where, PRINTED)
    assert reps, where
    assert "C D" not in out


# ------------------------------------------------------------- negatives

@pytest.mark.parametrize("phrase", [
    "brain stem", "bone marrow", "T cell", "B cell", "red blood cell",
    "notable", "cannot", "a 5-year-old child", "A 23-year-old woman",
    "SA 14-14-2 vaccine", "MGIT 960 system", "type A 5 patients",
    "vitamin D 3 drops",
])
def test_legitimate_spaces_are_never_touched(phrase):
    """The second piece must contain a digit and the joined form must be
    printed by the book — an arbitrary uppercase/number pair is left
    exactly as it came out of the extractor."""
    out, reps = join_medical_tokens(phrase, PRINTED)
    assert out == phrase
    assert reps == []


def test_candidate_without_source_evidence_is_left_alone():
    """`Q 7` looks like a candidate but the book never prints `Q7`."""
    out, reps = join_medical_tokens("Q 7 was wrong", PRINTED)
    assert out == "Q 7 was wrong"
    assert reps == []


def test_no_evidence_pool_is_a_no_op():
    for pool in (None, Counter()):
        out, reps = join_medical_tokens("C D4 here", pool)
        assert out == "C D4 here"
        assert reps == []


def test_lowercase_second_piece_is_not_a_candidate():
    """Digits are required, and the shape the extractor produces is
    uppercase; `C d4` is not a case this pass claims to fix."""
    out, reps = join_medical_tokens("C d4 here", PRINTED)
    assert out == "C d4 here"


# ------------------------------------------------------------- fidelity

@pytest.mark.parametrize("text", [
    "C D4+ T cells, C D8+ cells; STA T3 mutation and U L97 gene.",
    "The C D40 ligand binds C D40 (MGIT 960 aside).",
    "N O D1 and I L4 were both measured.",
])
def test_whitespace_only_change(text):
    out, _reps = join_medical_tokens(text, PRINTED)
    strip = lambda s: "".join(s.split())  # noqa: E731
    assert strip(out) == strip(text)
    assert len(out) < len(text)            # spaces removed, nothing added


def test_character_multiset_never_changes():
    text = "C D4 and U L97 and N O D1 and Q 9"
    out, _ = join_medical_tokens(text, PRINTED)
    strip = lambda t: sorted(t.replace(" ", ""))  # noqa: E731
    assert strip(out) == strip(text)


def test_idempotent():
    once, _ = join_medical_tokens("C D4 and STA T3", PRINTED)
    twice, reps = join_medical_tokens(once, PRINTED)
    assert twice == once
    assert reps == []


# ------------------------------------------- integration with spacing_fix

def test_spacing_fix_uses_the_pool_when_given():
    from qbank.tables import spacing_fix
    mixed = Counter({"cd4": 1})
    assert spacing_fix("the C D4 count", mixed) == "the CD4 count"
    # without a pool the behaviour is unchanged: shape-only repairs
    assert spacing_fix("the C D4 count") == "the C D4 count"


def test_spacing_fix_shape_rules_still_apply_with_a_pool():
    from qbank.tables import spacing_fix
    mixed = Counter({"cd4": 1})
    out = spacing_fix("the1strib shows C D4 ( EAC ) .", mixed)
    assert "1st rib" in out
    assert "(EAC)." in out
    assert "CD4" in out


# ---------------------------------------- table-only solutions (audit 5)

def test_table_only_solution_is_not_missing_solution_text():
    """MIC-034-022: the printed solution is a grid, not prose. An empty
    solution_text there is faithful extraction — flagging it INCOMPLETE
    was a QA/schema-rule bug, per the forensic audit."""
    from qbank.parse import grade_and_status
    rec = {
        "q_header_page": 586, "key_page": 587, "s_header_page": 600,
        "question_text": "Match the following parasites with their hosts:",
        "options": {l: f"opt {l}" for l in "ABCD"},
        "correct_option": "A", "solution_text": "", "flags": [],
        "tables": [{"table_id": "034-T03", "source_pages": [601]}],
        "_sol_tables": ["034-T03"],
    }
    grade, status, reasons = grade_and_status(rec)
    assert "missing:solution_text" not in reasons, reasons
    assert status != "INCOMPLETE"


def test_empty_solution_without_a_table_is_still_incomplete():
    from qbank.parse import grade_and_status
    rec = {
        "q_header_page": 1, "key_page": 2, "s_header_page": 3,
        "question_text": "A stem", "options": {l: f"opt {l}" for l in "ABCD"},
        "correct_option": "A", "solution_text": "", "flags": [],
        "tables": [{"table_id": "X-T01", "source_pages": [1]}],   # question side
    }
    _grade, status, reasons = grade_and_status(rec)
    assert "missing:solution_text" in reasons
    assert status == "INCOMPLETE"


def test_solution_table_on_another_page_does_not_count():
    """Only a table the SOLUTION BLOCK produced counts — a grid on the
    question side never satisfies the solution requirement."""
    from qbank.parse import grade_and_status
    rec = {
        "q_header_page": 1, "key_page": 2, "s_header_page": 3,
        "question_text": "A stem", "options": {l: f"opt {l}" for l in "ABCD"},
        "correct_option": "A", "solution_text": "", "flags": [],
        "tables": [{"table_id": "X-T01", "source_pages": [1, 2]}],
    }
    _grade, _status, reasons = grade_and_status(rec)
    assert "missing:solution_text" in reasons


# --------------------------- hyphen compounds are not lost spaces

def test_hyphenated_compound_is_not_a_suspect():
    """`re-assortment`: tokenising on the hyphen made `assortment` look
    like a fragment of the printed `reassortment` and raised a REVIEW on
    025-T01, which locks the export gate."""
    from collections import Counter as C
    from qbank.tables import qa_suspects
    words = C({"reassortment": 3, "re": 40, "large": 20, "genetic": 12})
    pairs = C()
    assert qa_suspects([["Large genetic re-assortment"]], words, pairs) == []


def test_real_wrap_fragment_is_still_flagged():
    """The (c) branch must keep working: `osteocal` + `cin`."""
    from collections import Counter as C
    from qbank.tables import qa_suspects
    words = C({"osteocalcin": 2, "osteocal": 1, "cin": 1, "the": 90})
    found = qa_suspects([["the osteocal cin level"]], words, C())
    assert "cin" in found or "osteocal" in found


# ------------------- questions the BOOK prints without their items

def test_match_question_with_no_items_is_reported():
    """MIC-030-006 / MIC-034-022: stem + combination options only, the
    numbered list never printed. Faithful extraction, unusable question."""
    from qbank.parse import grade_and_status
    base = {"q_header_page": 1, "key_page": 2, "s_header_page": 3,
            "options": {l: f"1-B, 2-{l}" for l in "ABCD"},
            "correct_option": "A", "flags": [], "tables": []}
    rec = {**base, "question_text": "Match the following:",
           "solution_text": "prose solution", "_sol_tables": []}
    _g, status, reasons = grade_and_status(rec)
    assert "source_missing_match_items" in reasons
    assert status == "REVIEW_NEEDED"


def test_match_question_with_a_printed_table_is_fine():
    from qbank.parse import grade_and_status
    rec = {"q_header_page": 1, "key_page": 2, "s_header_page": 3,
           "question_text": "Match the following:", "solution_text": "prose",
           "options": {l: f"opt {l}" for l in "ABCD"},
           "correct_option": "A", "flags": [], "_sol_tables": ["S-T01"],
           "tables": [{"table_id": "Q-T01", "source_pages": [1]}]}
    _g, status, reasons = grade_and_status(rec)
    assert "source_missing_match_items" not in reasons
    assert status == "READY"


def test_match_question_with_inline_items_is_fine():
    from qbank.parse import grade_and_status
    rec = {"q_header_page": 1, "key_page": 2, "s_header_page": 3,
           "question_text": "Match the following: 1. Ascaris 2. Hookworm",
           "solution_text": "prose", "tables": [], "_sol_tables": [],
           "options": {l: f"opt {l}" for l in "ABCD"},
           "correct_option": "A", "flags": []}
    _g, status, reasons = grade_and_status(rec)
    assert "source_missing_match_items" not in reasons
    assert status == "READY"
