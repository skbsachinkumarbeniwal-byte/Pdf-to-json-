"""Regression tests for the bugs found while re-running the real
Microbiology ED8 book through the pipeline.

  1. FOOTER vs ANSWER-KEY ROW — page 14 of the real book prints the
     answer-key row `14   d` with its number in the bottom band of the
     page. The old "text == page number and low on the page" test threw
     that row away together with the footer, so the printed key row for
     question 14 vanished and chapter 1 failed its census (K=22 Q=23).
     The footer rule now also requires the line to be CENTRED and ALONE
     on its baseline.
  2. PUNCTUATION SPACING — the corrected text layer glues/splits
     punctuation ("time,e.g-", "host ’s"). Those repairs used to be
     left to Gemini, whose (correct) proposal the fidelity validator
     then rejected as a character substitution; they now happen
     deterministically and are counted.
  3. GEMINI FAILURE STATE — a chapter whose tables came back with no
     answer is remembered so the next run retries it instead of
     resuming past a half-processed book.
"""

from __future__ import annotations

import pymupdf

from qbank import state as state_mod
from qbank.tables import punct_spacing
from qbank.textlayer import Book, word_rows


def _page_with_key_row_that_looks_like_the_footer(doc, file_page: int):
    """A 14-page document whose LAST page reproduces the real page-14
    geometry (the printed number has to match the file page number, or
    the footer rule never even considers the line)."""
    for _ in range(file_page - 1):
        doc.new_page(width=612, height=792)
    p = doc.new_page(width=612, height=792)
    # a question header so the chapter scan has something to chew on
    p.insert_text((56, 60), "Question 14:", fontsize=11)
    p.insert_text((76, 90), "Stem text.", fontsize=10)
    p.insert_text((56, 400), "Answer Key", fontsize=11)
    p.insert_text((190, 430), "Question No.", fontsize=9)
    p.insert_text((370, 430), "Correct Option", fontsize=9)
    # the answer-key row: number + letter on ONE baseline, low on the
    # page, and the number string equals this page's printed number
    p.insert_text((228.4, 729), "14", fontsize=10)
    p.insert_text((375.2, 729), "d", fontsize=10)
    # the real footer: centred, alone on its baseline
    p.insert_text((300, 775), "14", fontsize=10)
    return doc


def test_answer_key_row_survives_the_footer_rule(tmp_path):
    pdf = tmp_path / "p14.pdf"
    doc = pymupdf.open()
    _page_with_key_row_that_looks_like_the_footer(doc, 14)
    doc.save(str(pdf))
    doc.close()

    book = Book(str(pdf))
    book.set_offset(0)
    pd = book.page(14)
    kept_texts = [l.text.strip() for l in pd.lines]
    assert "Answer Key" in kept_texts
    assert "14" in kept_texts, "the printed key-row number was dropped"
    assert "d" in kept_texts
    # the centred footer IS furniture: dropped, and audited as dropped
    assert len(pd.footers) == 1, [l.text for l in pd.footers]
    assert pd.footers[0].text.strip() == "14"
    assert pd.footers[0].y0 > 750, pd.footers[0]      # the real footer

    # and the key row must be reconstructible as a key row
    rows = [w for w in word_rows(pd)
            if [x.text.strip() for x in w] == ["14", "d"]]
    assert rows, [ [x.text for x in w] for w in word_rows(pd) ]
    book.close()


def test_footer_kept_audit_records_what_looked_like_content(tmp_path):
    pdf = tmp_path / "p14b.pdf"
    doc = pymupdf.open()
    _page_with_key_row_that_looks_like_the_footer(doc, 14)
    doc.save(str(pdf))
    doc.close()
    book = Book(str(pdf))
    book.set_offset(0)
    pd = book.page(14)
    kept = pd.footer_kept
    assert kept, "the resurrected line must be audited"
    assert kept[0]["text"] == "14"
    assert "baseline" in kept[0]["why"]
    book.close()


# ---- punctuation spacing (deterministic, counted) ----------------------

def test_punct_spacing_repairs_only_spacing():
    cases = {
        "at thesame time,e.g- post-exposure":
            ("at thesame time, e.g- post-exposure", 1),
        "antibody,colostrum": ("antibody, colostrum", 1),
        "Actively produced by the host \u2019s immune system":
            ("Actively produced by the host\u2019s immune system", 1),
        "1 , 2 & 5": ("1, 2 & 5", 1),
        "Note:see below": ("Note: see below", 1),
    }
    for src, (want, n) in cases.items():
        got, k = punct_spacing(src)
        assert got == want, (src, got)
        assert k == n, (src, k)
        # content can never change: only spaces are added/removed
        assert got.replace(" ", "") == src.replace(" ", "")


def test_punct_spacing_leaves_codes_and_numbers_alone():
    for text in ("C2,C3", "1,000 patients", "O157:H7", "A-4,B-3,C-2,D-1",
                 "IgG3: no change", "Ratio 1:2"):
        got, n = punct_spacing(text)
        assert (got, n) == (text, 0), (text, got, n)


# ---- LLM failure state / retry -----------------------------------------

def test_llm_failure_state_is_recorded_and_cleared(tmp_path, monkeypatch):
    from qbank import config
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    st = {"pdf_progress": {}}
    state_mod.note_llm_failures(st, "MIC", "MIC-001",
                                {"rearrange_no_answer": 2})
    prog = state_mod.progress(st, "MIC")
    assert prog["chapters_llm_failed"]["MIC-001"] == {
        "rearrange_no_answer": 2}
    # a later successful re-run clears the marker
    state_mod.note_llm_failures(st, "MIC", "MIC-001", {})
    assert "MIC-001" not in prog["chapters_llm_failed"]
    # zero-valued counters are not failures
    state_mod.note_llm_failures(st, "MIC", "MIC-002",
                                {"rearrange_no_answer": 0})
    assert prog["chapters_llm_failed"] == {}


# ---- empty cells are legitimate --------------------------------------

def test_table_with_empty_cells_is_a_valid_answer():
    """The old shape rule refused every table containing an empty cell
    and reported 'not an even pipe-markdown table' — but the real book's
    tables carry empty header corners and footnote rows (real case:
    MIC-009-T01, a 4-column table whose footnote row fills one cell)."""
    from qbank.refine import md_table_shape, valid_rearrangement

    footnote_row = ("| Test | Substance detected | Sensitivity | Specificity |\n"
                    "|---|---|---|---|\n"
                    "| CCNA | Free toxins | High | High |\n"
                    "| *Must be combined with a toxin test | | | |")
    assert md_table_shape(footnote_row) == 4
    assert valid_rearrangement(footnote_row)

    corner = ("|  | Gram positive | Gram negative |\n|---|---|---|\n"
              "| Teichoic acid | Present | Absent |")
    assert md_table_shape(corner) == 3

    # a table without a consistent column count is still refused
    assert md_table_shape("| A | B |\n|---|---|\n| x | y |\n| only one |") is None
    assert md_table_shape("just prose, no table") is None


# ---- the final validator must also catch a bad word boundary -----------

def test_final_validator_rejects_a_boundary_the_book_contradicts():
    """Same corruption as the rearrange envelope test, one stage later:
    the final pass may not turn the printed "incompletely immunized"
    into "in completely immunized" (the letters are identical, so the
    character-identity check alone cannot see it)."""
    from collections import Counter
    from qbank import refine_final as rf
    words = Counter({"in": 1500, "completely": 1, "incompletely": 2,
                     "immunized": 6})
    rec = rf._cell_change("Meningitis incompletely immunized infants",
                          "Meningitis in completely immunized infants",
                          words, None, None)
    assert rec["kind"] == "segmentation_change"
    assert rec["fatal"] == "segmentation_change"
    assert "incompletely -> in completely" in rec["detail"]["boundaries"]


def test_final_validator_still_accepts_a_supported_join():
    """The evidence-positive direction keeps flowing (this is how the
    printed form is restored)."""
    from collections import Counter
    from qbank import refine_final as rf
    words = Counter({"in": 1500, "completely": 1, "incompletely": 2,
                     "immunized": 6})
    rec = rf._cell_change("Meningitis in completely immunized infants",
                          "Meningitis incompletely immunized infants",
                          words, None, None)
    assert rec and not rec.get("fatal")
    assert rec["kind"] in ("spacing", "word_restore", "presentation")
