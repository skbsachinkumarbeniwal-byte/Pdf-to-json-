"""Regression tests for scripts/verify_extraction.py.

The verifier is the independent check that a finished output tree matches
the source PDF. Three properties must never regress:

  * presentation the pipeline is ALLOWED to add (<br>, bullets, the ";"
    and "," separators the typesetter ran together) is erased on BOTH
    sides of the table-cell check;
  * CONTENT differences are still fatal — a changed word, a changed
    number, an added word, reordered words, or a cell that is simply not
    printed must all fail, no matter which acceptance path is used;
  * a cell reconstructed by joining the printed lines that carried it
    still verifies, and an earlier line that only PREFIX-matches cannot
    sabotage the exact match.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_extraction", ROOT / "scripts" / "verify_extraction.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ve = _load_verifier()


# ---------------------------------------------------------------- norm_cell

@pytest.mark.parametrize("printed, shipped", [
    # 028 mycetoma row: the page prints the items run together, the
    # pipeline ships them as a bulleted <br> list.
    ("Pseudallescheria boydii (Anamorph Scedosporium apiospermum)."
     "Madurella mycetomatisMadurella grisea",
     "• Pseudallescheria boydii (Anamorph Scedosporium apiospermum)."
     "<br>• Madurella mycetomatis<br>• Madurella grisea"),
    # "; "-separated lists the typesetter ran together
    ("Otitis media in childrenExacerbation of COPDPuerperal sepsis",
     "Otitis media in children; Exacerbation of COPD; Puerperal sepsis"),
    ("conjunctivitis /", "conjunctivitis/"),
    ("12 mg", "12mg"),                      # spacing is not content
])
def test_norm_cell_erases_presentation_only(printed, shipped):
    assert ve.norm_cell(printed) == ve.norm_cell(shipped)


@pytest.mark.parametrize("printed, shipped", [
    ("Madurella grisea", "Madurella griseaa"),      # added letter
    ("Madurella grisea", "Madurella grisa"),        # dropped letter
    ("58", "59"),                                   # changed number
    ("Sporothrix schenckii", "Schenckii sporothrix"),       # reordered
])
def test_norm_cell_keeps_content(printed, shipped):
    assert ve.norm_cell(printed) != ve.norm_cell(shipped)


def test_verifier_cannot_see_word_boundary_differences():
    """KNOWN BLIND SPOT, by design of norm().

    norm() erases ALL whitespace, so "in completely" and "incompletely"
    normalise identically — this verifier cannot catch a split-word
    corruption. That class is owned elsewhere, by the deterministic
    restore path in qbank/refine.py (restore_split_words /
    segmentation_weakens) and by tools/segmentation_audit.py, both of
    which are assert-tested in test_refine.py. If this test ever starts
    failing, norm() grew whitespace sensitivity and those tools'
    expectations need a re-read.
    """
    assert ve.norm_cell("incompletely immunized") == \
        ve.norm_cell("in completely immunized")
    space_preserving = lambda s: "".join(s.split())  # noqa: E731
    assert ve.norm_cell("incompletely immunized") == \
        space_preserving("in completely immunized")   # same blindness
    assert ve.norm_cell("in completely immunized") != \
        ve.norm_cell("incompletely immunization")     # real change: caught


def test_markup_strip_does_not_eat_real_angle_text():
    # a genuine "<" comparison is not markup and must not be erased
    assert ve.norm_cell("CD4 < 200") == ve.norm_cell("CD4 < 200")


# ------------------------------------------------------------ _cell_in_lines

def test_cell_in_lines_ordered_concatenation():
    lines = [ve.norm_cell("Madurella"), ve.norm_cell("mycetomatis")]
    assert ve._cell_in_lines(ve.norm_cell("Madurella mycetomatis"), lines)


def test_cell_in_lines_rejects_reordered_content():
    lines = [ve.norm_cell("Madurella"), ve.norm_cell("mycetomatis")]
    assert not ve._cell_in_lines(ve.norm_cell("mycetomatis Madurella"), lines)


def test_cell_in_lines_prefix_line_does_not_sabotage_exact_match():
    # "T1" prefix-matches "T1a" — the exact single-line match must win
    lines = [ve.norm_cell("T1"), ve.norm_cell("T1a"), ve.norm_cell("T2a")]
    assert ve._cell_in_lines(ve.norm_cell("T1a"), lines)


def test_cell_in_lines_ignores_intervening_lines():
    lines = [ve.norm_cell("Madurella"), ve.norm_cell("noise line"),
             ve.norm_cell("mycetomatis")]
    assert ve._cell_in_lines(ve.norm_cell("Madurella mycetomatis"), lines)


# ------------------------------------------------------------------ _table_ok

def _ok(md, line_sets, zone_texts=()):
    return ve._table_ok(md, line_sets, zone_texts)


def test_table_ok_passes_for_markup_only_difference():
    md = ("| Subcutaneous mycoses | Causative agent/s |\n"
          "|---|---|\n"
          "| Mycetoma | • Madurella mycetomatis<br>• Madurella grisea |")
    printed = ve.norm_cell("Madurella mycetomatisMadurella grisea")
    assert _ok(md, [[ve.norm_cell("Subcutaneous mycoses"),
                     ve.norm_cell("Causative agent/s"),
                     ve.norm_cell("Mycetoma"), printed]])


def test_table_ok_fails_on_hallucinated_word():
    md = ("| Causative agent/s |\n|---|---|\n"
          "| • Madurella grisea |")
    printed = ve.norm_cell("Madurella mycetomatis")
    assert not _ok(md, [[printed]])


def test_table_ok_fails_on_changed_number():
    md = ("| Proportion method | 59 |\n|---|---|")
    printed = ve.norm_cell("Proportion method58")
    assert not _ok(md, [[printed]])


def test_table_ok_separator_row_is_not_a_cell():
    """The |---|---| row carries no content, so it must never be treated
    as a cell that has to be found in the PDF."""
    md = "| Group | Amount |\n|---|---|\n| A | 5 mg |"
    assert _ok(md, [[ve.norm_cell("Group"), ve.norm_cell("Amount"),
                     ve.norm_cell("A"), ve.norm_cell("5 mg")]])
    # ...and it really is a no-op: the same table without it still fails
    # when a content cell is unprinted
    assert not _ok("| Group | Amount |\n| A | 7 mg |",
                   [[ve.norm_cell("Group"), ve.norm_cell("Amount"),
                     ve.norm_cell("A"), ve.norm_cell("5 mg")]])


def test_table_ok_accepts_recut_cell_via_zone_text():
    # page prints ONE merged cell; refinement splits it into three rows —
    # the letters are all there, in order, so the substring path verifies
    md = ("| Helminth | Host |\n|---|---|\n"
          "| Wuchereria bancrofti | mosquito |\n"
          "| Brugia malayi | mosquito |\n"
          "| Loa loa | fly |")
    zone_q = ve.norm_cell("HelminthHostWuchereria bancroftimosquito"
                          "Brugia malayimosquitoLoa loafly")
    assert _ok(md, [], (zone_q,))


def test_table_ok_rejects_cell_printed_nowhere():
    md = ("| Helminth |\n|---|---|\n| Brugia malayi |")
    zone_q = ve.norm_cell("Wuchereria bancrofti Loa loa Onchocerca volvulus")
    assert not _ok(md, [], (zone_q,))


def test_table_ok_requires_every_cell_not_just_one():
    md = ("| Helminth | Host |\n|---|---|\n"
          "| Brugia malayi | gorilla |")
    zone_q = ve.norm_cell("HelminthHostBrugia malayimosquito")
    assert not _ok(md, [], (zone_q,))


# ------------------------------------------------------------ _page_line_sets

@pytest.fixture(scope="module")
def tiny_pdf(tmp_path_factory):
    """2-page PDF: a cell split across two printed baselines, a run-together
    list, and a decoy line that only prefix-matches."""
    path = tmp_path_factory.mktemp("verify") / "tiny.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 100), "Causative agent/s")
    p1.insert_text((72, 120), "Madurella")
    p1.insert_text((72, 140), "mycetomatis")
    p1.insert_text((72, 160), "T1")
    p1.insert_text((72, 180), "T1a Meningitis")
    p2 = doc.new_page()
    p2.insert_text((72, 100), "Exophiala jeanselmeiPhialophora richardsiae")
    doc.save(str(path))
    doc.close()
    return path


def test_page_line_sets_joins_printed_lines(tiny_pdf):
    doc = pymupdf.open(str(tiny_pdf))
    lines, texts = ve._page_line_sets(doc, [1, 2], 0)
    assert ve._table_ok("| Madurella mycetomatis |", lines, tuple(texts))
    assert ve._table_ok("| • Exophiala jeanselmei<br>• Phialophora richardsiae |",
                        lines, tuple(texts))
    doc.close()


def test_page_line_sets_rejects_unprinted_cell(tiny_pdf):
    doc = pymupdf.open(str(tiny_pdf))
    lines, texts = ve._page_line_sets(doc, [1, 2], 0)
    assert not ve._table_ok("| Madurella grisea |", lines, tuple(texts))
    doc.close()


def test_page_line_sets_honours_page_offset(tiny_pdf):
    """A table declaring book page 1 lands on file page 1 + offset."""
    doc = pymupdf.open(str(tiny_pdf))
    lines, texts = ve._page_line_sets(doc, [1], 1)      # -> file page 2
    assert ve._table_ok("| Exophiala jeanselmei Phialophora richardsiae |",
                        lines, tuple(texts))
    assert not ve._table_ok("| Madurella mycetomatis |", lines, tuple(texts))
    doc.close()


def test_page_line_sets_skips_out_of_range_pages(tiny_pdf):
    doc = pymupdf.open(str(tiny_pdf))
    lines, texts = ve._page_line_sets(doc, [99, 1], 0)
    assert ve._table_ok("| Madurella mycetomatis |", lines, tuple(texts))
    doc.close()
