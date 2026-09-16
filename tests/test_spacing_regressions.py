"""Regression tests for the evidence-based spacing/structure pipeline.

Every case here is a real failure class observed in the MARROW books
(ENT PDF when present). Tests skip gracefully when the book PDF is not
in the workspace; the synthetic mini-book tests cover the rest.

Rules under test:
  * a spacing change must carry book-internal evidence (vocabulary /
    pair counts) or be in the frozen visually-verified merge list;
  * legitimate multi-word terms and proper names must never change;
  * the fidelity envelope must reject unjustified whitespace in BOTH
    directions (model splits of real words, model glues of spaced
    book words) unless independently supported;
  * cross-page continuations merge into ONE logical table.
"""
from pathlib import Path

import pytest

from qbank import tables as T
from qbank.llm import merge_llm

PDF = Path("/home/user/uploads/ENT_ed8_CLEAN_corrected.pdf")
pytestmark = pytest.mark.skipif(
    not PDF.exists(), reason="ENT book PDF not in workspace")


@pytest.fixture(scope="module")
def book():
    import pymupdf
    from qbank.textlayer import Book
    return Book(pymupdf.open(str(PDF)))


@pytest.fixture(scope="module")
def vocab(book):
    return T.build_vocab(book)


def fp(t, vocab):
    """the build_box fixed-point loop, isolated"""
    for _ in range(6):
        parts, nf = T._repair_tokens(t.split(" "), vocab[0], vocab[1])
        t = " ".join(parts)
        if not nf:
            break
    return t


# ---------------- A/F/G: wrapped fragments merge with evidence --------

def test_wrap_fragment_surface(vocab):
    assert fp("medial su rface", vocab) == "medial surface"


def test_wrap_fragment_single_char_tail(vocab):
    # "bon e": join outranks both fragments by a wide margin
    assert fp("bon e", vocab) == "bone"


def test_wrap_fragment_sp_henoid(vocab):
    assert fp("Sp henoid", vocab) == "Sphenoid"


def test_wrap_fragment_or_bit(vocab):
    assert fp("or bit", vocab) == "orbit"


# ---------------- C/D: glued misprints unglue, real words stay --------

def test_unglue_retractionnot(vocab):
    assert fp("retractionnot", vocab) == "retraction not"


def test_unglue_theincus(vocab):
    assert fp("theincus", vocab) == "the incus"


def test_unglue_andhas(vocab):
    assert fp("andhas", vocab) == "and has"


def test_no_unglue_notable(vocab):
    assert fp("notable", vocab) == "notable"


def test_no_unglue_cannot(vocab):
    assert fp("cannot", vocab) == "cannot"


def test_no_merge_brain_stem(vocab):
    assert fp("brain stem", vocab) == "brain stem"


def test_no_merge_in_complete(vocab):
    assert fp("In complete", vocab) == "In complete"


# ---------------- recursive peel of glued multi-word blobs ------------

def test_peel_mega_blob(vocab):
    got = fp("Intracranialintraduraltumorwithoutinfiltration", vocab)
    assert got == "Intracranial intradural tumor without infiltration"


def test_peel_header_blob(vocab):
    assert fp("Causesofreferred", vocab) == "Causes of referred"


# ---------------- I/J: hyphen and punctuation -------------------------

def test_hyphen_kept(vocab):
    assert fp("retro- auricular", vocab) == "retro- auricular"


def test_comma_glue(vocab):
    assert fp("damage,fetal", vocab) == "damage, fetal"


def test_digit_range(vocab):
    assert fp("50 – 3 00 milliseconds", vocab) == "50–300 milliseconds"


# ---------------- zero-evidence cases: verified merge layer -----------

def test_verified_merge_osteocalcin(book, vocab):
    from collections import Counter
    for bx in book.page(411).table_boxes:
        bt = T.build_box(book, 411, bx, Counter(), vocab=vocab)
        txt = " ".join(c for r in bt.rows for c in r)
        if "osteocal" in txt:
            assert "osteocal cin" not in txt
            assert "osteocalcin" in txt
            return
    pytest.fail("osteocal cell not found on page 411")


def test_verified_map_stays_small():
    assert len(T._VERIFIED_MERGES) <= 20


def test_punct_attached_blob_still_repaired(book, vocab):
    # regression: the glued blob glued to trailing "(parasellar)" is not
    # alpha, so token rules cannot see it - the verified layer must
    from collections import Counter
    for bx in book.page(505).table_boxes:
        bt = T.build_box(book, 505, bx, Counter(), vocab=vocab)
        txt = " ".join(c for r in bt.rows for c in r)
        if "parasellar" in txt:
            assert "region with intracranial extradural" in txt
            assert "regionwithintracranialextradural" not in txt
            return
    pytest.fail("parasellar row not found on page 505")


# ---------------- fidelity envelope, both directions ------------------

def test_envelope_rejects_model_split_of_real_word(vocab):
    det = [["medial surface"]]
    out, n = merge_llm(det, [["medial su rface"]], vocab)
    assert out == det and n == 0


def test_envelope_rejects_model_glue_of_spaced_words(vocab):
    det = [["long process of the incus"]]
    out, n = merge_llm(det, [["long process of theincus"]], vocab)
    assert out == det and n == 0





def test_envelope_accepts_model_join_when_glued_form_is_book_word(vocab):
    out, n = merge_llm([["the do me of the jugular bulb"]],
                       [["the dome of the jugular bulb"]], vocab)
    assert n == 1 and "dome" in out[0][0]


def test_envelope_accepts_fragment_rewording(vocab):
    out, n = merge_llm([["bonedestruct ion"]],
                       [["bone destruction"]], vocab)
    assert n == 1 and out[0][0] == "bone destruction"


def test_envelope_char_safety(vocab):
    # any character difference must veto the model cell
    out, n = merge_llm([["osteocalcin"]], [["osteocalcinX"]], vocab)
    assert out == [["osteocalcin"]] and n == 0


# ---------------- QA suspects: no false positives ---------------------

def test_qa_no_flag_brain_stem(vocab):
    assert not T.qa_suspects([["brain stem"]], vocab[0])


def test_qa_no_flag_proper_names(vocab):
    assert not T.qa_suspects(
        [["Killian's incision", "Freer's incision",
          "Mucoperichondrial flap"]], vocab[0])


def test_qa_no_flag_antihelix(vocab):
    assert not T.qa_suspects([["Concha and antihelix Some supply"]],
                             vocab[0])


def test_qa_no_flag_corrected_osteocalcin(vocab):
    # the corrected form is not a suspect; the split form is repaired
    # deterministically (verified-merge layer), so QA stays quiet
    assert not T.qa_suspects([["osteocalcin"]], vocab[0])


def test_qa_flags_func_collision(vocab):
    # function word glued into a content word: "or"+"optic"
    assert T.qa_suspects([["oroptic chiasm"]], vocab[0])


# ---------------- cross-page: ONE logical table -----------------------

def test_cross_page_single_logical_table(book, vocab):
    ct = T.ChapterTables(book, 2, 26, 29, vocab, None, None)
    p28 = book.page(28).table_boxes
    lt = ct.lookup(28, p28[0])
    assert lt is not None and lt.cross_page
    assert lt.source_pages == [28, 29]
    # the two physical boxes map to the SAME logical table
    assert ct.lookup(29, book.page(29).table_boxes[0]) is lt


def test_unrelated_tables_not_merged(book, vocab):
    # 014: single-page tables stay single-page
    ct = T.ChapterTables(book, 14, 231, 231, vocab, None, None)
    lt = ct.lookup(231, book.page(231).table_boxes[0])
    assert lt is not None and not lt.cross_page


# ---------------- blind-book (ANA) evidence ---------------------------
# Real failures observed on a book that played NO part in designing or
# testing the rules (MARROW ED8 Anatomy, raw text layer).

ANA_PDF = "/home/user/ana_book.pdf"
needs_ana = pytest.mark.skipif(not Path(ANA_PDF).exists(),
                               reason="ANA blind book absent")


@pytest.fixture(scope="module")
def ana():
    if not Path(ANA_PDF).exists():
        pytest.skip("ANA blind book absent")
    from qbank.textlayer import Book
    b = Book(ANA_PDF)
    return b, T.build_vocab(b)


@needs_ana
def test_ana_func_prefix_unglue(ana):
    # glued function prefixes printed <=2x, survivor strongly evidenced
    w, p = ana[1]
    assert T._repair_tokens(["oftouch"], w, p)[0] == ["of touch"]
    assert T._repair_tokens(["tomotor"], w, p)[0] == ["to motor"]
    assert T._repair_tokens(["ofinternal"], w, p)[0] == ["of internal"]
    assert T._repair_tokens(["ofpain"], w, p)[0] == ["of pain"]


@needs_ana
def test_ana_printed_word_never_peeled(ana):
    # "everywhere" is printed once as one word: not a glued artifact
    w, p = ana[1]
    assert T._repair_tokens(["everywhere"], w, p)[0] == ["everywhere"]


@needs_ana
def test_ana_variant_copies_not_merged(ana):
    # p322/323 print two variant COPIES of the same table (repeated
    # header AND repeated first row): two logical tables, not one
    b, v = ana
    ct = T.ChapterTables(b, 17, 322, 323, v, None, None)
    lt1 = ct.lookup(322, b.page(322).table_boxes[0])
    lt2 = ct.lookup(323, b.page(323).table_boxes[0])
    assert lt1 is not None and lt2 is not None
    assert lt1 is not lt2 and not lt1.cross_page


@needs_ana
def test_ana_single_letter_fragment_merge(ana):
    # "t he" printed by the book's corruption; join overwhelmingly
    # common, split spacing printed nowhere
    w, p = ana[1]
    assert T._repair_tokens(["t", "he"], w, p)[0] == ["the"]
    assert T._repair_tokens(["a", "he"], w, p)[0] == ["a", "he"]


@needs_ana
def test_envelope_accepts_model_unglue_of_rare_artifact(ana):
    # det token printed <=2x is a glued artifact: the model's un-glue
    # wins even without pair evidence (Gemini owns table spacing)
    det = [["extends the whole of the medulla oblongatatill the 2nd"]]
    model = [["extends the whole of the medulla oblongata till the 2nd"]]
    out, n = merge_llm(det, model, ana[1])
    assert n == 1 and out[0][0] == model[0][0]

