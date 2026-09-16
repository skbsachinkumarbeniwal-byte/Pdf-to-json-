"""FINAL table refinement: deterministic pre-check -> Gemini visual
refinement -> deterministic fidelity validator (accept / review /
reject), medical-correction safety, audit report, and the before/after
regression contract (nothing unrelated may change).

All model calls are stubbed — no network, no API key.
"""
import json
from collections import Counter

import pytest

from qbank import refine_final as F

# the book vocabulary the classifications are judged against
WORDS = Counter({
    "layered": 4, "collagen": 3, "retraction": 6, "not": 80,
    "osteocalcin": 5, "analgesic": 4, "paracetamol": 3,
    "antipyretic": 4, "grade": 6,
})
VOCAB = (WORDS, Counter())


def md(rows):
    out = []
    for i, r in enumerate(rows):
        out.append("| " + " | ".join(r) + " |")
        if i == 0:
            out.append("|" + "---|" * len(r))
    return "\n".join(out)


def table(md_text, **kw):
    t = {"table_id": kw.pop("table_id", "T1"),
         "markdown": md_text,
         "source_pages": kw.pop("source_pages", [10]),
         "merged_continuation": kw.pop("merged_continuation", False),
         "header_deduplicated": kw.pop("header_deduplicated", False),
         "validation": {}}
    t.update(kw)
    if kw.pop("flagged", False):
        t["validation"]["table_qa"] = {"status": "REVIEW",
                                       "suspect_fragments": []}
    return t


def stub(answer):
    """A stubbed Gemini final-refiner fn + call log."""
    calls = []

    def fn(book, regions, t, md_text, context=None):
        calls.append({"regions": regions, "md": md_text,
                      "context": context})
        return answer
    return fn, calls


def run_stage(t, answer, tmp_path=None, ledger_name="ledger.jsonl",
              key="SUB|q|t", **kw):
    """Run the stage on one table record; returns (status, t, rows)."""
    fn, calls = stub(answer)
    ledger = None
    if tmp_path is not None:
        ledger = tmp_path / ledger_name
    st = F.final_refine_table(t, None, fn, ledger_path=ledger,
                              ledger_key=key, **kw)
    rows = []
    if ledger is not None and ledger.exists():
        rows = [json.loads(l) for l in
                ledger.read_text().splitlines() if l.strip()]
    return st, t, rows, calls


def refined(action="REFINED", md_text=None, changes=None, tid="T1"):
    return {"table_id": tid, "action": action,
            "refined_table": md_text,
            "changes": changes or []}


# --------------------------------------------------------------- 1+2+3
# content repairs: split word / glued word / punctuation spacing
# (all character-identical after whitespace normalisation)

def test_split_word_repair_accepted(tmp_path):
    before = md([["Term"], ["layere d collagen"]])
    after = md([["Term"], ["layered collagen"]])
    st, t, rows, _ = run_stage(
        table(before, flagged=True), refined(md_text=after),
        tmp_path, vocab=VOCAB)
    assert st == "accepted"
    assert t["markdown"] == after
    assert t["validation"]["pre_final_markdown"] == before
    assert t["validation"]["table_qa"]["refined_final_by_gemini"] is True
    ch = t["validation"]["final_refine"]["changes"]
    assert ch[0]["cell"] == "R2C1"
    assert ch[0]["kind"] == "word_restore"     # joined form is a book word
    assert rows[0]["status"] == "accepted"
    assert rows[0]["gemini_call"] is True


def test_glued_word_repair_accepted(tmp_path):
    before = md([["Sign"], ["retractionnot"]])
    after = md([["Sign"], ["retraction not"]])
    st, t, _, _ = run_stage(
        table(before, flagged=True), refined(md_text=after),
        tmp_path, vocab=VOCAB)
    assert st == "accepted"
    assert t["markdown"] == after
    ch = t["validation"]["final_refine"]["changes"]
    assert ch[0]["kind"] == "spacing"          # un-glue, both parts words


def test_punctuation_spacing_repair(tmp_path):
    before = md([["Note"], ["membrane.It is thin"]])
    after = md([["Note"], ["membrane. It is thin"]])
    st, t, _, _ = run_stage(
        table(before, flagged=True), refined(md_text=after),
        tmp_path, vocab=VOCAB)
    assert st == "accepted"
    assert t["markdown"] == after


# --------------------------------------------------------------- 4
def test_medical_spelling_corruption_keeps_evidence(tmp_path):
    before = md([["Marker"], ["osteocal cin level"]])
    after = md([["Marker"], ["osteocalcin level"]])
    changes = [{"cell": "R2C1", "before": "osteocal cin level",
                "after": "osteocalcin level",
                "reason": "mid-word wrap of a known term",
                "confidence": 0.9, "evidence": "visual+medical"}]
    st, t, rows, _ = run_stage(
        table(before, flagged=True),
        refined(md_text=after, changes=changes), tmp_path, vocab=VOCAB)
    assert st == "accepted"
    ch = t["validation"]["final_refine"]["changes"][0]
    assert ch["kind"] == "word_restore"
    # the model's evidence + confidence must survive into the audit
    assert ch["evidence"] == "visual+medical"
    assert ch["confidence"] == 0.9
    # miscounted cell ref (model said R2C1 — matches; but even a wrong
    # id must still attach by content): re-run with a wrong ref
    changes2 = [dict(changes[0], cell="R9C9")]
    st2, t2, _, _ = run_stage(
        table(before, flagged=True),
        refined(md_text=after, changes=changes2), vocab=VOCAB)
    assert st2 == "accepted"
    assert t2["validation"]["final_refine"]["changes"][0]["evidence"] \
        == "visual+medical"


# --------------------------------------------------------------- 5
def test_number_unit_repair():
    before = md([["Dose"], ["50 – 300 mg"]])
    after = md([["Dose"], ["50–300 mg"]])
    # same numbers, tighter range -> ACCEPT
    r = F.fidelity_compare(before, after, None, VOCAB)
    assert r["verdict"] == "ACCEPT"

    # broken number WITHOUT source-page evidence -> REJECT
    b2 = md([["Dose"], ["5 mg"]])
    a2 = md([["Dose"], ["50 mg"]])
    r2 = F.fidelity_compare(b2, a2, None, VOCAB)
    assert r2["verdict"] == "REJECT"
    assert r2["reject_reasons"] == ["changed_number:R2C1"]

    # WITH page-text evidence (page prints 50, not a bare 5) -> ACCEPT
    ev = F._page_number_evidence({10: "Give 50 mg daily."}, [10])
    r3 = F.fidelity_compare(b2, a2, ev, VOCAB)
    assert r3["verdict"] == "ACCEPT"
    assert r3["cells_changed"][0]["kind"] == "number_repair"
    assert r3["cells_changed"][0]["evidence"] == "page_text"
    # end-to-end through the stage: page_text flows into the validator
    st, t2, _, _ = run_stage(table(b2, flagged=True),
                             refined(md_text=a2), vocab=VOCAB,
                             page_text={10: "Give 50 mg daily."})
    assert st == "accepted" and t2["markdown"] == a2
    # ...and the same answer is rejected when the evidence is absent
    st2, t3, _, _ = run_stage(table(b2, flagged=True),
                              refined(md_text=a2), vocab=VOCAB)
    assert st2 == "rejected" and t3["markdown"] == b2


# --------------------------------------------------------------- 6
def test_long_cell_formatting(tmp_path):
    words = ("The drug acts by blocking the receptor on the target "
             "cell membrane and reducing the downstream signalling "
             "cascade in the tissue").split()
    long_text = " ".join(words)
    cut1, cut2 = 5, 10                        # WORD-boundary breaks
    after = md([["Mechanism"],
                [" ".join(words[:cut1]) + "<br>"
                 + " ".join(words[cut1:cut2]) + "<br>"
                 + " ".join(words[cut2:])]])
    before = md([["Mechanism"], [long_text]])
    # the pre-check flags a long cell (recorded on the ledger row)
    t0 = table(before)
    assert "long_cell" in F.precheck(t0, VOCAB)
    st, t, rows, calls = run_stage(t0, refined(md_text=after), tmp_path,
                                   vocab=VOCAB)
    assert st == "accepted"
    assert len(calls) == 1                    # all mode: the table is sent
    assert t["validation"]["final_refine"]["changes"][0]["kind"] \
        == "presentation"
    assert rows[0]["precheck"] == ["long_cell"]
    # a line break INSIDE a word is a defect the model introduced: it
    # is REFUSED (fatal), not queued for a human — the source's words
    # stay intact and the answer is dropped whole
    bad_after = md([["Mechanism"], [long_text[:10] + "<br>"
                                   + long_text[10:]]])
    assert bad_after.replace("<br>", " ") != long_text  # cuts mid-word
    st2, t2, rows2, _ = run_stage(
        table(before), refined(md_text=bad_after), tmp_path, vocab=VOCAB)
    assert st2 == "rejected"
    assert t2["markdown"] == before           # nothing applied
    assert "mid_word_break" in rows2[-1]["reject_reasons"][0]


# --------------------------------------------------------------- 7
def test_bullet_list_formatting(tmp_path):
    before = md([["Drug", "Use"],
                 ["paracetamol", "analgesic; antipyretic; antitussive"]])
    after = md([["Drug", "Use"],
                ["paracetamol",
                 "• analgesic<br>• antipyretic<br>• antitussive"]])
    t0 = table(before)
    assert "list_like_cell" in F.precheck(t0, VOCAB)
    st, t, _, _ = run_stage(t0, refined(md_text=after), tmp_path,
                            vocab=VOCAB)
    assert st == "accepted"
    assert t["markdown"] == after
    assert t["validation"]["final_refine"]["changes"][0]["kind"] \
        == "presentation"
    # bullet markers are presentation: the character stream is proven
    # identical by the validator, never by the model's word


# --------------------------------------------------------------- 8
def test_cross_page_table_stays_one_table(tmp_path):
    before = md([["Part", "Action"],
                 ["Roof", "dome shaped"],
                 ["Wall", "lateral part"]])
    t = table(before, table_id="002-T03", source_pages=[10, 11],
              merged_continuation=True, header_deduplicated=True,
              flagged=True)
    after = md([["Part", "Action"],
                ["Roof", "dome<br>shaped"],
                ["Wall", "lateral<br>part"]])
    st, t2, rows, _ = run_stage(t, refined(md_text=after, tid="002-T03"),
                                tmp_path, vocab=VOCAB)
    assert st == "accepted"
    # identity fields are preserved exactly
    assert t2["table_id"] == "002-T03"
    assert t2["source_pages"] == [10, 11]
    assert t2["merged_continuation"] is True
    assert t2["header_deduplicated"] is True
    assert rows[0]["cross_page"] is True
    # re-adding the repeated header on the continued page = structural
    # change (+1 row) -> REJECTED, one table never becomes two
    with_header = md([["Part", "Action"],
                      ["Part", "Action"],
                      ["Roof", "dome shaped"],
                      ["Wall", "lateral part"]])
    st2, t3, rows2, _ = run_stage(
        table(before, source_pages=[10, 11], merged_continuation=True,
              header_deduplicated=True, flagged=True),
        refined(md_text=with_header), tmp_path, vocab=VOCAB,
        ledger_name="ledger2.jsonl")
    assert st2 == "rejected"
    assert t3["markdown"] == before
    assert rows2[0]["reject_reasons"][0].startswith("structural_change")


# --------------------------------------------------------------- 9
def test_multi_column_table_cell_mapping(tmp_path):
    before = md([["A", "B", "C", "D"],
                 ["w", "x", "y", "z"],
                 ["k", "v", "osteo calcin", "m"]])
    after = md([["A", "B", "C", "D"],
                ["w", "x", "y", "z"],
                ["k", "v", "osteocalcin", "m"]])
    t = table(before, flagged=True)
    st, t2, _, _ = run_stage(t, refined(md_text=after), tmp_path,
                            vocab=VOCAB)
    assert st == "accepted"
    ch = t2["validation"]["final_refine"]["changes"]
    assert len(ch) == 1
    assert ch[0]["cell"] == "R3C3"            # mapped by position
    assert ch[0]["before"] == "osteo calcin"
    assert ch[0]["after"] == "osteocalcin"
    assert t2["markdown"] == after


# --------------------------------------------------------------- 10
def test_all_mode_sends_every_table(tmp_path):
    """QBANK_FINAL_REFINE=all (the default): even a table the
    pre-check finds completely clean is sent to Gemini. If the model
    finds nothing to fix it answers NO_CHANGE and the table ships
    byte-identical — the pre-check no longer gates the call, the
    safety still comes from the fidelity validator."""
    before = md([["Type", "Function", "Site"],
                 ["Kinase", "adds phosphate", "cytosol"],
                 ["Lipase", "cleaves ester", "gut"]])
    t = table(before)                         # unflagged, pre-check clean
    assert F.precheck(t, VOCAB) == []         # the check still runs...
    st, t2, rows, calls = run_stage(t, refined(action="NO_CHANGE",
                                               md_text=before), tmp_path,
                                    vocab=VOCAB)
    assert st == "no_change"
    assert len(calls) == 1                    # ...but the table is sent
    assert t2["markdown"] == before           # nothing was changed
    assert "final_refine" not in t2["validation"]
    assert rows[0]["status"] == "no_change"
    assert rows[0]["gemini_call"] is True
    assert rows[0]["precheck"] == []


# --------------------------------------------------------------- 11
def test_ambiguous_medical_correction_goes_review(tmp_path):
    before = md([["Finding"], ["dome or roof"]])
    # model is unsure which reading the print supports (L3)
    st, t, rows, _ = run_stage(
        table(before, flagged=True),
        refined(action="REVIEW", md_text=before), tmp_path, vocab=VOCAB)
    assert st == "review"
    assert t["markdown"] == before            # nothing applied
    assert t["validation"]["table_qa"]["status"] == "REVIEW"
    assert "refinement_review" in t["validation"]
    assert rows[0]["status"] == "review"
    # a REFINED answer whose repair is character-identical but whose
    # ONLY evidence is medical knowledge (L3) is routed to the human,
    # never auto-accepted
    before2 = md([["Marker"], ["osteocal cin level"]])
    after = md([["Marker"], ["osteocalcin level"]])
    st2, t2, _, _ = run_stage(
        table(before2, flagged=True),
        refined(md_text=after,
                changes=[{"cell": "R2C1", "before": "osteocal cin level",
                          "after": "osteocalcin level",
                          "confidence": 0.8, "evidence": "medical"}]),
        vocab=VOCAB)
    assert st2 == "review"
    assert t2["markdown"] == before2
    # ...the SAME answer with visual evidence is auto-accepted (L1)
    st4, t4, _, _ = run_stage(
        table(before2, flagged=True),
        refined(md_text=after,
                changes=[{"cell": "R2C1", "before": "osteocal cin level",
                          "after": "osteocalcin level",
                          "confidence": 0.9, "evidence": "visual"}]),
        vocab=VOCAB)
    assert st4 == "accepted"
    # low-confidence content repair -> review too
    st3, t3, _, _ = run_stage(
        table(before2, flagged=True),
        refined(md_text=after,
                changes=[{"cell": "R2C1", "before": "osteocal cin level",
                          "after": "osteocalcin level",
                          "confidence": 0.3, "evidence": "visual"}]),
        vocab=VOCAB)
    assert st3 == "review"


# --------------------------------------------------------------- 12+13+14
def test_hallucinated_addition_rejected(tmp_path):
    before = md([["Dose"], ["dose 5 mg daily"]])
    after = md([["Dose"], ["dose 5 mg daily also fatal"]])
    st, t, rows, _ = run_stage(
        table(before, flagged=True), refined(md_text=after), tmp_path,
        vocab=VOCAB)
    assert st == "rejected"
    assert t["markdown"] == before            # source kept
    assert "final_refine" not in t["validation"]
    assert rows[0]["reject_reasons"] == ["hallucinated_addition:R2C1"]


def test_ai_deletion_rejected(tmp_path):
    before = md([["Finding"], ["calcification in the bone wall"]])
    after = md([["Finding"], ["calcification in bone wall"]])
    st, t, rows, _ = run_stage(
        table(before, flagged=True), refined(md_text=after), tmp_path,
        vocab=VOCAB)
    assert st == "rejected"
    assert t["markdown"] == before
    assert rows[0]["reject_reasons"] == ["deletion:R2C1"]


def test_ai_number_modification_rejected(tmp_path):
    before = md([["Grade"], ["Grade 1 stenosis"]])
    after = md([["Grade"], ["Grade 2 stenosis"]])
    st, t, rows, _ = run_stage(
        table(before, flagged=True), refined(md_text=after), tmp_path,
        vocab=VOCAB, page_text={10: "unrelated page text"})
    assert st == "rejected"
    assert t["markdown"] == before
    assert rows[0]["reject_reasons"] == ["changed_number:R2C1"]


def test_substitution_rejected():
    """A 'medical correction' that swaps a character (Tetrology ->
    Tetralogy) is a content change: REJECT, source kept."""
    before = md([["Lesion"], ["Tetrology of Fallot"]])
    after = md([["Lesion"], ["Tetralogy of Fallot"]])
    r = F.fidelity_compare(before, after, None, VOCAB)
    assert r["verdict"] == "REJECT"
    assert r["reject_reasons"] == ["content_substitution:R2C1"]


def test_case_change_is_content_not_presentation(tmp_path):
    """Case is content in medical text: 'IgG' -> 'igg' is a
    corruption, never a spacing fix or presentation refinement."""
    before = md([["Marker"], ["IgG level raised"]])
    after = md([["Marker"], ["igg level raised"]])
    r = F.fidelity_compare(before, after, None, VOCAB)
    assert r["verdict"] == "REJECT"
    assert r["reject_reasons"] == ["content_substitution:R2C1"]
    # ...through the stage as well
    st, t, _, _ = run_stage(table(before, flagged=True),
                            refined(md_text=after), vocab=VOCAB)
    assert st == "rejected"
    assert t["markdown"] == before
    # a case-IDENTICAL presentation change still goes through
    ok = md([["Marker"], ["IgG level<br>raised"]])
    st2, t2, _, _ = run_stage(table(before, flagged=True),
                              refined(md_text=ok), vocab=VOCAB)
    assert st2 == "accepted"


# --------------------------------------------------------------- 15
# table-as-image duplication: the stage never creates an asset

def _run_mini(tmp, name, final_on, refine_stub, monkeypatch):
    from qbank import config, run, llm as llm_mod
    from test_mini_book import _build_book
    out = tmp / name
    monkeypatch.setenv("QBANK_FINAL_REFINE", "all" if final_on else "off")
    for attr, val in [("OUTPUT_ROOT", out), ("DATA_DIR", out / "data"),
                      ("ASSETS_DIR", out / "assets" / "questions"),
                      ("SPLIT_DIR", out / "split"),
                      ("SUBJECTS_DIR", out / "subjects"),
                      ("STATE_FILE", out / "state.json")]:
        monkeypatch.setattr(config, attr, val)
    pdf = tmp / f"{name}.pdf"
    _build_book(pdf)
    monkeypatch.setattr(llm_mod, "enabled", lambda: True)
    monkeypatch.setattr(llm_mod, "check_model", lambda *a, **k: True)
    monkeypatch.setattr(llm_mod, "transcriber",
                        lambda cache_dir=None, **k:
                        (lambda book, pg, box: None))
    monkeypatch.setattr(llm_mod, "verifier",
                        lambda cache_dir=None, **k:
                        (lambda book, pg, box, s: None))
    monkeypatch.setattr(llm_mod, "refiner",
                        lambda cache_dir=None, **k:
                        (lambda book, pgs, m: m))
    monkeypatch.setattr(llm_mod, "refine_final", refine_stub)
    return run.run_book(str(pdf), "TST", page_offset="auto", force=True,
                        output_root=out), out


def _refine_stub(accepted):
    def factory(cache_dir=None, **k):
        counter = k.get("counter") or {"n": 0}

        def refine(book, regions, t, m, context=None):
            counter["n"] += 1
            if accepted:
                return refined(
                    md_text=m.replace("adds phosphate",
                                      "adds<br>phosphate"),
                    tid=t.get("table_id"))
            return refined(action="NO_CHANGE", md_text=m,
                           tid=t.get("table_id"))
        return refine
    return factory


def test_refinement_creates_no_table_image(tmp_path, monkeypatch):
    """Running the final stage (with an accepted refinement) must leave
    the asset tree and image manifest byte-identical to a run without
    it: no table image, no duplicate asset, no manifest row. The
    mini-book's table is pre-check CLEAN, so this also proves that
    `all` mode sends even a clean table for refinement."""
    res_off, out_off = _run_mini(
        tmp_path, "out_off", False, _refine_stub(False), monkeypatch)
    # same run WITH the stage: force the model, accept one refinement
    res_on, out_on = _run_mini(
        tmp_path, "out_on", True, _refine_stub(True), monkeypatch)
    assert res_on["census_failures"] == []

    def assets(root):
        return sorted(str(p.relative_to(root)) for p in
                      (root / "assets" / "questions").rglob("*")
                      if p.is_file())
    assert assets(out_on) == assets(out_off)

    def manifest(root):
        rows = []
        for f in sorted((root / "split" / "TST").glob("*/image_manifest.jsonl")):
            rows += [json.loads(l) for l in f.read_text().splitlines()
                     if l.strip()]
        return rows
    assert manifest(out_on) == manifest(out_off)

    ch2 = out_on / "split" / "TST" / "TST-002"
    s = json.loads((ch2 / "solutions.jsonl").read_text().splitlines()[0])
    t = s["tables"][0]
    assert "adds<br>phosphate" in t["markdown"]   # refinement applied
    assert "pre_final_markdown" in t["validation"]
    assert "image" not in json.dumps(t).lower()   # no image field added
    comp = json.loads((ch2 / "chapter_completeness.json").read_text())
    assert comp["images"]["table_renders"] == 0
    assert comp["images"]["tables_suppressed"] >= 1
    assert comp["tables"]["final_refine"]["regression_ok"] is True


# --------------------------------------------------------------- 14
# before/after regression: NOTHING unrelated may change

def _split_rows(root):
    out = {}
    for nf in ("questions", "answers", "solutions"):
        out[nf] = []
        for f in sorted((root / "split" / "TST").glob(f"*/{nf}.jsonl")):
            out[nf] += [json.loads(l) for l in f.read_text().splitlines()
                        if l.strip()]
    return out


def test_full_pipeline_regression_no_unrelated_change(tmp_path,
                                                      monkeypatch):
    """The §14 contract, end to end on the synthetic book: with the
    final stage on (one accepted refinement) every count, id, page
    list, question text, option, answer and solution is identical to
    a run without it — only the refined table's markdown differs."""
    res_off, out_off = _run_mini(tmp_path, "rg_off", False,
                                 _refine_stub(False), monkeypatch)
    res_on, out_on = _run_mini(tmp_path, "rg_on", True,
                               _refine_stub(True), monkeypatch)
    assert res_off["total_questions"] == res_on["total_questions"] == 3

    a, b = _split_rows(out_off), _split_rows(out_on)
    for nf in ("questions", "answers", "solutions"):
        assert len(a[nf]) == len(b[nf])
        for ra, rb in zip(a[nf], b[nf]):
            ra2, rb2 = dict(ra), dict(rb)
            ta = {t["table_id"]: t for t in ra2.pop("tables", [])}
            tb = {t["table_id"]: t for t in rb2.pop("tables", [])}
            # everything OUTSIDE the tables field: byte-identical
            assert ra2 == rb2, nf
            assert set(ta) == set(tb)           # same table ids
            for tid in ta:
                xa, xb = dict(ta[tid]), dict(tb[tid])
                ma, mb = xa.pop("markdown"), xb.pop("markdown")
                xa.pop("validation", None)      # provenance may differ
                xb.pop("validation", None)
                assert xa == xb                 # id/pages/continuity
                if tid == "002-T01":
                    assert mb != ma             # the intended change
                    assert "adds<br>phosphate" in mb
                else:
                    assert ma == mb             # untouched tables
    # solutions outside the intended table change
    so = {r["q_id"]: r for r in a["solutions"]}
    sn = {r["q_id"]: r for r in b["solutions"]}
    assert so.keys() == sn.keys()
    for qid in so:
        assert so[qid]["solution_text"] == sn[qid]["solution_text"]
    # the stage's own regression verdict
    audit = json.loads((out_on / "data" /
                        "table_refinement_audit.json").read_text())
    assert audit["before_after_regression"]["ok"] is True
    c = audit["before_after_regression"]["counts"]
    assert c == {"questions": 3, "answers": 3, "solutions": 3,
                 "tables": 1, "cross_page_tables": 0}
    assert audit["tables_refined"] == 1
    assert audit["gemini_api_calls"] == 1


# --------------------------------------------------------------- 16
# report: all required audit fields + CLI

def test_audit_report_fields_and_cli(tmp_path, capsys, monkeypatch):
    before = md([["A"], ["layere d x"], ["dose 5 mg"], ["ok"]])
    # the ledger where write_audit() reads it back from
    ledger_name = str(F.LEDGER)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    t = table(before, table_id="T1", flagged=True)
    st, _, _, _ = run_stage(t, refined(
        md_text=before.replace("layere d x", "layered x")), tmp_path,
        vocab=VOCAB, ledger_name="data/" + ledger_name,
        key="SUB|q|T1")
    assert st == "accepted"
    t2 = table(before, table_id="T2", flagged=True)
    bad = refined(md_text=before.replace("dose 5 mg", "dose 5 mg extra"))
    st2, _, _, _ = run_stage(t2, bad, tmp_path, vocab=VOCAB,
                             ledger_name="data/" + ledger_name,
                             key="SUB|q|T2")
    assert st2 == "rejected"
    t3 = table(before, table_id="T3")           # clean, unflagged
    st3, _, _, _ = run_stage(t3, refined(action="NO_CHANGE",
                                         md_text=before, tid="T3"),
                             tmp_path, vocab=VOCAB,
                             ledger_name="data/" + ledger_name,
                             key="SUB|q|T3")
    assert st3 == "no_change"                  # all mode: sent, nothing to fix
    # all three rows landed under ONE ledger, one subject prefix
    ledger = tmp_path / "data" / ledger_name
    got = [json.loads(l) for l in ledger.read_text().splitlines()]
    assert {g["table_id"] for g in got} == {"T1", "T2", "T3"}
    rep = F.write_audit(tmp_path, "SUB", api_calls=3,
                        regression=[{"regression_ok": True,
                                     "counts": {"questions": 1}}])
    for field in ("total_tables", "tables_unchanged", "tables_refined",
                  "tables_rejected", "tables_review",
                  "spacing_repairs", "medical_spelling_repairs",
                  "number_repairs", "presentation_only_refinements",
                  "cross_page_tables_checked", "fidelity_violations",
                  "rejected_hallucinations", "rejected_deletions",
                  "rejected_number_changes", "render_page_waste_issues",
                  "gemini_api_calls", "before_after_regression"):
        assert field in rep, field
    assert rep["total_tables"] == 3
    assert rep["tables_refined"] == 1
    assert rep["tables_rejected"] == 1
    assert rep["tables_unchanged"] == 1
    assert rep["medical_spelling_repairs"] == 1
    assert rep["rejected_hallucinations"] == 1
    assert rep["gemini_api_calls"] == 3
    assert rep["before_after_regression"]["ok"] is True
    corr = rep["corrections"]
    assert len(corr) == 1
    for k in ("table_id", "cell", "before", "after", "reason",
              "evidence", "confidence"):
        assert k in corr[0]
    assert corr[0]["table_id"] == "T1"
    # load_audit is subject-scoped
    assert F.load_audit(tmp_path, "SUB") is not None
    assert F.load_audit(tmp_path, "OTHER") is None
    # the CLI prints the report
    from qbank import cli, config
    monkeypatch.setattr(config, "OUTPUT_ROOT", tmp_path)
    rc = cli.cmd_table_audit(type("Args", (), {"book": "SUB",
                                               "limit": 10})())
    assert rc == 0
    out = capsys.readouterr().out
    assert "total_tables: 3" in out
    assert "T1 R2C1" in out                     # the correction is shown
    rc2 = cli.cmd_table_audit(type("Args", (), {"book": "NOPE",
                                                "limit": 10})())
    assert rc2 == 1


# --------------------------------------------------------------- 17
# the Gemini call itself: crop image + metadata + cache + counter

def test_refine_final_sends_crop_metadata_and_caches(tmp_path,
                                                     monkeypatch):
    import json as _json
    from qbank import llm

    ANSWER = _json.dumps({"table_id": "T1", "action": "REFINED",
                          "refined_table": md([["A"], ["x y"]]),
                          "changes": []})
    posts = []

    def fake_post(url, payload, key):
        posts.append(payload)
        return {"candidates": [{"content": {"parts": [
            {"text": ANSWER}]}}]}
    monkeypatch.setattr(llm, "_post", fake_post)

    class FakePix:
        def tobytes(self, fmt):
            return b"png"

    class FakePage:
        def get_pixmap(self, clip, matrix):
            return FakePix()

    class FakeDoc:
        name = "fake.pdf"

        def __getitem__(self, i):
            return FakePage()

    class FakeBook:
        doc = FakeDoc()

    counter = {"n": 0}
    fn = llm.refine_final(cache_dir=tmp_path, key="K", counter=counter)
    t = {"table_id": "T1", "source_pages": [5, 6],
         "merged_continuation": True, "header_deduplicated": True}
    regions = [(5, (0, 0, 100, 200)), (6, (0, 0, 100, 200))]
    ans = fn(FakeBook(), regions, t, md([["A"], ["xy"]]),
             "surrounding prose")
    assert ans["action"] == "REFINED"
    payload = posts[0]
    parts = payload["contents"][0]["parts"]
    # ONE crop per contributing page (the source visual is the authority)
    imgs = [p for p in parts if "inline_data" in p]
    assert len(imgs) == 2
    ask = parts[-1]["text"]
    assert md([["A"], ["xy"]]) in ask          # current extraction sent
    assert '"table_id": "T1"' in ask           # metadata sent
    assert '"cross_page": true' in ask
    assert "ONE logical table" in ask          # span note
    assert "surrounding prose" in ask          # context sent
    assert counter["n"] == 1
    # second call: cache hit, no second API call
    ans2 = fn(FakeBook(), regions, t, md([["A"], ["xy"]]),
              "surrounding prose")
    assert ans2 == ans
    assert len(posts) == 1 and counter["n"] == 1
    # different content -> new cache key -> new call
    fn(FakeBook(), regions, t, md([["A"], ["x y"]]), "surrounding prose")
    assert len(posts) == 2
    # text-only mode (no book): no image part, conservative note present
    posts.clear()
    ans3 = fn(None, [], t, md([["A"], ["xy"]]), None)
    p3 = posts[0]["contents"][0]["parts"]
    assert all("inline_data" not in p for p in p3)
    assert "No page image" in p3[0]["text"]
    assert ans3["action"] == "REFINED"


def test_refine_final_prompt_contract():
    p = __import__("qbank.llm", fromlist=["REFINE_FINAL_PROMPT"]) \
        .REFINE_FINAL_PROMPT
    assert "NO_CHANGE" in p and "REFINED" in p and "REVIEW" in p
    assert "1:1" in p                            # cell mapping rule
    assert "Do NOT add" in p                     # no additions
    assert "NOT a licence" in p                  # medical knowledge rule
    assert "AUTHORITY" in p                      # crop is the authority
    assert '"evidence"' in p                     # structured output
    assert "50 – 300" in p and "50–300" in p     # the range example
    assert "osteocal cin" in p                   # the spelling example


def test_refine_final_off_switch(tmp_path, monkeypatch):
    """QBANK_FINAL_REFINE=off: the stage is skipped entirely — even a
    broken refiner must never be called."""
    from qbank import config, run, llm as llm_mod
    from test_mini_book import _build_book
    out = tmp_path / "out"
    monkeypatch.setenv("QBANK_FINAL_REFINE", "off")
    for attr, val in [("OUTPUT_ROOT", out), ("DATA_DIR", out / "data"),
                      ("ASSETS_DIR", out / "assets" / "questions"),
                      ("SPLIT_DIR", out / "split"),
                      ("SUBJECTS_DIR", out / "subjects"),
                      ("STATE_FILE", out / "state.json")]:
        monkeypatch.setattr(config, attr, val)
    pdf = tmp_path / "mini.pdf"
    _build_book(pdf)
    monkeypatch.setattr(llm_mod, "enabled", lambda: True)
    monkeypatch.setattr(llm_mod, "check_model", lambda *a, **k: True)
    monkeypatch.setattr(llm_mod, "transcriber",
                        lambda cache_dir=None, **k:
                        (lambda book, pg, box: None))
    monkeypatch.setattr(llm_mod, "verifier",
                        lambda cache_dir=None, **k:
                        (lambda book, pg, box, s: None))
    monkeypatch.setattr(llm_mod, "refiner",
                        lambda cache_dir=None, **k:
                        (lambda book, pgs, m: m))

    def boom_factory(cache_dir=None, **k):
        def _never(*a, **kw):
            raise AssertionError("final refiner must not be called")
        return _never
    monkeypatch.setattr(llm_mod, "refine_final", boom_factory)
    res = run.run_book(str(pdf), "TST", page_offset="auto", force=True,
                       output_root=out)
    assert res["census_failures"] == []
    assert not (out / "data" / F.LEDGER).exists()
    assert (out / "data" / F.AUDIT).exists()   # honest zero report


def test_flagged_mode_only_sends_flagged_tables(tmp_path):
    """only='flagged': a suspicious-but-unflagged table is NOT sent;
    a flagged one is."""
    before = md([["A"], ["layere d collagen x"]])
    clean_t = table(before, table_id="T1")      # unflagged
    fn, calls = stub(refined(md_text=before, tid="T1"))
    st = F.final_refine_table(clean_t, None, fn, only="flagged",
                              vocab=VOCAB)
    assert st == "skip" and calls == []
    flagged_t = table(before, table_id="T2", flagged=True)
    st2, t2, _, calls2 = run_stage(flagged_t, refined(md_text=before,
                                                      tid="T2"), vocab=VOCAB)
    assert st2 == "no_change" and len(calls2) == 1


def test_model_no_answer_and_invalid_answer(tmp_path):
    before = md([["A"], ["x y z"]])
    st, t, rows, _ = run_stage(table(before, flagged=True), None, tmp_path,
                               ledger_name="a.jsonl")
    assert st == "empty"
    assert t["markdown"] == before
    assert rows[0]["status"] == "empty"
    st2, t2, rows2, _ = run_stage(
        table(before, flagged=True),
        {"table_id": "T1", "action": "REFINED",
         "refined_table": "here is your table, hope it helps"},
        tmp_path, ledger_name="b.jsonl")
    assert st2 == "invalid"
    assert t2["markdown"] == before
    assert rows2[0]["status"] == "invalid"
    st3, t3, _, _ = run_stage(
        table(before, flagged=True),
        {"table_id": "T1", "action": "REFINED", "refined_table": None},
        vocab=VOCAB)
    assert st3 == "invalid"
    assert t3["markdown"] == before


def test_render_qa_finding_reported(tmp_path):
    long_cell = "x" * 300
    before = md([["A"], [long_cell]])
    after = md([["A"], [long_cell]])
    t = table(before, flagged=True)
    st, t2, rows, _ = run_stage(t, refined(action="NO_CHANGE",
                                           md_text=before), tmp_path)
    assert st == "no_change"
    assert "unreadably_wide_cell" in rows[0]["render_qa_before"]
    assert "unreadably_wide_cell" in rows[0]["render_qa_after"]
    assert F.render_qa(md([["A"], ["short"]])) == []


# ------------------------------------------- separator periods vs marks
# Live case (028-T02/T03): the printed "…apiospermum). Madurella … grisea
# E jeanselmei" was refined into bullets with "E. jeanselmei". The run-in
# separator period is presentation and may be dropped, but the period the
# model PUT on "E" is not printed anywhere: the extraction keeps deciding
# characters, and the verifier checks them.
_REF_BEFORE = ("Pseudallescheria boydii (Anamorph Scedosporium apiospermum)"
               ". Madurella mycetomatis Madurella grisea E jeanselmei "
               "Acremonium falciforme")
_REF_BULLETS = ("• Pseudallescheria boydii (Anamorph Scedosporium "
                "apiospermum)<br>• Madurella mycetomatis<br>• Madurella "
                "grisea<br>• E jeanselmei<br>• Acremonium falciforme")


def test_a_separator_period_may_become_bullets():
    rec = F._cell_change(_REF_BEFORE, _REF_BULLETS, None, None, None)
    assert rec is not None and rec["kind"] == "presentation"
    assert rec.get("fatal") is None


def test_a_period_moved_onto_another_word_is_fatal():
    rec = F._cell_change(_REF_BEFORE, _REF_BULLETS.replace("E jeanselmei",
                                                          "E. jeanselmei"),
                         None, None, None)
    assert rec is not None and rec.get("fatal"), rec
    assert rec["kind"] in ("mark_moved", "hallucinated_addition")


def test_an_abbreviation_period_is_still_content():
    rec = F._cell_change("Staphylococcus aureus and S. epidermidis",
                         "Staphylococcus aureus and S epidermidis",
                         None, None, None)
    assert rec is not None and rec.get("fatal") == "deletion", rec


def test_a_period_the_marks_rule_sees_when_it_moves():
    """Same characters, different word: a plain multiset check called
    this a `reorder` (a REVIEW that locked the export gate)."""
    rec = F._cell_change("Staph. aureus", "Staph aureus .", None, None, None)
    assert rec is not None
    assert rec.get("fatal") or rec["kind"] == "reorder"


# --- a <br> where the print has a run-in separator is not a cut word ----
def test_a_runin_separator_turned_into_a_line_break_is_not_mid_word():
    """The print runs list items together with a period
    ("…apiospermum).Madurella…"); the model bulleting them is
    restructuring, not a cut word. The counter demanded two LETTERS
    around the break and called the dot case mid_word_break, which threw
    a faithful rearrangement away (ch24 of the real book: 6 of them)."""
    from qbank.refine_final import _new_mid_word_breaks
    b = "Scedosporium apiospermum).Madurella mycetomatis"
    a = "Scedosporium apiospermum).<br>Madurella mycetomatis"
    assert _new_mid_word_breaks(b, a) == 0
    assert _new_mid_word_breaks("receptor", "recep<br>tor") == 1
    assert _new_mid_word_breaks("dome shaped", "dome<br>shaped") == 0


def test_an_invented_content_mark_is_fatal_in_the_final_stage():
    """052-T04 shipped "…of the epididymis.<br>These tubules…" while the
    print separates the sentences with nothing but a space."""
    from qbank.refine_final import _cell_change
    c = _cell_change(
        "Efferent ducts from the head open into the tubules of the "
        "epididymis These tubules coalesce",
        "Efferent ducts from the head open into the tubules of the "
        "epididymis.<br>These tubules coalesce", set(), None, None)
    assert c and c["kind"] == "mark_added" and c["fatal"] == "mark_added"
    # the printed run-in dot stays fine
    ok = _cell_change("apiospermum).Madurella", "apiospermum).<br>Madurella",
                      set(), None, None)
    assert ok and ok["kind"] == "presentation"
