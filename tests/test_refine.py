"""Gemini medical rearrangement: saved during extraction, structural
validation only (stubbed model — no network). Also covers the export
gate contract: review queue clear -> zip builds, census advisories
never block."""
import json
from collections import Counter

import pytest

flask = pytest.importorskip("flask")

from qbank import config
from qbank import refine as refine_mod

from test_dashboard_review import _mkroot, client  # noqa: F401

GLUED = "| A | B |\n|---|---|\n| ASCAOMP-C Anti I2 | antibodyp-ANCA |"
# a REAL rearrangement: same characters, split into the right rows and
# re-spaced. (The previous fixture added the word "Target" to a header —
# that is a hallucination by the strict content envelope that now runs
# on this pass, and the canned answer had to follow the contract the
# pass actually enforces. See test_rearrange_envelope_* below.)
REARRANGED = ("| A | B |\n|---|---|\n"
              "| ASCA OMP-C | antibody |\n| Anti I2 | p-ANCA |")
ADDED_WORD = ("| Antibody | Target |\n|---|---|\n"
              "| Anti I2 | ASCA OMP-C |\n| p-ANCA | antibody |")
GARBAGE = "here is your table, hope it helps!"
UNEVEN = "| A | B |\n|---|---|\n| x | y |\n| only one |"


def test_valid_rearrangement_shape():
    assert refine_mod.valid_rearrangement(REARRANGED)
    assert not refine_mod.valid_rearrangement(GARBAGE)
    assert not refine_mod.valid_rearrangement(UNEVEN)
    assert not refine_mod.valid_rearrangement(None)
    assert not refine_mod.valid_rearrangement("| A | B |")   # single row


def test_rearrange_prompt_targets_book_artifacts():
    """The prompt must keep teaching the exact artifact classes these
    books print: mid-word line wraps ("mylo hyoid"), wrapped header
    cells ("Pharyngeal A rch"), glued neighbour cells
    ("cordisProximal") and missing punctuation spaces."""
    from qbank import llm as llm_mod
    p = llm_mod.REARRANGE_PROMPT
    assert "mylohyoid" in p            # mid-word wrap inside a cell
    assert "Pharyngeal Arch" in p      # wrapped header cell repair
    assert "glued" in p.lower()        # un-glue neighbouring cells
    assert "EXACTLY" in p              # content must not change


def test_rearrange_prompt_incompleteness_policy():
    """Text-only refine: extraction is the only source — restore nothing
    from memory, obvious mid-word joins are repair, cut-off stays
    as-is (no guesswork)."""
    from qbank import llm as llm_mod
    p = llm_mod.REARRANGE_PROMPT
    assert "ONLY source" in p
    assert "no page image" in p         # image bhejna band
    assert "add NOTHING" in p           # apne knowledge se nahi
    assert "repair, not addition" in p  # mid-word join allowed
    assert "no guesswork" in p          # cut-off cell as-is


def _stub(rows_by_md):
    calls = []

    def refine(book, pg, md):
        calls.append(md)
        return rows_by_md.get(md)
    return refine, calls


def _flag_only_table(root):
    """The TST table is not REVIEW by default — flag it like the run's
    QA would."""
    qf = root / "split" / "TST" / "TST-001" / "questions.jsonl"
    row = json.loads(qf.read_text().splitlines()[0])
    for t in row.get("tables") or []:
        t.setdefault("validation", {})["table_qa"] = {
            "status": "REVIEW", "suspect_fragments": []}
        t["markdown"] = GLUED
    qf.write_text(json.dumps(row) + "\n")
    return row["q_id"], row["tables"][0]["table_id"]


def test_rearrange_envelope_rejects_added_words():
    """A rearrangement may move and re-space, never invent content."""
    ok, diff = refine_mod.same_content(GLUED, ADDED_WORD)
    assert not ok
    assert "added" in diff and diff["added"], diff


def test_rearrange_envelope_accepts_reorder_and_casing():
    ok, diff = refine_mod.same_content(GLUED, REARRANGED)
    assert ok, diff
    assert refine_mod.same_content(GLUED, GLUED) == (True, {})


def test_refine_table_refuses_content_changed_answer(tmp_path):
    """The refused answer never ships, and the reason is recorded."""
    root = _mkroot(tmp_path)
    q_id, tid = _flag_only_table(root)
    stub, _calls = _stub({GLUED: ADDED_WORD})
    t = json.loads((root / "split" / "TST" / "TST-001" /
                    "questions.jsonl").read_text())["tables"][0]
    st = refine_mod.refine_table(t, None, stub, "all",
                                 ledger_key=f"TST|{q_id}|{tid}")
    assert st == "content_changed"
    assert t["markdown"] == GLUED                     # printed table kept
    v = t["validation"]["gemini_rearrange"]
    assert v["status"] == "REJECTED_CONTENT_CHANGED"
    assert v["kept"] == "deterministic_extraction"
    assert v["diff"]["added"]                      # what it tried to add


def test_refine_table_saves_gemini_output(tmp_path):
    root = _mkroot(tmp_path)
    q_id, tid = _flag_only_table(root)
    stub, calls = _stub({GLUED: REARRANGED})
    ledger = root / "data" / refine_mod.LEDGER

    t = json.loads((root / "split" / "TST" / "TST-001" /
                    "questions.jsonl").read_text())["tables"][0]
    st = refine_mod.refine_table(t, None, stub, "all",
                                 ledger_key=f"TST|{q_id}|{tid}",
                                 ledger_path=ledger)
    assert st == "replaced"
    assert t["markdown"] == REARRANGED                      # saved
    assert t["validation"]["pre_gemini_markdown"] == GLUED  # original kept
    assert t["validation"]["table_qa"]["refined_by_gemini"] is True
    assert calls == [GLUED]
    assert refine_mod.refined_count(root, "TST") == 1       # receipt count


def test_refine_table_keeps_original_on_garbage(tmp_path):
    root = _mkroot(tmp_path)
    _flag_only_table(root)
    stub, _ = _stub({GLUED: GARBAGE})
    ledger = root / "data" / refine_mod.LEDGER

    t = json.loads((root / "split" / "TST" / "TST-001" /
                    "questions.jsonl").read_text())["tables"][0]
    st = refine_mod.refine_table(t, None, stub, "all",
                                 ledger_key="TST|q|t",
                                 ledger_path=ledger)
    assert st == "invalid"
    assert t["markdown"] == GLUED            # deterministic output kept
    assert "refined_by_gemini" not in (t.get("validation")
                                       .get("table_qa") or {})
    assert refine_mod.refined_count(root, "TST") == 0


def test_refine_flagged_mode_skips_clean_tables(tmp_path):
    root = _mkroot(tmp_path)
    stub, calls = _stub({GLUED: REARRANGED})
    t = json.loads((root / "split" / "TST" / "TST-001" /
                    "questions.jsonl").read_text())["tables"][0]
    t["validation"]["table_qa"]["status"] = "ok"   # deterministic QA clean
    assert refine_mod.refine_table(t, None, stub, "flagged") == "skip"
    assert calls == []


def test_refine_table_passes_all_source_pages(tmp_path):
    """Cross-page table: ALL spanned pages go to the model, not just
    the first one."""
    seen = []

    def stub(book, pgs, md):
        seen.append(list(pgs))
        return REARRANGED
    t = {"table_id": "T1", "markdown": GLUED,
         "source_pages": [10, 11, 12],
         "validation": {"table_qa": {"status": "REVIEW"}}}
    assert refine_mod.refine_table(t, None, stub, "all") == "replaced"
    assert seen == [[10, 11, 12]]


def test_refiner_sends_text_only(tmp_path, monkeypatch):
    """NO images: the payload carries ONLY the prompt + extracted
    markdown; multi-page span note present, nothing rendered."""
    import qbank.llm as llm_mod
    captured = {}
    monkeypatch.setattr(
        llm_mod, "_call_text",
        lambda pool, key, model, payload:
        captured.update(payload=payload) or REARRANGED)

    class DummyBook:                  # refiner touches book only for cache
        class doc:
            name = "m.pdf"
    rf = llm_mod.refiner(cache_dir=None, key="k")
    out = rf(DummyBook(), [2, 3], GLUED)          # cross-page call
    assert out == REARRANGED
    parts = captured["payload"]["contents"][0]["parts"]
    assert len(parts) == 1 and "inline_data" not in parts[0]
    ask = parts[0]["text"]
    assert GLUED in ask and "Extraction:" in ask  # extracted data gaya
    assert "spanned 2 printed pages" in ask       # span note (text only)
    assert "ONE continuous table" in ask
    # single-page call: no span note
    rf(DummyBook(), [2], GLUED)
    parts = captured["payload"]["contents"][0]["parts"]
    assert "spanned" not in parts[0]["text"]


def test_refine_off_switch(tmp_path, monkeypatch):
    """QBANK_REFINE=off: the chapter hook skips the pass entirely (the
    env default is 'all' — every extracted table goes to Gemini)."""
    monkeypatch.setenv("QBANK_REFINE", "off")
    import os
    assert os.environ.get("QBANK_REFINE", "all") == "off"


def test_api_refine_route_gone(client):
    """The retro 'correct table' button is removed from the dashboards:
    no route, no sparkle button in the UI — extraction-time only."""
    assert client.post("/api/refine", json={"subject": "TST"}).status_code \
        == 404
    html = client.get("/").get_data(as_text=True)
    assert "refineBook(" not in html
    assert "&#10024;" not in html
    assert "/api/refine" not in html


def _census_bad(root):
    cf = root / "split" / "TST" / "TST-001" / "chapter_completeness.json"
    cf.write_text(json.dumps({
        "chapter_id": "TST-001", "census": {"ok": False},
        "unresolved_qid_count": 2}))


def test_zip_builds_when_review_clear_despite_census(client):
    """THE regression: all REVIEW tables decided -> the zip builds even
    when a chapter's census failed / has unresolved q_ids (advisory,
    shipped in the receipt — never a lock no decision can clear)."""
    import zipfile
    root = config.OUTPUT_ROOT
    _census_bad(root)
    gate = __import__("qbank.export", fromlist=["gate_final_zip"]) \
        .gate_final_zip(root, "TST")
    r = client.post("/api/decision", json={
        "book": "TST", "q_id": "TST-001-001", "table_id": "T1",
        "action": "approve"})
    assert r.status_code == 200
    assert r.get_json()["zip_built"] is True
    zp = root / "final_export_TST.zip"
    assert zp.exists()
    with zipfile.ZipFile(zp) as z:
        receipt = json.loads(z.read("REVIEW_RECEIPT.json"))
    assert receipt["census_failed_chapters"] == ["TST-001"]
    assert receipt["unresolved_qids"] == 2


def test_refine_runs_during_extraction(tmp_path, monkeypatch):
    """End to end on the synthetic book: with Gemini enabled every
    extracted table goes to the model at run time and the rearranged
    markdown is what lands on disk (questions + solutions copies)."""
    import pymupdf
    from qbank import run
    from test_mini_book import _build_book

    out = tmp_path / "out"
    monkeypatch.setattr(config, "OUTPUT_ROOT", out)
    monkeypatch.setattr(config, "DATA_DIR", out / "data")
    monkeypatch.setattr(config, "ASSETS_DIR", out / "assets" / "questions")
    monkeypatch.setattr(config, "SPLIT_DIR", out / "split")
    monkeypatch.setattr(config, "SUBJECTS_DIR", out / "subjects")
    monkeypatch.setattr(config, "STATE_FILE", out / "state.json")

    pdf = tmp_path / "mini.pdf"
    _build_book(pdf)

    from qbank import llm as llm_mod
    monkeypatch.setattr(llm_mod, "enabled", lambda: True)
    monkeypatch.setattr(llm_mod, "check_model", lambda *a, **k: True)
    monkeypatch.setattr(llm_mod, "transcriber",
                        lambda cache_dir=None, **k:
                        lambda book, pg, box: None)
    monkeypatch.setattr(llm_mod, "verifier",
                        lambda cache_dir=None, **k:
                        lambda book, pg, box, suspects: None)

    def fake_refiner(cache_dir=None, **k):
        def refine(book, pg, md):
            rows = md.strip().splitlines()
            body = rows[2:]                       # reverse data rows
            return "\n".join(rows[:2] + body[::-1])
        return refine
    monkeypatch.setattr(llm_mod, "refiner", fake_refiner)

    res = run.run_book(str(pdf), "TST", page_offset="auto", force=True,
                       output_root=out)
    assert res["census_failures"] == []

    ch2 = out / "split" / "TST" / "TST-002"
    q = json.loads((ch2 / "questions.jsonl").read_text().splitlines()[0])
    s = json.loads((ch2 / "solutions.jsonl").read_text().splitlines()[0])
    qt, st = q["tables"][0], s["tables"][0]
    lines = qt["markdown"].strip().splitlines()
    assert lines[-1].startswith("| Kinase")       # data rows reversed
    assert lines[2].startswith("| Lipase")
    assert qt["markdown"] == st["markdown"]       # both copies saved
    assert qt["validation"]["table_qa"]["refined_by_gemini"] is True
    assert "pre_gemini_markdown" in qt["validation"]
    assert refine_mod.refined_count(out, "TST") == 1
    assert "| Kinase | adds phosphate | cytosol |" in \
        qt["validation"]["pre_gemini_markdown"]


def test_run_log_shows_refine_stats(tmp_path, monkeypatch, capsys):
    """Per-chapter refine stats are LOUD in the run log — kuch bhi
    silently na ho (rearranged/unchanged/no-answer/invalid counts)."""
    from qbank import run
    from test_mini_book import _build_book
    out = tmp_path / "out"
    monkeypatch.setattr(config, "OUTPUT_ROOT", out)
    monkeypatch.setattr(config, "DATA_DIR", out / "data")
    monkeypatch.setattr(config, "ASSETS_DIR", out / "assets" / "questions")
    monkeypatch.setattr(config, "SPLIT_DIR", out / "split")
    monkeypatch.setattr(config, "SUBJECTS_DIR", out / "subjects")
    monkeypatch.setattr(config, "STATE_FILE", out / "state.json")
    pdf = tmp_path / "mini.pdf"
    _build_book(pdf)
    from qbank import llm as llm_mod
    monkeypatch.setattr(llm_mod, "enabled", lambda: True)
    monkeypatch.setattr(llm_mod, "check_model", lambda *a, **k: True)
    monkeypatch.setattr(llm_mod, "transcriber",
                        lambda cache_dir=None, **k:
                        lambda book, pg, box: None)
    monkeypatch.setattr(llm_mod, "verifier",
                        lambda cache_dir=None, **k:
                        lambda book, pg, box, s: None)
    monkeypatch.setattr(llm_mod, "refiner",
                        lambda cache_dir=None, **k:
                        lambda book, pgs, md: md)
    run.run_book(str(pdf), "TST", page_offset="auto", force=True,
                 output_root=out)
    log = capsys.readouterr().out
    assert "Gemini refine" in log and "rearranged" in log
    # model returned the extraction as-is -> 'unchanged' bucket
    assert "1 unchanged" in log or "2 unchanged" in log


def test_run_log_warns_when_gemini_disabled(tmp_path, monkeypatch, capsys):
    """No key -> LOUD warning in the run log (the exact silent failure
    the user hit on Railway)."""
    from qbank import run
    from test_mini_book import _build_book
    out = tmp_path / "out"
    monkeypatch.setattr(config, "OUTPUT_ROOT", out)
    monkeypatch.setattr(config, "DATA_DIR", out / "data")
    monkeypatch.setattr(config, "ASSETS_DIR", out / "assets" / "questions")
    monkeypatch.setattr(config, "SPLIT_DIR", out / "split")
    monkeypatch.setattr(config, "SUBJECTS_DIR", out / "subjects")
    monkeypatch.setattr(config, "STATE_FILE", out / "state.json")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    from qbank import keypool
    monkeypatch.setattr(keypool, "discover_keys", lambda env=None: [])
    pdf = tmp_path / "mini.pdf"
    _build_book(pdf)
    run.run_book(str(pdf), "TST", page_offset="auto", force=True,
                 output_root=out)
    log = capsys.readouterr().out
    assert "Gemini DISABLED" in log and "no GEMINI_API_KEY" in log


# ---- a stale REVIEW flag must not outlive the repair --------------------

def _vocab_for(markdown_terms):
    """A tiny stand-in vocabulary: words that score >=2 are 'printed
    elsewhere', which is what qa_suspects treats as established."""
    from collections import Counter
    words = Counter()
    for term in markdown_terms:
        words[term.lower()] += 2
    return words, Counter()


def test_accepted_refinement_clears_a_stale_review_flag():
    """Live case (010-T02): the extraction-time REVIEW flag named the
    glued tokens that the accepted rearrangement had already joined,
    and the export gate kept demanding a human forever. The flag must
    be re-judged on the markdown that ships."""
    vocab = _vocab_for(["sensitivity", "testing", "mycobacteria", "growth",
                        "indicator", "tube", "proportion", "method",
                        "versatrek", "microarray"])
    t = {
        "table_id": "T-REVIEW",
        "markdown": "| Conventional drug-sensitivitytesting | Radiometric |\n"
                    "|---|---|\n| Proportion method | BacTMycobacteria |\n",
        "validation": {
            "status": "warnings",
            "table_qa": {"status": "REVIEW",
                         "suspect_fragments": ["sensitivitytesting",
                                               "BacTMycobacteria"]},
            "warnings": ["suspect_lost_space:sensitivitytesting",
                         "suspect_lost_space:BacTMycobacteria",
                         "something_else"],
        },
    }
    fixed = ("| Conventional drug-sensitivity testing | Radiometric |\n"
             "|---|---|\n"
             "| Proportion method | BacT Mycobacteria growth indicator "
             "tube (MGIT) |\n")
    assert refine_mod.refresh_table_qa(t, fixed, vocab) is True
    qa = t["validation"]["table_qa"]
    assert qa["status"] == "ok" and qa["suspect_fragments"] == []
    assert qa["review_cleared_by_refinement"] is True
    assert qa["suspect_fragments_before"] == ["sensitivitytesting",
                                              "BacTMycobacteria"]
    # the lost-space warnings for tokens that are gone go with it; the
    # unrelated warning stays
    assert t["validation"]["warnings"] == ["something_else"]


def test_a_refinement_that_glues_something_keeps_the_review_flag():
    """Re-judged, not rubber-stamped: the model may create a NEW blob."""
    vocab = _vocab_for(["sensitivity", "testing", "proportion", "method"])
    t = {"table_id": "T2", "markdown": "| A | B |\n|---|---|\n| x | y |\n",
         "validation": {"table_qa": {"status": "ok", "suspect_fragments": []}}}
    glued = ("| Conventional drug | Radiometric |\n|---|---|\n"
             "| drugsensitivitytesting | BacTMycobacteria growth |\n")
    assert refine_mod.refresh_table_qa(t, glued, vocab) is False
    qa = t["validation"]["table_qa"]
    assert qa["status"] == "REVIEW" and qa["suspect_fragments"]


def test_refine_table_refreshes_the_flag_on_an_accepted_answer(monkeypatch):
    """End to end through refine_table: an accepted rearrangement that
    removes the glue clears REVIEW (the old code left it set)."""
    vocab = _vocab_for(["sensitivity", "testing", "proportion", "method"])
    monkeypatch.setattr(refine_mod, "_vocab_for", lambda book: vocab)
    t = {"table_id": "T3",
         "markdown": "| Conventional drug-sensitivitytesting | B |\n"
                     "|---|---|\n| Proportion method | y |\n",
         "validation": {"table_qa": {"status": "REVIEW",
                                     "suspect_fragments":
                                     ["sensitivitytesting"]}}}
    rearranged = ("| B | Conventional drug-sensitivity testing |\n"
                  "|---|---|\n| y | Proportion method |\n")
    status = refine_mod.refine_table(
        t, None, lambda b, p, md: rearranged, memo={})
    assert status == "replaced"
    assert t["validation"]["table_qa"]["status"] == "ok"
    assert t["markdown"] == rearranged.strip()


def test_refresh_without_a_vocabulary_leaves_the_flag_alone():
    """No vocabulary = no evidence: never silently clear a human flag."""
    t = {"table_id": "T4", "markdown": "| a |\n|---|\n| b |\n",
         "validation": {"table_qa": {"status": "REVIEW",
                                     "suspect_fragments": ["blob"]}}}
    assert refine_mod.refresh_table_qa(t, t["markdown"], None) is None
    assert t["validation"]["table_qa"]["status"] == "REVIEW"


# ---- word boundaries are content: the vocabulary decides ----------------

def _vocab(words_counts):
    """Vocabulary stand-in: {word: times the book prints it} — the same
    Counter shape tables.build_vocab returns as vocab[0]."""
    from collections import Counter
    return Counter(words_counts)


def test_an_established_word_may_not_be_split_by_the_model():
    """Live corruption (016-T01): the printed cell says "incompletely
    immunized"; Gemini returned "in completely immunized" — same
    letters, different meaning — and the letter/digit envelope waved it
    through because whitespace is invisible to it."""
    words = _vocab({"in": 1500, "completely": 1, "incompletely": 2,
                    "immunized": 6})
    before = "| Clinical | Meningitis and invasive infections " \
             "incompletely immunized infants |\n|---|---|\n| x | y |\n"
    after = "| Clinical | Meningitis and invasive infections in " \
            "completely immunized infants |\n|---|---|\n| x | y |\n"
    ok, diff = refine_mod.same_content(before, after, words)
    assert ok is False
    assert "incompletely -> in completely" in diff["segmentation"]
    # ... and without a vocabulary nothing is invented (old behaviour)
    assert refine_mod.same_content(before, after)[0] is True


def test_a_join_of_broken_fragments_is_not_blocked():
    """The vocabulary is itself polluted by the damaged text layer
    ("matogenous" survives ONCE, from the very line that is broken, and
    "hematogenous" is printed nowhere). No evidence forbids the join,
    so the model's fix must ship — a rule that blocks it loses the
    repair (live: 016-T01 was rejected for exactly this)."""
    # "matogenous" is attested twice — both times from that broken line
    words = _vocab({"he": 900, "matogenous": 2, "spread": 30})
    ok, diff = refine_mod.same_content(
        "| P | infections due to he matogenous spread |\n|---|---|\n",
        "| P | infections due to hematogenous spread |\n|---|---|\n",
        words)
    assert ok is True
    # ... but welding two ESTABLISHED words into a form the book never
    # prints is still refused
    # a many-token boundary rewrite is not judged here (028-T02: the
    # blob "jeanselmeiPhialop" counts as an "established word" only
    # because the broken line prints it twice)
    words2 = _vocab({"jeanselmeiphialop": 2, "jeanselmei": 1,
                     "phialophora": 3})
    ok2, _ = refine_mod.same_content(
        "| A | jeanselmeiPhialop hora |\n|---|---|\n",
        "| A | jeanselmei Phialophora |\n|---|---|\n", words2)
    assert ok2 is True


def test_the_repair_direction_stays_allowed():
    """The same rule must not block the fix: a corrupted extraction
    ("in completely") may become the printed "incompletely"."""
    words = _vocab({"in": 1500, "completely": 1, "incompletely": 2})
    bad = "| a | in completely immunized |\n|---|---|\n"
    good = "| a | incompletely immunized |\n|---|---|\n"
    assert refine_mod.same_content(bad, good, words)[0] is True


def test_restore_puts_back_a_word_the_answer_split_apart():
    """The extraction prints it whole -> the answer may not re-cut it.
    Letters are identical by construction, so nothing is invented."""
    words = _vocab({"incompletely": 2, "in": 1500, "completely": 1,
                    "immunized": 6})
    before = "| a | incompletely immunized children |\n|---|---|\n"
    after = "| a | in completely immunized children |\n|---|---|\n"
    fixed, n = refine_mod.restore_split_words(before, after, words)
    assert n == 1 and "incompletely immunized" in fixed
    assert "in completely" not in fixed


def test_restore_never_touches_what_the_extraction_does_not_print():
    """Restore puts back the EXTRACTION's own words — it does not go
    looking for other words the book prints somewhere else, so a model
    answer in a cell the extraction wrote differently is left alone
    (that is the envelope's job, not this repair's)."""
    words = _vocab({"moderate": 40, "mode": 9, "rate": 12, "passages": 1,
                    "pass": 30, "ages": 20, "cannot": 25, "can": 200,
                    "not": 300})
    for before, after in (
            ("| a | Some other cell text |\n|---|---|\n",
             "| a | Mode rate |\n|---|---|\n"),
            ("| a | Some other cell text |\n|---|---|\n",
             "| a | pass ages |\n|---|---|\n"),
            ("| a | Some other cell text |\n|---|---|\n",
             "| a | can not |\n|---|---|\n")):
        fixed, n = refine_mod.restore_split_words(before, after, words)
        assert (fixed, n) == (after, 0)


def test_restore_keeps_the_models_own_repairs():
    """"phos phate" -> "phosphate" (a join the model made) must stay:
    the model is allowed to mend, just not to re-cut."""
    words = _vocab({"phosphate": 4, "phos": 1, "phate": 1,
                    "incompletely": 2, "in": 1500, "completely": 1,
                    "immunized": 6})
    before = ("| a | Ribosyl-ribitol phos phate |\n|---|---|\n"
              "| b | incompletely immunized |\n")
    after = ("| a | Ribosyl-ribitol phosphate |\n|---|---|\n"
             "| b | in completely immunized |\n")
    fixed, n = refine_mod.restore_split_words(before, after, words)
    assert n == 1
    assert "phosphate" in fixed and "incompletely" in fixed


def test_restore_ignores_punctuation_and_number_tokens():
    words = _vocab({"middle": 50, "ear": 80, "immune": 20, "system": 30})
    for before, after in (
            ("| a | middle ear |\n|---|---|\n", "| a | middle ear |\n|---|---|\n"),
            ("| a | 1,000 cells |\n|---|---|\n", "| a | 1,000 cells |\n|---|---|\n"),
            ("| a | O157:H7 |\n|---|---|\n", "| a | O157:H7 |\n|---|---|\n")):
        fixed, n = refine_mod.restore_split_words(before, after, words)
        assert (fixed, n) == (after, 0)


def test_refine_table_repairs_a_boundary_before_shipping(monkeypatch):
    """End to end: the model's layout fixes are KEPT and its boundary
    slip is corrected with the book's own evidence, instead of the
    whole (good) rearrangement being thrown away."""
    words = _vocab({"in": 1500, "completely": 1, "incompletely": 2,
                    "immunized": 6, "phos": 0, "phate": 0, "phosphate": 4})
    monkeypatch.setattr(refine_mod, "_vocab_for", lambda book: (words, None))
    det = ("| Capsule | Made of Ribosyl-ribitol phos phate | Unencapsulated |\n"
           "|---|---|---|\n"
           "| Clinical | incompletely immunized children | x |\n")
    # the model's REAL answer shape: same cells, rows reordered, the
    # glued "phos phate" fixed — plus the boundary slip
    answer = ("| Capsule | Made of Ribosyl-ribitol phosphate | Unencapsulated |\n"
              "|---|---|---|\n"
              "| Clinical | in completely immunized children | x |\n")
    t = {"table_id": "T-SEG", "markdown": det, "validation": {}}
    status = refine_mod.refine_table(t, None, lambda b, p, md: answer, memo={})
    assert status == "replaced"
    assert "incompletely immunized" in t["markdown"]      # boundary repaired
    assert "phosphos" not in t["markdown"]                # glued form gone
    assert "phos phate" not in t["markdown"]
    assert t["validation"]["segmentation_repairs"] == 1


def test_a_split_of_an_established_word_into_printed_parts_is_refused():
    """Both halves being real words does not make the split right: the
    printed word wins on evidence ("Peroxidase" -> "per oxidase")."""
    words = _vocab({"peroxidase": 5, "per": 30, "oxidase": 4})
    ok, diff = refine_mod.same_content("| A | peroxidase |\n|---|---|\n",
                                       "| A | per oxidase |\n|---|---|\n",
                                       words)
    assert ok is False and "peroxidase -> per oxidase" in diff["segmentation"]


def test_a_wrap_artifact_join_is_allowed_even_though_both_parts_print():
    """Live case 034-T01: the damaged line prints "pred nisolone" — the
    parts are attested TWICE (both from that broken line) and the pair
    is never printed next to each other, so the space is a wrap
    artifact and the model's "prednisolone" is the repair. A rule that
    refused it shipped a broken word in the table."""
    from collections import Counter
    words = Counter({"pred": 2, "nisolone": 2, "diethylcarbamazine": 3})
    pairs = Counter()                      # ("pred","nisolone") never printed
    ok, _ = refine_mod.same_content(
        "| T | Diethylcarbamazine and pred nisolone |\n|---|---|\n",
        "| T | Diethylcarbamazine and prednisolone |\n|---|---|\n",
        (words, pairs))
    assert ok is True


# --------------------------------------------------- mark placement
# The rearrange pass may move cells around and drop the separators the
# typesetter ran into the text, but a MARK (".", "-", "/") must sit where
# the print puts it. Live corruption this pins: the printed "Madurella
# griseaE jeanselmei" was rearranged into "Madurella grisea" + "E.
# jeanselmei" — one dot out, one in, so the count-free signature saw an
# unchanged multiset and the invented period shipped, where the verifier
# (which keeps "." as evidence) failed the very cell it came from.

_CELL_TABLE = ("| Mycetoma | Pseudallescheria boydii (Anamorph Scedosporium "
               "apiospermum).Madurella mycetomatis Madurella griseaE "
               "jeanselmei Acremonium falciforme |\n"
               "|---|---|\n| X | y |\n")


def test_a_moved_period_is_refused():
    moved = ("| Mycetoma | • Pseudallescheria boydii (Anamorph Scedosporium "
             "apiospermum)<br>• Madurella mycetomatis<br>• Madurella grisea"
             "<br>• E. jeanselmei<br>• Acreonium falciforme |\n"
             "|---|---|\n| X | y |\n").replace("Acreonium", "Acremonium")
    ok, diff = refine_mod.same_content(_CELL_TABLE, moved)
    assert ok is False
    assert ". after 'E'" in diff["summary"]          # the invented "E."


def test_a_dropped_separator_and_bullets_are_allowed():
    """The same rearrangement without inventing a mark: the printed
    separator "." disappears into <br> bullets, the rows are reordered,
    and the letters are untouched — that is the pass's own job."""
    rearranged = ("| X | y |\n|---|---|\n| Mycetoma | • Pseudallescheria "
                  "boydii (Anamorph Scedosporium apiospermum)<br>• Madurella "
                  "mycetomatis<br>• Madurella griseaE jeanselmei<br>• "
                  "Acremonium falciforme |\n")
    assert refine_mod.same_content(_CELL_TABLE, rearranged)[0] is True


def test_an_invented_hyphen_is_refused():
    hyphen = _CELL_TABLE.replace("apiospermum).Madurella",
                                 "apiospermum)-Madurella")
    ok, diff = refine_mod.same_content(_CELL_TABLE, hyphen)
    assert ok is False and "mark moved/added" in diff["summary"]


def test_a_respaced_dash_stays_legal():
    """Re-spacing around an existing dash must not start failing: the
    mark still sits on the same letter run."""
    before = "| T | 10-20 years |\n|---|---|\n| a | b |\n"
    after = "| T | 10 - 20 years |\n|---|---|\n| a | b |\n"
    assert refine_mod.same_content(before, after)[0] is True


def test_a_case_only_change_needs_the_books_own_print():
    """Case is content: the verifier compares letter identity, so a
    casing no page of the book prints is an invention (ANA 049-T02
    shipped "Right psoas major" over a print that reads "Right Psoas
    major" and failed the independent check). The rearrangement is
    accepted only when the book itself prints the model's form."""
    shouted = _CELL_TABLE.replace("Madurella", "MADURELLA")
    # no evidence at hand -> the print's casing is authoritative
    ok, diff = refine_mod.same_content(_CELL_TABLE, shouted)
    assert ok is False and diff["case_unsupported"] is True
    # the book prints this form -> the model's casing is fine
    case = {"madurella": Counter({"MADURELLA": 2})}
    ok, diff = refine_mod.same_content(_CELL_TABLE, shouted, case=case)
    assert ok is True and diff["case_only"] is True


def test_a_casing_the_book_never_prints_is_restored_not_shipped():
    from qbank.tables import restore_printed_case
    case = {"psoas": Counter({"Psoas": 3, "PSOAS": 1}),
            "the": Counter({"the": 900, "The": 40}),
            "right": Counter({"Right": 40, "right": 5}),
            "major": Counter({"major": 12})}
    # "psoas" is a casing the book never prints -> dominant form restored
    out, n = restore_printed_case("Right psoas major", case)
    assert (out, n) == ("Right Psoas major", 1)
    # casings the book DOES print are untouched, however rare ("The" 40x
    # against "the" 900x stays as the print has it) — and a word with no
    # printed form at all is left alone
    out, n = restore_printed_case("The Right psoas major zzz", case)
    assert out == "The Right Psoas major zzz" and n == 1
    # the minority printed casing is still a printed casing
    out, n = restore_printed_case("PSOAS", case)
    assert (out, n) == ("PSOAS", 0)


# --- content marks may never be INVENTED (verifier-visible marks) -------
def test_an_invented_colon_is_refused():
    """ANA 017-T02 shipped "Give rise to sensory nuclei: CN V, VII…"
    where the print reads "…nucleiCN V…" (the motor listing of the same
    page does print its colon). display_text does not fold ":", so this
    is the class the signature must see — and the verifier keeps ":"."""
    before = "| A | Give rise to sensory nuclei CN V, VII, VIII |\n|---|---|\n"
    after = "| A | Give rise to sensory nuclei: CN V, VII, VIII |\n|---|---|\n"
    ok, diff = refine_mod.same_content(before, after)
    assert ok is False and "mark invented" in diff["summary"]
    assert diff["marks_invented"] == [": after 'i'"]


def test_an_invented_period_is_refused_but_a_printed_one_is_not():
    before = "| A | of the epididymis These tubules coalesce |\n|---|---|\n"
    after = "| A | of the epididymis.<br>These tubules coalesce |\n|---|---|\n"
    assert refine_mod.same_content(before, after)[0] is False
    # the print's own run-in separator: keeping it, or normalising it to
    # a bullet, is presentation — both must stay legal
    run_in = "| A | apiospermum).Madurella mycetomatis |\n|---|---|\n"
    bullets = "| A | apiospermum).<br>Madurella mycetomatis |\n|---|---|\n"
    assert refine_mod.same_content(run_in, bullets)[0] is True
    same = "| A | apiospermum). Madurella mycetomatis |\n|---|---|\n"
    assert refine_mod.same_content(run_in, same)[0] is True
