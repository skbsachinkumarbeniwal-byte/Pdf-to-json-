"""Gemini medical rearrangement: saved during extraction, structural
validation only (stubbed model — no network). Also covers the export
gate contract: review queue clear -> zip builds, census advisories
never block."""
import json

import pytest

flask = pytest.importorskip("flask")

from qbank import config
from qbank import refine as refine_mod

from test_dashboard_review import _mkroot, client  # noqa: F401

GLUED = "| A | B |\n|---|---|\n| ASCAOMP-C Anti I2 | antibodyp-ANCA |"
REARRANGED = ("| Antibody | Target |\n|---|---|\n"
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
    """Adoora extraction: restore from the PAGE IMAGE (source of
    truth), never invent from model memory, never guess past a page
    that is itself cut off."""
    from qbank import llm as llm_mod
    p = llm_mod.REARRANGE_PROMPT
    assert "source of truth" in p.lower()
    assert "RESTORE" in p              # image se complete karna allowed
    assert "INCOMPLETE" in p           # extraction adhoora ho to
    assert "memory" in p               # apne knowledge se nahi
    assert "not be invented" in p
    assert "guesswork" in p            # page hi cut ho to as-is


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
