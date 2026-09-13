"""Gemini same-content table refinement: envelope, retro pass, and the
dashboard trigger (stubbed model — no network)."""
import json

import pytest

flask = pytest.importorskip("flask")

from qbank import config
from qbank import refine as refine_mod

from test_dashboard_review import _mkroot, client  # noqa: F401

GLUED = "| A | B |\n|---|---|\n| ASCAOMP-C Anti I2 | antibodyp-ANCA |"
SPLIT = "| A | B |\n|---|---|\n| ASCA OMP-C Anti I2 | antibody p-ANCA |"
ADDED = "| A | B |\n|---|---|\n| ASCA OMP-C Anti I2 60 | antibody p-ANCA |"
DELETED = "| A | B |\n|---|---|\n| ASCA OMP-C |"


def test_envelope_splits_ok_adds_rejected():
    assert refine_mod.same_content(GLUED, SPLIT)
    assert not refine_mod.same_content(GLUED, ADDED)
    assert not refine_mod.same_content(GLUED, DELETED)
    assert not refine_mod.same_content(GLUED, None)




def _set_glued(root):
    qf = root / "split" / "TST" / "TST-001" / "questions.jsonl"
    row = json.loads(qf.read_text().splitlines()[0])
    row["tables"][0]["markdown"] = GLUED
    qf.write_text(json.dumps(row) + "\n")


def _stub(rows_by_md):
    calls = []

    def refine(book, pg, md):
        calls.append(md)
        return rows_by_md.get(md)
    return refine, calls


def test_refine_subject_rewrites_flagged_only(tmp_path, monkeypatch):
    root = _mkroot(tmp_path)
    _set_glued(root)
    monkeypatch.setattr(config, "OUTPUT_ROOT", root)
    monkeypatch.setattr(config, "load_books",
                        lambda: {"TST": {"path": "p.pdf"}})
    monkeypatch.setattr(config, "resolve_book_path",
                        lambda e: "p.pdf")

    class FakeBook:
        def close(self):
            pass
    import qbank.textlayer as tl
    monkeypatch.setattr(tl, "Book", lambda p: FakeBook())

    stub, calls = _stub({GLUED: SPLIT})
    res = refine_mod.refine_subject(root, "TST", refine_fn=stub,
                                    log=lambda *a: None)
    assert res["refined"] == 1 and res["rejected"] == 0
    qf = root / "split" / "TST" / "TST-001" / "questions.jsonl"
    row = json.loads(qf.read_text().splitlines()[0])
    assert row["tables"][0]["markdown"] == SPLIT
    assert row["tables"][0]["validation"]["table_qa"][
        "refined_by_gemini"] is True
    # envelope ledger + receipt count
    assert refine_mod.refined_count(root, "TST") == 1
    # refined table is back in the review queue for the user
    from qbank import review as rv
    q = rv.review_tables(root)
    assert any(t["state"].startswith("stale") or t["state"] == "pending"
               for t in q)


def test_refine_subject_rejects_invented_content(tmp_path, monkeypatch):
    root = _mkroot(tmp_path)
    _set_glued(root)
    monkeypatch.setattr(config, "OUTPUT_ROOT", root)
    monkeypatch.setattr(config, "load_books",
                        lambda: {"TST": {"path": "p.pdf"}})
    monkeypatch.setattr(config, "resolve_book_path", lambda e: "p.pdf")

    class FakeBook:
        def close(self):
            pass
    import qbank.textlayer as tl
    monkeypatch.setattr(tl, "Book", lambda p: FakeBook())

    stub, calls = _stub({GLUED: ADDED})     # model adds percentages
    res = refine_mod.refine_subject(root, "TST", refine_fn=stub,
                                    log=lambda *a: None)
    assert res["refined"] == 0 and res["rejected"] >= 1
    qf = root / "split" / "TST" / "TST-001" / "questions.jsonl"
    row = json.loads(qf.read_text().splitlines()[0])
    assert row["tables"][0]["markdown"] == GLUED   # original kept


def test_api_refine_validation(client, monkeypatch):
    assert client.post("/api/refine", json={}).status_code == 400
    r = client.post("/api/refine", json={"subject": "NOPE"})
    assert r.status_code == 409
    html = client.get("/").get_data(as_text=True)
    assert "refineBook(" in html and "&#10024;" in html
