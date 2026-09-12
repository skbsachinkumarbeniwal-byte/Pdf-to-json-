"""dashboard.py <-> review layer integration: /review page, queue,
audit, decision/edit endpoints, zip + crops routes (flask
test_client, synthetic output root)."""
import io
import json
import zipfile

import pytest

flask = pytest.importorskip("flask")

from qbank import config

MD = "| A | B |\n|---|---|\n| gluedof text | ok |"
MD_FIXED = "| A | B |\n|---|---|\n| glued of text | ok |"


def _mkroot(tmp_path):
    root = tmp_path / "out"
    ch = root / "split" / "TST" / "TST-001"
    ch.mkdir(parents=True)
    (ch / "questions.jsonl").write_text(json.dumps({
        "q_id": "TST-001-001",
        "question_text": "Which artery?",
        "options": [{"id": "A", "text": "RCA",
                     "images": [{"file": "f1"}]},
                    {"id": "B", "text": "LAD"},
                    {"id": "C", "text": "LCX"},
                    {"id": "D", "text": "PDA"}],
        "tables": [{"table_id": "T1", "markdown": MD,
                    "source_pages": [10],
                    "validation": {"table_qa": {
                        "status": "REVIEW",
                        "suspect_fragments": ["gluedof"]}}}]}) + "\n")
    (ch / "answers.jsonl").write_text(json.dumps(
        {"q_id": "TST-001-001", "correct_option": "A"}) + "\n")
    (ch / "solutions.jsonl").write_text(json.dumps(
        {"q_id": "TST-001-001", "solution_text": "Because X.",
         "source_pages": [50]}) + "\n")
    (root / "data").mkdir()
    (root / "data" / "audit_report.jsonl").write_text(json.dumps(
        {"kind": "thin_options", "q_id": "TST-001-001",
         "severity": "HIGH"}) + "\n")
    return root


@pytest.fixture()
def client(tmp_path, monkeypatch):
    root = _mkroot(tmp_path)
    monkeypatch.setattr(config, "OUTPUT_ROOT", root)
    monkeypatch.setattr(config, "SUBJECTS_DIR", root / "subjects")
    import dashboard
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c


def test_review_page_served(client):
    r = client.get("/review")
    assert r.status_code == 200
    assert b"Table Review Dashboard" in r.data


def test_queue_lists_pending_table(client):
    q = client.get("/api/queue").get_json()
    assert len(q) == 1
    assert q[0]["book"] == "TST" and q[0]["state"] == "pending"
    assert q[0]["suspects"] == ["gluedof"]


def test_audit_report_served(client):
    a = client.get("/api/audit").get_json()
    assert [f["kind"] for f in a] == ["thin_options"]


def test_decision_moves_state(client):
    r = client.post("/api/decision", json={
        "book": "TST", "q_id": "TST-001-001", "table_id": "T1",
        "action": "approve"})
    assert r.status_code == 200
    q = client.get("/api/queue").get_json()
    assert q[0]["state"] == "decided:approve"


def test_edit_rewrites_copies_and_decides(client):
    r = client.post("/api/edit", json={
        "book": "TST", "q_id": "TST-001-001", "table_id": "T1",
        "markdown": MD_FIXED, "action": "approve"})
    j = r.get_json()
    assert j["ok"] and j["copies"] >= 1
    on_disk = (config.OUTPUT_ROOT / "split" / "TST" / "TST-001"
               / "questions.jsonl").read_text()
    assert "glued of text" in on_disk
    q = client.get("/api/queue").get_json()
    assert q[0]["state"] == "decided:approve"


def test_edit_missing_fields_400(client):
    r = client.post("/api/edit", json={"book": "TST"})
    assert r.status_code == 400


def test_zip_missing_then_present(client):
    assert client.get("/zip/TST").status_code == 404
    zp = config.OUTPUT_ROOT / "final_export_TST.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("hello.txt", "hi")
    r = client.get("/zip/TST")
    assert r.status_code == 200
    assert zipfile.ZipFile(io.BytesIO(r.data)).read("hello.txt") == b"hi"


def test_drive_url_mapping():
    import dashboard
    assert dashboard._drive_url(
        "https://drive.google.com/file/d/1iynmgRjzl7L4H_tQd9Gxj-"
        "3lfdVmDo4Y/view?usp=sharing") == \
        "https://drive.google.com/uc?export=download&id=" \
        "1iynmgRjzl7L4H_tQd9Gxj-3lfdVmDo4Y"
    assert dashboard._drive_url(
        "https://drive.google.com/open?id=ABC1234567890") == \
        "https://drive.google.com/uc?export=download&id=ABC1234567890"
    plain = "https://example.com/x.pdf"
    assert dashboard._drive_url(plain) == plain


def test_fetch_saves_pdf_and_registers_book(client, tmp_path, monkeypatch):
    import dashboard
    from qbank import config as cfg

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            return b"%PDF-1.4 fake book"

    monkeypatch.setattr(dashboard.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp())
    (tmp_path / "pdfs").mkdir()
    monkeypatch.setattr(dashboard, "PDF_DIR", tmp_path / "pdfs")
    monkeypatch.setattr(cfg, "BOOKS_FILE", tmp_path / "books.json")
    cfg.BOOKS_FILE.write_text("{}")
    r = client.post("/api/fetch", json={
        "url": "https://example.com/books/obg.pdf", "subject": "OBG"})
    j = r.get_json()
    assert j["ok"], j
    assert (tmp_path / "pdfs" / "obg.pdf").read_bytes() == b"%PDF-1.4 fake book"
    assert "OBG" in json.loads(cfg.BOOKS_FILE.read_text())


def test_fetch_rejects_non_pdf(client, tmp_path, monkeypatch):
    import dashboard

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            return b"<html>login page</html>"

    monkeypatch.setattr(dashboard.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp())
    (tmp_path / "pdfs").mkdir()
    monkeypatch.setattr(dashboard, "PDF_DIR", tmp_path / "pdfs")
    r = client.post("/api/fetch", json={
        "url": "https://example.com/x", "subject": "OBG"})
    assert r.status_code == 400


def test_crops_route(client):
    (config.OUTPUT_ROOT / "crops").mkdir()
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 16
    (config.OUTPUT_ROOT / "crops" / "tst_p0010.png").write_bytes(png)
    r = client.get("/crops/tst_p0010.png")
    assert r.status_code == 200 and r.data == png
    assert client.get("/crops/nope.png").status_code == 404
    # traversal must never serve a file (400 by our check, 404 by
    # werkzeug routing — either is fine)
    r = client.get("/crops/..%2fsecret.png")
    assert r.status_code in (400, 404) and r.data != png

def test_question_lookup_case_insensitive_and_404(client):
    r = client.get("/api/question/tst-001-001")
    assert r.status_code == 200
    j = r.get_json()
    assert j["q"]["q_id"] == "TST-001-001"
    assert j["answer"]["correct_option"] == "A"
    assert j["solution"]["solution_text"] == "Because X."
    assert client.get("/api/question/NOPE-000-000").status_code == 404


def test_question_edit_verified_and_images_preserved(client):
    r = client.post("/api/question/TST-001-001/edit", json={
        "book": "TST",
        "patch": {
            "question_text": "Which artery is occluded?",
            "options": [{"id": "A", "text": "Right coronary artery"},
                        {"id": "B", "text": "LAD"},
                        {"id": "C", "text": "LCX"},
                        {"id": "D", "text": "PDA"}],
            "solution_text": "RCA supplies the inferior wall.",
            "correct_option": "A"}})
    j = r.get_json()
    assert j["ok"], j
    ch = config.OUTPUT_ROOT / "split" / "TST" / "TST-001"
    q = json.loads((ch / "questions.jsonl").read_text().splitlines()[0])
    assert q["question_text"] == "Which artery is occluded?"
    assert q["options"][0]["text"] == "Right coronary artery"
    assert q["options"][0]["images"] == [{"file": "f1"}]   # preserved
    sol = json.loads((ch / "solutions.jsonl").read_text().splitlines()[0])
    assert sol["solution_text"] == "RCA supplies the inferior wall."
    ledger = (config.OUTPUT_ROOT / "human_edit_ledger.jsonl").read_text()
    assert "question_edit" in ledger


def test_question_edit_unknown_id_refused(client):
    r = client.post("/api/question/NOPE-000-000/edit", json={
        "book": "TST", "patch": {"question_text": "x"}})
    assert r.get_json()["ok"] is False

def test_question_edit_partial_patch_leaves_options(client):
    # regression: patch without options must not trip read-back
    r = client.post("/api/question/TST-001-001/edit", json={
        "book": "TST", "patch": {"question_text": "Only stem changed"}})
    j = r.get_json()
    assert j["ok"], j
    ch = config.OUTPUT_ROOT / "split" / "TST" / "TST-001"
    q = json.loads((ch / "questions.jsonl").read_text().splitlines()[0])
    assert q["question_text"] == "Only stem changed"
    assert q["options"][0]["text"] == "RCA"          # untouched
    assert q["options"][0]["images"] == [{"file": "f1"}]

def test_lookup_flexible_forms(client):
    # exact id, case-insensitive, chapter-number form, bare number
    for term in ("TST-001-001", "tst-001-001", "001-001", "1"):
        r = client.get(f"/api/lookup?term={term}")
        ids = [x["q"]["q_id"] for x in r.get_json()]
        assert ids == ["TST-001-001"], (term, ids)
    # other subject's id / absent number match nothing
    for term in ("ENT-021-008", "2"):
        r = client.get(f"/api/lookup?term={term}")
        assert [x["q"]["q_id"] for x in r.get_json()] == []
    r = client.get("/api/lookup?term=TST")
    assert len(r.get_json()) == 1


def test_assets_route(client, tmp_path, monkeypatch):
    base = tmp_path / "assets" / "questions"
    (base / "TST").mkdir(parents=True)
    (base / "TST" / "x.webp").write_bytes(b"WEBPFAKE")
    monkeypatch.setattr(config, "ASSETS_DIR", base)
    r = client.get("/assets/TST/x.webp")
    assert r.status_code == 200 and r.data == b"WEBPFAKE"
    assert client.get("/assets/../secrets.txt").status_code == 404
    assert client.get("/assets/TST/nope.webp").status_code == 404

def test_export_refused_while_gate_locked(client):
    # no subject -> 400 (there is no combined export any more)
    assert client.post("/api/export").status_code == 400
    assert client.get("/download").status_code == 400
    # the book's own gate locks ITS zip
    r = client.post("/api/export", json={"subject": "TST"})
    assert r.status_code == 409
    assert "REVIEW" in r.get_json()["error"]

def _add_clean_subject(root):
    ch = root / "split" / "AAA" / "AAA-001"
    ch.mkdir(parents=True)
    (ch / "questions.jsonl").write_text(json.dumps({
        "q_id": "AAA-001-001", "question_text": "q",
        "options": [{"id": l, "text": t}
                    for l, t in zip("ABCD", "abcd")]}) + "\n")
    (ch / "answers.jsonl").write_text(json.dumps(
        {"q_id": "AAA-001-001", "correct_option": "A"}) + "\n")
    (ch / "solutions.jsonl").write_text(json.dumps(
        {"q_id": "AAA-001-001", "solution_text": "s"}) + "\n")
    (ch / "image_manifest.jsonl").write_text("")
    (ch / "chapter_completeness.json").write_text(json.dumps({
        "chapter_id": "AAA-001", "census": {"ok": True},
        "qa_status_counts": {}, "unresolved_qid_count": 0}))


def test_per_book_gate_and_independent_zip(client):
    from qbank.export import gate_final_zip
    root = config.OUTPUT_ROOT
    _add_clean_subject(root)
    # TST still has a pending REVIEW table; AAA is clean
    assert gate_final_zip(root, "AAA")["locked"] is False
    assert gate_final_zip(root, "TST")["locked"] is True
    # independent zip: AAA builds, TST refused, AAA zip has no TST files
    r = client.post("/api/export", json={"subject": "AAA"})
    assert r.status_code == 200, r.get_json()
    names = zipfile.ZipFile(root / "final_export_AAA.zip").namelist()
    assert any("AAA-001/questions.jsonl" in n for n in names)
    assert not any("TST" in n for n in names)
    assert client.post("/api/export", json={"subject": "TST"}
                       ).status_code == 409
    assert client.get("/download?subject=AAA").status_code == 200
    assert client.get("/download?subject=TST").status_code == 404
    # status exposes the independent zip
    st = client.get("/api/status").get_json()
    assert st["zips"]["AAA"]["name"] == "final_export_AAA.zip"


def test_purge_frees_volume_keeps_zip(client):
    from qbank import purge as purge_mod
    root = config.OUTPUT_ROOT
    _add_clean_subject(root)
    # a shipped zip for AAA
    import zipfile as zf
    with zf.ZipFile(root / "final_export_AAA.zip", "w") as z:
        z.writestr("REVIEW_RECEIPT.json", "{}")
    # ledger rows for AAA (must be filtered) and TST (must stay)
    from qbank import review as rv
    rv.record_decision(root, "AAA", "AAA-001-001", "001-T01", "approve")
    rv.record_decision(root, "TST", "TST-001-001", "001-T01", "approve")
    led = root / rv.DECISIONS
    assert "AAA" in led.read_text()
    assert (root / "split" / "AAA").is_dir()
    r = client.post("/api/purge", json={"subject": "AAA"})
    assert r.status_code == 200, r.get_json()
    assert not (root / "split" / "AAA").exists()
    assert not (root / "assets" / "questions" / "AAA").exists()
    assert (root / "final_export_AAA.zip").exists()   # zip kept
    # TST (not purged) still intact
    assert (root / "split" / "TST").is_dir()
    assert "AAA" not in led.read_text()          # AAA rows gone
    assert "TST" in led.read_text()              # TST rows kept
    # no subject -> 400
    assert client.post("/api/purge", json={}).status_code == 400
    # direct module call with keep_zip=False drops the zip too
    purge_mod.purge_subject(root, "AAA", keep_zip=False)
    assert not (root / "final_export_AAA.zip").exists()


def test_status_tolerates_half_written_zip(client):
    # simulate the run-thread/status race: a garbage .zip on the volume
    bad = config.OUTPUT_ROOT / "final_export_ZZZ.zip"
    bad.write_bytes(b"PK\x03\x04 not a real zip")
    r = client.get("/api/status")
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    assert all(b["subject"] != "ZZZ" for b in j.get("books", []))
    bad.unlink()
