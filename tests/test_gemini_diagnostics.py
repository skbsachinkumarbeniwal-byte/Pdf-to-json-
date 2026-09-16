"""Gemini failure visibility: nothing may fail silently.

Covers the exact traps that made "GEMINI ON" runs produce raw tables
with no explanation in the log: swallowed HTTP errors, thought parts
from thinking models, MAX_TOKENS truncation, cache keys that ignore the
model, and rejected model answers with no sample.
"""
import io
import json
import urllib.error

import pytest

from qbank import llm
from qbank import refine as refine_mod

TABLE = "| A | B |\n|---|---|\n| x | y |"


@pytest.fixture(autouse=True)
def _fresh_reports(monkeypatch):
    """report_error dedupes process-wide: start every test clean."""
    monkeypatch.setattr(llm, "_ERRORS_SEEN", set())


def _http_error(code: int, body: dict):
    return urllib.error.HTTPError(
        "https://x", code, "err", {}, io.BytesIO(json.dumps(body).encode()))


# ---- response reading -------------------------------------------------

def test_answer_text_plain():
    txt, why = llm._answer_text(
        {"candidates": [{"content": {"parts": [{"text": TABLE}]}}]})
    assert txt == TABLE and why == ""


def test_answer_text_skips_thought_parts():
    """A thinking model returns its reasoning NEXT TO the answer; the
    old code concatenated both, so the table arrived glued to prose."""
    txt, why = llm._answer_text({"candidates": [{"content": {"parts": [
        {"text": "Let me think about this table...", "thought": True},
        {"text": TABLE},
    ]}}]})
    assert txt == TABLE and "think" not in txt


def test_answer_text_max_tokens_is_named():
    txt, why = llm._answer_text({"candidates": [
        {"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]})
    assert txt == "" and "MAX_TOKENS" in why and "budget" in why


def test_answer_text_blocked_prompt():
    txt, why = llm._answer_text(
        {"promptFeedback": {"blockReason": "SAFETY"}})
    assert txt == "" and "SAFETY" in why


# ---- HTTP failures are printed, once ----------------------------------

def test_call_text_reports_404_with_hint(monkeypatch, capsys):
    monkeypatch.setattr(llm, "_post", lambda url, payload, key: (_ for _ in
                        ()).throw(_http_error(404, {"error": {
                            "status": "NOT_FOUND",
                            "message": "models/nope is not found"}})))
    assert llm._call_text(None, "K", "nope", {}) is None
    out = capsys.readouterr().out
    assert "HTTP 404 NOT_FOUND" in out
    assert "QBANK_LLM_MODEL" in out          # points at the fix
    assert "check_gemini.py" in out


def test_call_text_reports_each_failure_once(monkeypatch, capsys):
    monkeypatch.setattr(llm, "_post", lambda url, payload, key: (_ for _ in
                        ()).throw(_http_error(403, {"error": {
                            "status": "PERMISSION_DENIED",
                            "message": "key not allowed"}})))
    assert llm._call_text(None, "K", "m", {}) is None
    first = capsys.readouterr().out
    assert "HTTP 403 PERMISSION_DENIED" in first
    assert llm._call_text(None, "K", "m", {}) is None
    assert capsys.readouterr().out == ""      # deduped: no log spam


def test_call_reports_network_failure(monkeypatch, capsys):
    def boom(url, payload, key):
        raise OSError("no route to host")
    monkeypatch.setattr(llm, "_post", boom)
    assert llm._call(None, "K", "m", {}) is None
    out = capsys.readouterr().out
    assert "call failed" in out and "no route to host" in out


def test_call_returns_none_when_answer_is_not_json(monkeypatch, capsys):
    monkeypatch.setattr(llm, "_post", lambda u, p, k: {"candidates": [
        {"content": {"parts": [{"text": "sorry, here is the table: " + TABLE}]}}]})
    assert llm._call(None, "K", "m", {}) is None
    assert "not the expected JSON" in capsys.readouterr().out


# ---- MAX_TOKENS self-heal ---------------------------------------------

def test_max_tokens_retries_with_bigger_budget(monkeypatch):
    seen = []

    def fake_post(url, payload, key):
        seen.append(payload)
        if len(seen) == 1:
            return {"candidates": [{"content": {"parts": []},
                                    "finishReason": "MAX_TOKENS"}]}
        return {"candidates": [{"content": {"parts": [{"text": TABLE}]}}]}

    monkeypatch.setattr(llm, "_post", fake_post)
    payload = {"generationConfig": {"temperature": 0.0,
                                    "max_output_tokens": 1024}}
    assert llm._call_text(None, "K", "m", payload) == TABLE
    assert seen[1]["generationConfig"]["max_output_tokens"] == 2048
    assert payload["generationConfig"]["max_output_tokens"] == 1024  # untouched


def test_max_tokens_never_exceeds_ceiling():
    assert llm._bump_cap(
        {"generationConfig": {"max_output_tokens": 65536}})[
            "generationConfig"]["max_output_tokens"] == 65536


# ---- cache identity ----------------------------------------------------

def test_refiner_cache_key_includes_model(monkeypatch, tmp_path):
    """Switching QBANK_LLM_MODEL must actually re-ask the model — the
    old key ignored the model, so a new model served the old answer."""
    calls = []
    monkeypatch.setattr(llm, "_post", lambda u, p, k: (
        calls.append(p), {"candidates": [{"content": {"parts": [
            {"text": TABLE}]}}]})[1])

    class _Book:                                  # minimal doc stand-in
        class doc:
            name = "mini.pdf"

    for model in ("model-A", "model-B"):
        llm.refiner(cache_dir=tmp_path, model=model, key="K")(
            _Book, [1], TABLE)
    assert len(calls) == 2                        # no cross-model cache hit
    llm.refiner(cache_dir=tmp_path, model="model-B", key="K")(_Book, [1], TABLE)
    assert len(calls) == 2                        # same model: cached


# ---- model preflight ---------------------------------------------------

class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_check_model_ok(monkeypatch, capsys):
    monkeypatch.setenv("QBANK_LLM_FALLBACK_MODELS", "off")
    monkeypatch.setattr(llm.urllib.request, "urlopen",
                        lambda req, timeout=30: _Resp(
                            {"name": "models/m", "displayName": "Gemini M"}))
    assert llm.check_model("m", key="K") is True
    assert "model check OK: m (Gemini M)" in capsys.readouterr().out


def test_check_model_lists_usable_ids_on_404(monkeypatch, capsys):
    def fake_open(req, timeout=30):
        url = req.full_url
        if url.endswith("/m"):
            raise _http_error(404, {"error": {
                "status": "NOT_FOUND", "message": "models/m is not found"}})
        return _Resp({"models": [
            {"name": "models/gemini-2.5-flash",
             "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embedding-001",
             "supportedGenerationMethods": ["embedContent"]},
        ]})

    monkeypatch.setenv("QBANK_LLM_FALLBACK_MODELS", "off")   # single model
    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_open)
    assert llm.check_model("m", key="K") is False
    out = capsys.readouterr().out
    assert "HTTP 404 NOT_FOUND" in out
    assert "models this key can use: gemini-2.5-flash" in out
    assert "embedding-001" not in out             # only generateContent


def test_check_model_without_key(monkeypatch, capsys):
    """No key anywhere -> a clear message, not a crash.

    The env is cleaned first: on a machine that HAS GEMINI_API_KEY set
    (the normal case when running the pipeline) the pool would answer
    the call and this test failed for the wrong reason.
    """
    for var in ("GEMINI_API_KEY", "GEMINI_API_KEYS"):
        monkeypatch.delenv(var, raising=False)
    for i in range(1, 21):
        monkeypatch.delenv(f"GEMINI_API_KEY_{i}", raising=False)
    monkeypatch.setattr(llm.keypool, "get_pool", lambda *a, **k: None)
    assert llm.check_model("m", key="") is False
    assert "no API key" in capsys.readouterr().out


# ---- rejected answers show a sample ------------------------------------

def test_rejected_answer_is_sampled(capsys):
    t = {"table_id": "T9", "markdown": TABLE, "validation": {}}
    assert refine_mod.refine_table(
        t, None, lambda book, pgs, md: "here is your table, hope it helps!",
        only="all", memo={}) == "invalid"
    out = capsys.readouterr().out
    assert "T9: answer rejected" in out and "here is your table" in out
    assert t["markdown"] == TABLE                 # deterministic one kept


def test_reject_sampling_is_capped(monkeypatch, capsys):
    monkeypatch.setattr(refine_mod, "_reject_samples_left", 0)
    t = {"table_id": "T9", "markdown": TABLE, "validation": {}}
    refine_mod.refine_table(t, None, lambda b, p, md: "not a table", memo={})
    assert capsys.readouterr().out == ""


# ---- one unambiguous table is salvaged from a chatty answer ------------

# same content as TABLE, rearranged (a real rearrangement: the
# content envelope refuses answers that add/remove characters)
NEW = "| B | A |\n|---|---|\n| y | x |"


@pytest.mark.parametrize("answer", [
    "```markdown\n" + NEW + "\n```",                    # fenced
    "Here is the rearranged table:\n\n" + NEW,          # preamble
    "Sure!\n" + NEW + "\nLet me know if you need more.",  # chatter both ends
])
def test_salvage_unambiguous_table(answer):
    assert refine_mod.salvage_table(answer) == NEW


@pytest.mark.parametrize("answer", [
    NEW + "\nAnd the second one:\n" + TABLE,   # which one is THE table?
    "here is your table, hope it helps!",      # no table at all
    "",
])
def test_salvage_refuses_ambiguity(answer):
    assert refine_mod.salvage_table(answer) is None


def test_refine_table_salvages_wrapper_answer(capsys):
    t = {"table_id": "T2", "markdown": TABLE, "validation": {}}
    verdict = refine_mod.refine_table(
        t, None, lambda b, p, md: "Here is the rearranged table:\n\n" + NEW,
        memo={})
    assert verdict == "replaced"
    assert t["markdown"] == NEW
    qa = t["validation"]["table_qa"]
    assert qa["refined_by_gemini"] is True
    assert qa["salvaged_from_wrapper"] is True       # provenance kept
    assert t["validation"]["pre_gemini_markdown"] == TABLE
    assert "salvaged the single table block" in capsys.readouterr().out


def test_refine_table_keeps_deterministic_on_ambiguity(capsys):
    t = {"table_id": "T3", "markdown": TABLE, "validation": {}}
    verdict = refine_mod.refine_table(
        t, None, lambda b, p, md: NEW + "\nAnd also:\n" + NEW, memo={})
    assert verdict == "invalid" and t["markdown"] == TABLE
    assert "answer rejected" in capsys.readouterr().out


# ---- run_book preflight ------------------------------------------------

def test_run_book_warns_when_model_check_fails(tmp_path, monkeypatch, capsys):
    """A model id this key cannot use must be called out BEFORE the run
    starts, not discovered from a book full of raw tables."""
    from qbank import config
    from qbank import run
    from test_mini_book import _build_book
    out = tmp_path / "out"
    for name, val in (("OUTPUT_ROOT", out), ("DATA_DIR", out / "data"),
                      ("ASSETS_DIR", out / "assets" / "questions"),
                      ("SPLIT_DIR", out / "split"),
                      ("SUBJECTS_DIR", out / "subjects"),
                      ("STATE_FILE", out / "state.json")):
        monkeypatch.setattr(config, name, val)
    pdf = tmp_path / "mini.pdf"
    _build_book(pdf)
    from qbank import llm as llm_mod
    monkeypatch.setattr(llm_mod, "enabled", lambda: True)
    monkeypatch.setattr(llm_mod, "check_model", lambda *a, **k: False)
    monkeypatch.setattr(llm_mod, "transcriber",
                        lambda cache_dir=None, **k: lambda b, p, x: None)
    monkeypatch.setattr(llm_mod, "verifier",
                        lambda cache_dir=None, **k: lambda b, p, x, s: None)
    monkeypatch.setattr(llm_mod, "refiner",
                        lambda cache_dir=None, **k: lambda b, p, md: md)
    run.run_book(str(pdf), "TST", page_offset="auto", force=True,
                 output_root=out)
    assert "Gemini model check FAILED" in capsys.readouterr().out


def test_run_book_preflight_can_be_skipped(tmp_path, monkeypatch, capsys):
    from qbank import config
    from qbank import run
    from test_mini_book import _build_book
    out = tmp_path / "out"
    for name, val in (("OUTPUT_ROOT", out), ("DATA_DIR", out / "data"),
                      ("ASSETS_DIR", out / "assets" / "questions"),
                      ("SPLIT_DIR", out / "split"),
                      ("SUBJECTS_DIR", out / "subjects"),
                      ("STATE_FILE", out / "state.json")):
        monkeypatch.setattr(config, name, val)
    pdf = tmp_path / "mini.pdf"
    _build_book(pdf)
    from qbank import llm as llm_mod
    monkeypatch.setattr(llm_mod, "enabled", lambda: True)
    monkeypatch.setenv("QBANK_LLM_PREFLIGHT", "0")
    called = []
    monkeypatch.setattr(llm_mod, "check_model",
                        lambda *a, **k: called.append(1) or True)
    monkeypatch.setattr(llm_mod, "transcriber",
                        lambda cache_dir=None, **k: lambda b, p, x: None)
    monkeypatch.setattr(llm_mod, "verifier",
                        lambda cache_dir=None, **k: lambda b, p, x, s: None)
    monkeypatch.setattr(llm_mod, "refiner",
                        lambda cache_dir=None, **k: lambda b, p, md: md)
    run.run_book(str(pdf), "TST", page_offset="auto", force=True,
                 output_root=out)
    assert called == [] and "Gemini table pass enabled" in \
        capsys.readouterr().out


def test_check_model_falls_through_to_a_model_with_quota(monkeypatch, capsys):
    """Live failure this guards: the preflight asked the pool for a key
    WITHOUT naming a model, got the key-level "spent today" verdict and
    shut the whole Gemini pass off ("tables raw rahengi") — although the
    daily cap is per MODEL and a sibling model had quota left."""
    from qbank import keypool as kp

    monkeypatch.setenv("QBANK_LLM_FALLBACK_MODELS", "fb")
    pool = kp.KeyPool(["K1"], state_path=None)
    pool.note_429("quotaId: GenerateRequestsPerDayPerProjectPerModel"
                  "-FreeTier", model="m")          # primary spent
    monkeypatch.setattr(llm.urllib.request, "urlopen",
                        lambda req, timeout=30: _Resp(
                            {"name": "models/fb", "displayName": "Fallback"}))
    assert llm.check_model("m", pool=pool, key="K") is True
    out = capsys.readouterr().out
    assert "this run uses fb instead" in out
    assert "model check OK: fb (Fallback)" in out


# ---- model provenance --------------------------------------------------
def test_the_transcription_cache_is_keyed_on_the_model_and_prompt(
        tmp_path, monkeypatch):
    """Before this, the page-image transcription cache (T2) was keyed on
    the page and the box alone: an answer cached under one model could
    be served as another's, and a prompt change did not invalidate it.
    The key carries the model and the prompt fingerprint now (the
    rearrange/refine caches already did), and the stored payload names
    the model that ANSWERED, so provenance survives a replay."""
    import json as _json

    import pymupdf
    import qbank.llm as llm
    from qbank.textlayer import Book

    doc = pymupdf.open()
    doc.new_page(width=200, height=200)
    doc[0].insert_text((20, 60), "table cell text")
    path = tmp_path / "one.pdf"
    doc.save(str(path))
    book = Book(str(path))
    cache = tmp_path / "cache"
    box = (10.0, 10.0, 150.0, 120.0)

    key_a = llm._cache_path(cache, book, 1, box, "model-A")
    key_b = llm._cache_path(cache, book, 1, box, "model-B")
    assert key_a != key_b                      # per-model identity
    monkeypatch.setattr(llm, "PROMPT", llm.PROMPT + "\n(rule added)")
    assert llm._cache_path(cache, book, 1, box, "model-A") != key_a
    monkeypatch.undo()

    # a cached answer replays WITHOUT any network call, and names the
    # model that produced it
    key_a.parent.mkdir(parents=True, exist_ok=True)
    key_a.write_text(_json.dumps(
        {"model": "model-A", "rows": [["A"], ["1"]]}))
    calls = []
    monkeypatch.setattr(llm, "_post_adaptive",
                        lambda *a, **k: calls.append(a) or {"candidates": []})
    rows = llm.transcriber(cache_dir=cache, model="model-A",
                           key="K")(book, 1, box)
    assert rows == [["A"], ["1"]] and calls == []
    assert llm.last_model() == "model-A"


def test_the_answered_model_is_recorded_for_provenance():
    import qbank.llm as llm
    llm.note_model("gemini-x")
    assert llm.last_model() == "gemini-x"
    assert "gemini-x" in llm.models_used()

