"""Multi-key Gemini pool: discovery/dedupe, rotation, 429
classification, persistence, and the llm._call wiring."""
import io
import json
import time
import urllib.error

from qbank import keypool
from qbank.keypool import KeyPool, PoolExhausted, discover_keys


def test_discover_merges_and_dedupes():
    env = {"GEMINI_API_KEYS": "a1,a2 a3\na1",
           "GEMINI_API_KEY_1": "b1",
           "GEMINI_API_KEY_2": "b2",
           "GEMINI_API_KEY": "a2"}
    assert discover_keys(env) == ["a1", "a2", "a3", "b1", "b2"]


def test_single_key_still_works():
    assert discover_keys({"GEMINI_API_KEY": "solo"}) == ["solo"]


def test_daily_cap_advances_then_exhausts():
    pool = KeyPool(["k1", "k2"], max_calls_per_day=2)
    assert pool.acquire() == "k1"
    pool.note_call()
    pool.note_call()                      # cap reached -> exhausted
    assert pool.acquire() == "k2"
    pool.note_429("PerDayPerProjectPerModelFreeTierRequests exceeded")
    try:
        pool.acquire()
        raise AssertionError("expected PoolExhausted")
    except PoolExhausted:
        pass


def test_per_minute_429_cools_down_not_exhausts():
    pool = KeyPool(["k1", "k2"], max_calls_per_day=100)
    pool.note_429("Resource exhausted: PerMinutePerProjectPerModel...")
    assert pool.st["key1"]["status"] == "cooldown"
    assert pool.acquire() == "k2"         # cooled key is skipped
    pool.st["key1"]["cooldown_until"] = time.time() - 1
    pool.active = 0                       # pointer back at key1
    assert pool.acquire() == "k1"         # cooled down -> usable again
    assert pool.st["key1"]["status"] != "exhausted"


def test_state_persists_and_never_leaks_key_material(tmp_path):
    sp = tmp_path / "keypool_state.json"
    p1 = KeyPool(["secret-key-A", "secret-key-B"], max_calls_per_day=10,
                 state_path=sp)
    p1.note_call()
    p1.note_call()
    p2 = KeyPool(["secret-key-A", "secret-key-B"], max_calls_per_day=10,
                 state_path=sp)
    assert p2.st["key1"]["calls"] == 2
    raw = sp.read_text()
    assert "secret-key-A" not in raw and "secret-key-B" not in raw
    assert p2.st["key1"]["fp"] in raw     # fingerprint yes, key no


def test_state_ignores_different_key_material(tmp_path):
    sp = tmp_path / "keypool_state.json"
    KeyPool(["aaa", "bbb"], max_calls_per_day=10,
            state_path=sp).note_call()
    p2 = KeyPool(["ccc", "ddd"], max_calls_per_day=10, state_path=sp)
    assert p2.st["key1"]["calls"] == 0    # different keys: fresh start


def test_day_rollover_resets_counters(monkeypatch):
    pool = KeyPool(["k1"], max_calls_per_day=1)
    pool.note_call()
    assert pool.st["key1"]["status"] == "exhausted"
    monkeypatch.setattr(keypool.time, "strftime",
                        lambda fmt: "2030-01-01")
    assert pool.acquire() == "k1"         # new day -> usable again
    assert pool.st["key1"]["calls"] == 0


# ---- llm wiring ----------------------------------------------------------

def _resp(rows):
    return {"candidates": [{"content": {"parts": [
        {"text": json.dumps({"rows": rows})}]}}]}


def _http429(body: bytes):
    return urllib.error.HTTPError(
        "http://x", 429, "Too Many Requests", {}, io.BytesIO(body))


def test_call_rotates_key_on_daily_429(monkeypatch):
    from qbank import llm
    pool = KeyPool(["k1", "k2"], max_calls_per_day=100)
    seen = []

    def fake_post(url, payload, key):
        seen.append(key)
        if key == "k1":
            raise _http429(b'{"error":{"message":"quota exceeded for '
                           b'metric PerDayPerProjectPerModel"}}')
        return _resp([["a", "b"]])

    monkeypatch.setattr(llm, "_post", fake_post)
    rows = llm._call(pool, "", "m", {})
    assert rows == [["a", "b"]]
    assert seen == ["k1", "k2"]
    assert pool.st["key1"]["status"] == "exhausted"
    assert pool.st["key2"]["calls"] == 1


def test_call_gives_up_when_whole_pool_dead(monkeypatch):
    from qbank import llm
    pool = KeyPool(["k1", "k2"], max_calls_per_day=100)

    def fake_post(url, payload, key):
        raise _http429(b'{"error":{"message":"PerDayPerProject quota"}}')

    monkeypatch.setattr(llm, "_post", fake_post)
    assert llm._call(pool, "", "m", {}) is None
    assert pool.st["key1"]["status"] == "exhausted"
    assert pool.st["key2"]["status"] == "exhausted"


def test_enabled_follows_keypool_forms(monkeypatch):
    from qbank import llm
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("QBANK_LLM_TABLES", raising=False)
    assert not llm.enabled()
    monkeypatch.setenv("GEMINI_API_KEY_3", "k3")
    assert llm.enabled()
    monkeypatch.setenv("QBANK_LLM_TABLES", "0")
    assert not llm.enabled()


# ---- per-minute pacing (15/min free-tier rate, safe boundary) ----------

def test_minute_pacing_prefers_key_with_room():
    """A pacing-capped key is skipped, NOT exhausted — the pool keeps
    serving from the next key with room instead of waiting."""
    pool = KeyPool(["k1", "k2"], max_calls_per_day=100,
                   max_calls_per_minute=2)
    assert pool.acquire() == "k1"
    pool.note_call()
    pool.note_call()                      # k1 spent its 2/min
    assert pool.acquire() == "k2"         # room -> advance, no waiting
    assert pool.st["key1"]["status"] == "active"


def test_minute_pacing_waits_when_every_key_capped(monkeypatch, capsys):
    """No key with room: the pool sleeps for the window instead of
    tripping a 429 (fake clock, so no real 60 s wait)."""
    now = [1000.0]
    slept = []

    def fake_sleep(s):
        slept.append(s)
        now[0] += s

    monkeypatch.setattr(keypool.time, "time", lambda: now[0])
    monkeypatch.setattr(keypool.time, "sleep", fake_sleep)
    pool = KeyPool(["k1"], max_calls_per_day=100, max_calls_per_minute=1)
    assert pool.acquire() == "k1"
    pool.note_call()                      # spends the 1/min at t=1000
    assert pool.acquire() == "k1"         # waits for the window to slide
    assert slept == [60.0]
    assert "pacing cap" in capsys.readouterr().out


# ---- key-fault rotation --------------------------------------------------

def _http_err(code, body: bytes):
    return urllib.error.HTTPError(
        "http://x", code, "Bad", {}, io.BytesIO(body))


def test_call_rotates_on_revoked_key(monkeypatch, capsys):
    """One revoked key must not kill the run — the pool advances and
    the next key serves."""
    from qbank import llm
    pool = KeyPool(["k1", "k2"], max_calls_per_day=100)
    seen = []

    def fake_post(url, payload, key):
        seen.append(key)
        if key == "k1":
            raise _http_err(400, b'{"error":{"status":"INVALID_ARGUMENT",'
                                 b'"message":"API key not valid. Pass a '
                                 b'valid key."}}')
        return _resp([["a"]])

    monkeypatch.setattr(llm, "_post", fake_post)
    assert llm._call(pool, "", "m", {}) == [["a"]]
    assert seen == ["k1", "k2"]
    assert pool.st["key1"]["status"] == "exhausted"
    assert "advancing to the next key" in capsys.readouterr().out


def test_call_rotates_on_quota_403(monkeypatch, capsys):
    """Quota exhaustion arriving as 403 rotates exactly like a 429."""
    from qbank import llm
    pool = KeyPool(["k1", "k2"], max_calls_per_day=100)
    seen = []

    def fake_post(url, payload, key):
        seen.append(key)
        if key == "k1":
            raise _http_err(403, b'{"error":{"status":"PERMISSION_DENIED",'
                                 b'"message":"Quota exceeded for quota '
                                 b'metric"}}')
        return _resp([["a"]])

    monkeypatch.setattr(llm, "_post", fake_post)
    assert llm._call(pool, "", "m", {}) == [["a"]]
    assert seen == ["k1", "k2"]
    assert pool.st["key1"]["status"] == "exhausted"
    assert "advancing to the next key" in capsys.readouterr().out


def test_call_reports_pool_exhausted_not_network(monkeypatch, capsys):
    """A spent pool must say so — the old generic handler blamed the
    network for an empty pool."""
    from qbank import llm

    def boom(url, payload, key):
        raise AssertionError("no HTTP attempt with a dead pool")

    monkeypatch.setattr(llm, "_post", boom)
    monkeypatch.setenv("QBANK_LLM_FALLBACK_MODELS", "off")
    pool = KeyPool(["k1"], max_calls_per_day=0)
    assert llm._call(pool, "", "m", {}) is None
    out = capsys.readouterr().out
    assert "pool exhausted" in out and "network unreachable" not in out
    pool = KeyPool(["k1"], max_calls_per_day=0)
    assert llm._call_text(pool, "", "m2", {}) is None
    out = capsys.readouterr().out
    assert "pool exhausted" in out and "network unreachable" not in out


def test_call_does_not_burn_keys_on_404(monkeypatch, capsys):
    """A wrong model id fails identically on every key — report once,
    rotate never."""
    from qbank import llm
    pool = KeyPool(["k1", "k2"], max_calls_per_day=100)
    calls = []

    def fake_post(url, payload, key):
        calls.append(key)
        raise _http_err(404, b'{"error":{"status":"NOT_FOUND",'
                             b'"message":"models/m is not found"}}')

    monkeypatch.setattr(llm, "_post", fake_post)
    monkeypatch.setenv("QBANK_LLM_FALLBACK_MODELS", "off")
    assert llm._call(pool, "", "m", {}) is None
    assert calls == ["k1"]
    assert pool.st["key1"]["status"] == "active"
    assert "QBANK_LLM_MODEL" in capsys.readouterr().out


def test_throughput_scales_with_key_count(monkeypatch):
    """Jitni keys, utna rate: 3 keys x 2/min serves 6 back-to-back
    calls with ZERO waiting (one key's cap never blocks the pool
    while a sister key has room); only the 7th call waits."""
    now = [1000.0]
    slept = []

    def fake_sleep(s):
        slept.append(s)
        now[0] += s

    monkeypatch.setattr(keypool.time, "time", lambda: now[0])
    monkeypatch.setattr(keypool.time, "sleep", fake_sleep)
    pool = KeyPool(["k1", "k2", "k3"], max_calls_per_day=100,
                   max_calls_per_minute=2)
    used = []
    for _ in range(6):
        used.append(pool.acquire())
        pool.note_call()
    assert slept == []                        # 3 keys x 2/min, no wait
    assert sorted(set(used)) == ["k1", "k2", "k3"]  # every key served
    # 7th: all capped -> waits, then serves from the active pointer
    assert pool.acquire() == "k3"
    assert slept == [60.0]                    # then the window slides


# ---- cooldown must WAIT, never abandon the run -------------------------

def test_cooling_key_is_waited_for_not_abandoned(tmp_path, monkeypatch):
    """A per-minute 429 puts the key in a 60 s cooldown. The pool must
    WAIT for it: with one key, giving up meant 'pool exhausted — all 1
    keys spent today' and every later table of the book shipped without
    the model pass (seen on the real Microbiology run)."""
    from qbank import keypool

    pool = keypool.KeyPool(["K1"], state_path=tmp_path / "s.json",
                           max_calls_per_minute=12)
    pool.note_429("quotaId: GenerateRequestsPerMinutePerProjectPerModel")
    assert pool.st["key1"]["status"] == "cooldown"

    slept = []
    monkeypatch.setattr(keypool.time, "sleep", lambda s: slept.append(s))
    # the cooldown is a few seconds here, not a minute: fast test
    pool.st["key1"]["cooldown_until"] = keypool.time.time() + 3.0
    assert pool.acquire() == "K1"
    assert slept and 0 < slept[0] <= 3.0


def test_daily_quota_429_still_exhausts_the_key(tmp_path):
    from qbank import keypool

    pool = keypool.KeyPool(["K1"], state_path=tmp_path / "s.json")
    pool.note_429("quotaId: GenerateRequestsPerDayPerProjectPerModel")
    assert pool.st["key1"]["status"] == "exhausted"
    try:
        pool.acquire()
    except keypool.PoolExhausted as e:
        assert "spent today" in str(e)
    else:                                     # pragma: no cover
        raise AssertionError("a spent key must raise PoolExhausted")


def test_vague_429_is_transient_until_it_repeats(tmp_path):
    from qbank import keypool

    pool = keypool.KeyPool(["K1"], state_path=tmp_path / "s.json")
    pool.note_429("Resource has been exhausted")          # no quota id
    assert pool.st["key1"]["status"] == "cooldown"        # transient
    for _ in range(keypool.VAGUE_429_LIMIT):
        pool.note_429("Resource has been exhausted")
    assert pool.st["key1"]["status"] == "exhausted"       # now it counts


def test_quota_is_per_model_not_per_key(tmp_path):
    """The real cap is 'requests per DAY per PROJECT per MODEL'. One
    model running out must NOT stop the run: the same key still serves
    another model. (Seen live: gemini-3.5-flash-lite at 500/day while
    gemini-3.1-flash-lite answered normally seconds later.)"""
    from qbank import keypool

    pool = keypool.KeyPool(["K1"], state_path=tmp_path / "s.json")
    pool.note_429("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                  model="model-a")
    assert pool.bucket(0, "model-a")["status"] == "exhausted"
    # the other model's bucket is untouched -> acquire() still serves
    assert pool.acquire("model-b") == "K1"
    try:
        pool.acquire("model-a")
    except keypool.PoolExhausted:
        pass
    else:                                       # pragma: no cover
        raise AssertionError("model-a must be exhausted")
    # and the per-model counters are separate
    pool.note_call("model-b")
    assert pool.bucket(0, "model-b")["calls"] == 1
    assert pool.bucket(0, "model-a")["calls"] == 0


def test_a_spent_model_falls_through_to_the_next_model(monkeypatch, capsys):
    """Live failure this guards: gemini-3.5-flash-lite hit its 500/day
    PER MODEL cap at 21:00 and every remaining chapter was recorded as
    "no answer", although gemini-3.1-flash-lite answered 200 at once.
    One logical call must therefore walk a model chain instead of
    giving up on the whole run."""
    from qbank import llm

    pool = KeyPool(["k1"], max_calls_per_day=100)
    monkeypatch.setenv("QBANK_LLM_FALLBACK_MODELS", "fb-model")
    seen = []

    def fake_post(url, payload, key):
        model = url.rsplit("/", 1)[-1].split(":")[0]
        seen.append(model)
        if model == "primary":
            raise _http_err(429, b'{"error":{"status":"RESOURCE_EXHAUSTED",'
                                 b'"message":"Quota exceeded",'
                                 b'"details":[{"@type":"type.googleapis.com/'
                                 b'google.rpc.QuotaFailure","violations":'
                                 b'[{"quotaId":"GenerateRequestsPerDayPer'
                                 b'ProjectPerModel-FreeTier"}]}]}}')
        return {"candidates": [{"content": {"parts": [
            {"text": json.dumps({"rows": [["a", "b"]]})}]}}]}

    monkeypatch.setattr(llm, "_post_adaptive",
                        lambda api, payload, key, model: fake_post(api, payload, key))
    monkeypatch.setattr(llm, "_post", fake_post)
    rows = llm._call(pool, "", "primary", {})
    assert rows == [["a", "b"]]
    assert seen == ["primary", "fb-model"]
    assert "fb-model answered instead" in capsys.readouterr().out
    # the spent bucket belongs to `primary` only: the key is not burnt
    assert pool.bucket(0, "primary")["status"] == "exhausted"
    assert pool.bucket(0, "fb-model")["status"] == "active"
    assert pool.bucket(0, "fb-model")["calls"] == 1


def test_model_chain_off_keeps_a_single_model(monkeypatch):
    from qbank import llm
    monkeypatch.setenv("QBANK_LLM_FALLBACK_MODELS", "off")
    assert llm._model_chain("only") == ["only"]
    monkeypatch.setenv("QBANK_LLM_FALLBACK_MODELS", "only, other ,other")
    assert llm._model_chain("only") == ["only", "other"]
