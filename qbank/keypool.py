"""Multi-key Gemini quota pool (adopted compact from the old pipeline).

WHY: the free-tier daily request cap is enforced PER GOOGLE CLOUD
PROJECT, not per key. Keys minted in separate projects have
independent buckets, so rotating across them multiplies the usable
daily quota. Hitting a key's brake now ADVANCES TO THE NEXT KEY
instead of ending the run; the run degrades to the deterministic
output only when every key in the pool is spent.

CONFIGURATION (env; forms 1 and 2 merged, order preserved, dupes
removed so pasting a key twice cannot double the apparent quota):

  1. GEMINI_API_KEYS        comma / whitespace / newline separated
  2. GEMINI_API_KEY_1..20   one variable per key
  3. GEMINI_API_KEY         the original single key (still works)

STATE: counters persist in <OUTPUT_DIR>/data/keypool_state.json under
NON-SECRET labels (key1..keyN) with an sha1[:8] fingerprint for log
correlation. Full key material is never written to disk or printed.
Day rollover uses the local date: Gemini refills at Pacific midnight
(12:30 IST), which is BEFORE local midnight, so a same-day reset can
be one cycle behind the real refill — deliberately conservative: the
pool under-counts available quota rather than over-counting it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

# A per-MINUTE 429 is a burst, not a dead key: park it briefly and
# keep using the rest of the pool.
RPM_COOLDOWN_SECONDS = 60
# a 429 whose body carries no per-day quota id is retried this many
# times (with a cooldown each time) before the key is called spent
VAGUE_429_LIMIT = 5
DEFAULT_MAX_CALLS_PER_DAY = 480
# The free tier allows 15 requests/minute per project: pace every key
# at 12 so bursts never trip a 429 (env QBANK_MAX_CALLS_PER_MINUTE
# overrides; N keys sustain N x the rate).
DEFAULT_MAX_CALLS_PER_MINUTE = 12


def _fp(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def discover_keys(env=None) -> list[str]:
    env = os.environ if env is None else env
    keys: list[str] = []
    raw = env.get("GEMINI_API_KEYS", "")
    keys += [k for k in raw.replace(",", " ").split() if k]
    for i in range(1, 21):
        k = (env.get(f"GEMINI_API_KEY_{i}") or "").strip()
        if k:
            keys.append(k)
    single = (env.get("GEMINI_API_KEY") or "").strip()
    if single:
        keys.append(single)
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


class PoolExhausted(RuntimeError):
    pass


# Gemini's free-tier daily quota is enforced PER PROJECT **AND PER
# MODEL** ("GenerateRequestsPerDayPerProjectPerModel-FreeTier", 500/day
# for gemini-3.5-flash-lite at the time of writing). A pool that tracks
# one bucket per KEY therefore gets it wrong the moment a second model
# is used (the run's QBANK_LLM_MODEL changes, or a fallback model is
# tried): the key is marked "spent today" while its bucket for the
# other model is still full — which is exactly what happened to the
# Microbiology run, where one model's cap stopped every remaining
# chapter although another model answered normally one second later.
# Buckets are therefore keyed (key, model). `model=None` keeps the old
# single-bucket behaviour for callers that do not care.
LEGACY_MODEL = "-"


def _model_key(model: str | None) -> str:
    return model or LEGACY_MODEL


class KeyPool:
    def __init__(self, keys, max_calls_per_day=DEFAULT_MAX_CALLS_PER_DAY,
                 state_path=None,
                 max_calls_per_minute=DEFAULT_MAX_CALLS_PER_MINUTE):
        self.keys = list(keys)
        self.max_calls = int(max_calls_per_day)
        self.max_minute = int(max_calls_per_minute)
        # rolling-60s call stamps per (key, model): pacing state only,
        # in-memory (a 60 s window is meaningless across processes, so
        # unlike the daily counters it is not persisted)
        self._min: dict = {f"key{i + 1}": {} for i in range(len(self.keys))}
        self.state_path = Path(state_path) if state_path else None
        self.day = time.strftime("%Y-%m-%d")
        self._vague_429: dict = {}       # (key, model) -> vague 429 count
        self.st = {
            f"key{i + 1}": {"fp": _fp(k), "calls": 0, "status": "active",
                            "cooldown_until": 0, "models": {}}
            for i, k in enumerate(self.keys)
        }
        self.active = 0
        self._load()

    # ---- one key+model bucket ---------------------------------------
    def _sync(self, idx: int) -> None:
        """Keep the key-level fields meaningful for readers that only
        know about a key (the CLI summary, older callers, the tests):
        the LEGACY bucket IS the key; with real model buckets the key
        counts as exhausted only when every one of them is."""
        s = self.st[self.label(idx)]
        legacy = s["models"].get(LEGACY_MODEL)
        if legacy is not None:
            s["status"] = legacy["status"]
            s["cooldown_until"] = legacy["cooldown_until"]
            s["calls"] = legacy["calls"]
            return
        buckets = list(s["models"].values())
        if buckets and all(b["status"] == "exhausted" for b in buckets):
            s["status"] = "exhausted"
        elif s["status"] == "exhausted":
            s["status"] = "active"

    def bucket(self, idx: int, model: str | None = None) -> dict:
        s = self.st[self.label(idx)]
        mk = _model_key(model)
        b = s["models"].setdefault(
            mk, {"calls": 0, "status": "active", "cooldown_until": 0})
        return b

    # ---- persistence ------------------------------------------------
    def _load(self):
        if not self.state_path or not self.state_path.exists():
            return
        try:
            saved = json.loads(self.state_path.read_text())
        except Exception:
            return
        if saved.get("day") != self.day:
            return                      # rollover: counters start fresh
        for label, s in (saved.get("keys") or {}).items():
            mine = self.st.get(label)
            # a saved entry is trusted only for the SAME key material
            if mine is not None and mine["fp"] == s.get("fp"):
                mine["calls"] = int(s.get("calls", 0))
                mine["status"] = s.get("status", "active")
                mine["cooldown_until"] = float(s.get("cooldown_until", 0))
                models = s.get("models") or {}
                for m, b in models.items():
                    mine["models"][m] = {
                        "calls": int(b.get("calls", 0)),
                        "status": b.get("status", "active"),
                        "cooldown_until": float(b.get("cooldown_until", 0)),
                    }
                if not models:
                    # a state file written before per-model buckets: the
                    # old numbers describe the key as a whole
                    mine["models"][LEGACY_MODEL] = {
                        "calls": int(s.get("calls", 0)),
                        "status": s.get("status", "active"),
                        "cooldown_until": float(s.get("cooldown_until", 0))}
        act = saved.get("active", 0)
        self.active = act if isinstance(act, int) and 0 <= act < len(self.keys) else 0

    def _save(self):
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"day": self.day, "active": self.active,
                                   "keys": self.st}))
        tmp.replace(self.state_path)

    # ---- day rollover ------------------------------------------------
    def _rollover(self):
        today = time.strftime("%Y-%m-%d")
        if today != self.day:
            self.day = today
            for s in self.st.values():
                s.update(calls=0, status="active", cooldown_until=0)
                s["models"] = {}
            self._vague_429.clear()

    def label(self, idx: int | None = None) -> str:
        return f"key{(self.active if idx is None else idx) + 1}"

    # ---- rotation -----------------------------------------------------
    def _minute_wait(self, idx: int, now: float,
                     model: str | None = None) -> float:
        """Seconds until this key has per-minute room (0 = right now).
        The rolling 60 s window keeps every key under the API's
        requests-per-minute rate — the pool waits instead of tripping
        429s."""
        mk = _model_key(model)
        lab = self.label(idx)
        stamps = [t for t in self._min.setdefault(lab, {}).get(mk, [])
                  if now - t < 60.0]
        self._min[lab][mk] = stamps
        if len(stamps) < self.max_minute:
            return 0.0
        return max(0.0, 60.0 - (now - stamps[0]))

    def acquire(self, model: str | None = None) -> str:
        """The current usable key, advancing past exhausted keys and
        preferring one with per-minute room.

        A key that tripped a per-minute 429 is COOLING DOWN, not spent:
        the pool waits for it (up to its cooldown) and continues. That
        distinction is not cosmetic — with a single-key pool the old
        code `continue`d over cooling keys, found no wait candidates and
        raised PoolExhausted, which the caller reports as "all keys
        spent today" and every remaining table of the book came back
        "no-answer" for the rest of the run. Three chapters of the real
        book were degraded exactly that way (~60 s of cooldown each)
        before this fix. PoolExhausted now means what it says: every key
        is at its daily cap or rejected by the API."""
        self._rollover()
        while True:
            now = time.time()
            waits = []
            cooling = []
            for i in range(len(self.keys)):
                idx = (self.active + i) % len(self.keys)
                s = self.st[self.label(idx)]
                b = self.bucket(idx, model)
                if _model_key(model) == LEGACY_MODEL:
                    # no model given -> the fields on the key entry ARE
                    # the bucket (pre-per-model callers and state files)
                    b["status"], b["calls"] = s["status"], s["calls"]
                    b["cooldown_until"] = s["cooldown_until"]
                if s["status"] == "exhausted" and b["status"] == "exhausted":
                    continue
                if b["status"] == "exhausted":
                    continue
                if b["calls"] >= self.max_calls:
                    b["status"] = "exhausted"
                    self._sync(idx)
                    print(f"[keypool] {self.label(idx)} (fp {s['fp']}) hit "
                          f"its {self.max_calls}-call budget for "
                          f"{_model_key(model)} — advancing")
                    continue
                if s["status"] == "cooldown" and s["cooldown_until"] > now:
                    cooling.append(s["cooldown_until"] - now)
                    continue
                if b["status"] == "cooldown" and b["cooldown_until"] > now:
                    cooling.append(b["cooldown_until"] - now)
                    continue
                w = self._minute_wait(idx, now, model)
                if w > 0:
                    waits.append(w)
                    continue
                self.active = idx
                return self.keys[idx]
            if not waits and not cooling:
                raise PoolExhausted(f"all {len(self.keys)} keys spent today")
            wait = min(waits + cooling)
            why = ("rate-limit cooldown" if not waits
                   else f"pacing cap ({self.max_minute}/min per key)")
            print(f"[keypool] every usable key is waiting ({why}) — "
                  f"waiting {wait:.1f}s", flush=True)
            self._save()
            time.sleep(wait)

    def note_call(self, model: str | None = None):
        s = self.st[self.label()]
        b = self.bucket(self.active, model)
        self._vague_429.pop((s.get("fp", ""), _model_key(model)), None)
        s["calls"] += 1
        b["calls"] += 1
        self._min.setdefault(self.label(), {}).setdefault(
            _model_key(model), []).append(time.time())    # pacing window
        if b["calls"] >= self.max_calls:
            b["status"] = "exhausted"
            print(f"[keypool] {self.label()} (fp {s['fp']}) hit its "
                  f"{self.max_calls}-call budget for {_model_key(model)} "
                  f"— advancing")
        self._sync(self.active)
        self._save()

    def note_429(self, err_text: str = "", model: str | None = None):
        """Classify a 429 from the Gemini error body: a per-MINUTE quota
        id means a burst — 60 s cooldown, key stays in the pool; a
        per-DAY quota means the bucket is spent — exhaust and advance."""
        s = self.st[self.label()]
        b = self.bucket(self.active, model)
        low = (err_text or "").lower()
        per_day = ("perday" in low or "per day" in low
                   or "requests per day" in low)
        if per_day:
            # THIS model's daily bucket, not the key: another model can
            # still serve (the message names the model it applies to)
            b["status"] = "exhausted"
            print(f"[keypool] {self.label()} (fp {s['fp']}) daily quota "
                  f"exhausted for {_model_key(model)} — advancing "
                  f"(other models keep their own quota)")
            self._sync(self.active)
        else:
            # everything else (per-minute burst, a vague 429, an unparsed
            # body) is treated as TRANSIENT: cool the key and keep going.
            # Killing a key for the day on a message we did not
            # understand threw away the rest of the book's quota.
            key_fp = (s.get("fp", ""), _model_key(model))
            self._vague_429[key_fp] = self._vague_429.get(key_fp, 0) + 1
            if self._vague_429[key_fp] > VAGUE_429_LIMIT:
                b["status"] = "exhausted"
                print(f"[keypool] {self.label()} (fp {s['fp']}) 429 without a "
                      f"quota id {self._vague_429[key_fp]}x for "
                      f"{_model_key(model)} — treating as spent for today")
            else:
                b["status"] = "cooldown"
                b["cooldown_until"] = time.time() + RPM_COOLDOWN_SECONDS
                print(f"[keypool] {self.label()} (fp {s['fp']}) rate-limited "
                      f"— cooling {RPM_COOLDOWN_SECONDS}s (the pool waits, "
                      f"it does not give up)")
        self._sync(self.active)
        self._save()

    def note_bad_key(self, code: int, err_text: str = "", model=None):
        """This key was rejected for THIS run (revoked key, quota 403,
        per-project block): park it and advance — the next key in the
        pool may still serve."""
        s = self.st[self.label()]
        s["status"] = "exhausted"
        b = self.bucket(self.active, model)
        b["status"] = "exhausted"
        print(f"[keypool] {self.label()} (fp {s['fp']}) rejected by the "
              f"API (HTTP {code}) — advancing to the next key")
        self._sync(self.active)
        self._save()

    # ---- reporting -----------------------------------------------------
    def summary(self) -> dict:
        return {"day": self.day, "active": self.label(),
                "max_calls_per_day": self.max_calls,
                "max_calls_per_minute": self.max_minute,
                "keys": [{"label": l, **v} for l, v in sorted(self.st.items())]}

    def summary_text(self) -> str:
        return ", ".join(f"{l}={v['status']}({v['calls']})"
                         for l, v in sorted(self.st.items()))


_POOL: KeyPool | None = None


def get_pool(output_root=None, env=None) -> KeyPool | None:
    """Process-wide pool from env; None when no key is configured."""
    global _POOL
    if _POOL is None:
        keys = discover_keys(env)
        if keys:
            root = Path(output_root
                        or os.environ.get("OUTPUT_DIR", ".")).expanduser()
            day = int(os.environ.get("QBANK_MAX_CALLS_PER_DAY",
                                     str(DEFAULT_MAX_CALLS_PER_DAY)))
            minute = int(os.environ.get(
                "QBANK_MAX_CALLS_PER_MINUTE",
                str(DEFAULT_MAX_CALLS_PER_MINUTE)))
            _POOL = KeyPool(keys, day,
                            state_path=root / "data" / "keypool_state.json",
                            max_calls_per_minute=minute)
            print(f"[keypool] {len(keys)} key(s) in pool "
                  f"(fps: {', '.join(v['fp'] for v in _POOL.st.values())}); "
                  f"pacing {minute}/min/key x {len(keys)} keys = "
                  f"{minute * len(keys)}/min effective, "
                  f"cap {day}/day/key")
    return _POOL
