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


class KeyPool:
    def __init__(self, keys, max_calls_per_day=DEFAULT_MAX_CALLS_PER_DAY,
                 state_path=None,
                 max_calls_per_minute=DEFAULT_MAX_CALLS_PER_MINUTE):
        self.keys = list(keys)
        self.max_calls = int(max_calls_per_day)
        self.max_minute = int(max_calls_per_minute)
        # rolling-60s call stamps per key: pacing state only, in-memory
        # (a 60 s window is meaningless across processes, so unlike the
        # daily counters it is never persisted to keypool_state.json)
        self._min = {f"key{i + 1}": [] for i in range(len(self.keys))}
        self.state_path = Path(state_path) if state_path else None
        self.day = time.strftime("%Y-%m-%d")
        self.st = {
            f"key{i + 1}": {"fp": _fp(k), "calls": 0, "status": "active",
                            "cooldown_until": 0}
            for i, k in enumerate(self.keys)
        }
        self.active = 0
        self._load()

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

    def label(self, idx: int | None = None) -> str:
        return f"key{(self.active if idx is None else idx) + 1}"

    # ---- rotation -----------------------------------------------------
    def _minute_wait(self, idx: int, now: float) -> float:
        """Seconds until this key has per-minute room (0 = right now).
        The rolling 60 s window keeps every key under the API's 15/min
        rate — the pool waits instead of tripping 429s."""
        stamps = [t for t in self._min[self.label(idx)] if now - t < 60.0]
        self._min[self.label(idx)] = stamps
        if len(stamps) < self.max_minute:
            return 0.0
        return max(0.0, 60.0 - (now - stamps[0]))

    def acquire(self) -> str:
        """The current usable key, advancing past exhausted/cooldown
        keys and preferring one with per-minute room. When every usable
        key is merely pacing-capped, waits for the window to slide;
        raises PoolExhausted only when no key can serve at all (daily
        caps spent or every key cooling down)."""
        self._rollover()
        while True:
            now = time.time()
            waits = []
            for i in range(len(self.keys)):
                idx = (self.active + i) % len(self.keys)
                s = self.st[self.label(idx)]
                if s["status"] == "exhausted":
                    continue
                if s["status"] == "cooldown" and s["cooldown_until"] > now:
                    continue
                if s["calls"] >= self.max_calls:
                    s["status"] = "exhausted"
                    continue
                w = self._minute_wait(idx, now)
                if w > 0:
                    waits.append(w)
                    continue
                self.active = idx
                return self.keys[idx]
            if not waits:
                raise PoolExhausted(f"all {len(self.keys)} keys spent today")
            wait = min(waits)
            print(f"[keypool] pacing cap ({self.max_minute}/min per key) "
                  f"hit on every usable key — waiting {wait:.1f}s",
                  flush=True)
            time.sleep(wait)

    def note_call(self):
        s = self.st[self.label()]
        s["calls"] += 1
        self._min[self.label()].append(time.time())   # pacing window
        if s["calls"] >= self.max_calls:
            s["status"] = "exhausted"
            print(f"[keypool] {self.label()} (fp {s['fp']}) hit daily cap "
                  f"({self.max_calls}) — advancing")
        self._save()

    def note_429(self, err_text: str = ""):
        """Classify a 429 from the Gemini error body: a per-MINUTE quota
        id means a burst — 60 s cooldown, key stays in the pool; a
        per-DAY quota means the bucket is spent — exhaust and advance."""
        s = self.st[self.label()]
        low = (err_text or "").lower()
        per_minute = "perminute" in low or "per minute" in low
        if per_minute and s["status"] != "exhausted":
            s["status"] = "cooldown"
            s["cooldown_until"] = time.time() + RPM_COOLDOWN_SECONDS
            print(f"[keypool] {self.label()} (fp {s['fp']}) rate-limited "
                  f"(per-minute) — cooling {RPM_COOLDOWN_SECONDS}s")
        else:
            s["status"] = "exhausted"
            print(f"[keypool] {self.label()} (fp {s['fp']}) daily quota "
                  f"exhausted — advancing")
        self._save()

    def note_bad_key(self, code: int, err_text: str = ""):
        """This key was rejected for THIS run (revoked key, quota 403,
        per-project block): park it and advance — the next key in the
        pool may still serve."""
        s = self.st[self.label()]
        s["status"] = "exhausted"
        print(f"[keypool] {self.label()} (fp {s['fp']}) rejected by the "
              f"API (HTTP {code}) — advancing to the next key")
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
                  f"pacing {minute}/min/key, cap {day}/day/key")
    return _POOL
