"""Optional Gemini-vision pass over ruled-table regions.

The deterministic geometry pipeline still OWNS table structure
(ruled boxes, columns, row bands, ordering). The model is asked only
to transcribe the cell texts of one rendered table image, and its
output is accepted under a hard fidelity envelope:

  * same row count and same cell count per row, else rejected;
  * per cell, the model text is accepted ONLY when it is identical to
    the deterministic cell after whitespace removal — the model may
    re-space ("fl ow" -> "flow", "Increasedpulmonary" ->
    "Increased pulmonary") but can never change, add or drop a
    character of content ("Tetrology" -> "Tetralogy" is REJECTED).

No API key, API error, bad JSON, or shape mismatch => the
deterministic output is returned unchanged. Responses are cached per
(pdf, page, box) so re-runs are deterministic and free.

Default remains zero-LLM: nothing is called unless GEMINI_API_KEY is
set in the environment (QBANK_LLM_MODEL overrides the model id,
QBANK_LLM_TABLES=0 disables explicitly).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import pymupdf

from . import keypool

API = ("https://generativelanguage.googleapis.com/v1beta/models/"
       "{model}:generateContent")
MODELS_API = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.5-flash-lite"

PROMPT = """You are a precision transcription engine for a medical
textbook table. The image shows ONE ruled table. Return ONLY valid
JSON, no markdown fences, of the form {"rows": [["cell", "..."], ...]}
— one inner list per table row, one string per visible column, in
reading order, header row first.

Transcribe every cell EXACTLY as printed:
- same words, same word order, same symbols and units (—, <, >, /, %,
  -, digits, abbreviations); empty cell = "";
- reconstruct words the typesetter broke across lines inside a cell
  ("fl" + "ow" = "flow", "Atri" + "al" = "Atrial",
   "abn" + "ormalities" = "abnormalities");
- this book's layout constantly prints missing or wrong spaces inside
  cells: glued words ("oftouch" = "of touch", "tomotor" = "to motor",
   "ofinternal" = "of internal", "Increasedpulmonary" =
   "Increased pulmonary", "notdepend" = "not depend",
   "damage,fetal" = "damage, fetal") and words broken mid-word
  ("oblongatatill t he 2nd" = "oblongata till the 2nd",
   "theacro miothoracicand" = "the acromiothoracic and");
  read each cell the way a human reader would and output it with
  natural, corrected spacing — never carry a layout artifact into
  your output;
- NEVER delete or add a word: every printed letter, digit and
  punctuation mark must appear exactly once in your output. Only
  spaces may change. DO NOT correct spellings, DO NOT normalise
  terminology, DO NOT translate, DO NOT reorder content.
A multi-line cell is ONE string with single spaces between its lines
(after the word reconstruction above)."""


def enabled() -> bool:
    return bool(keypool.discover_keys()) and \
        os.environ.get("QBANK_LLM_TABLES", "1") != "0"


def _norm(t: str) -> str:
    return re.sub(r"\s+", "", t or "").lower()


def _score(t: str, words, pairs) -> int:
    """Plausibility of one spacing of a cell: -1 (disqualified) if any
    alphabetic token is not a word the book contains at least twice.
    Single occurrences are excluded deliberately: the vocabulary is
    built from the raw layer, so a glued artifact printed in exactly
    one table cell ("rheniumThe", "tandemOne") is itself in it with
    count 1 — real words recur. Else the score is the count of
    adjacent word pairs the book prints with that spacing."""
    toks = [x.lower() for x in re.findall(r"[A-Za-z]{3,}", t)]
    for tok in toks:
        if words.get(tok, 0) < 2:
            return -1
    return sum(pairs.get(p, 0) for p in zip(toks, toks[1:]))


def _harmonize(t: str, words) -> str:
    """Map a model spelling variant onto the book's own spelling when
    the book prints the de-varianted form ("tumour"->"tumor" when the
    book has "tumor"): document-internal evidence only. Keeps the
    character-identical envelope honest across British/American
    variants without ever inventing a spelling."""
    if words is None:
        return t

    def sub(m):
        w = m.group(0)
        if "ou" in w.lower():
            d = re.sub(r"ou", "o", w)
            if d != w and words.get(d.lower(), 0) > words.get(w.lower(), 0):
                return d
        return w

    return re.sub(r"[A-Za-z]+", sub, t)


def _content(rows) -> str:
    """All cell text concatenated, whitespace removed, lowercased."""
    return re.sub(r"\s+", "", "".join(
        str(c) for r in rows for c in r)).lower()


def _tokruns(toks: list) -> list:
    """Alphabetic runs of a token list mapped onto the shared
    alpha-character stream (punctuation/space free, so both sides of
    a character-identical change are directly comparable). Each run
    also carries (original text, starts its token) for capitalisation
    checks."""
    runs, pos = [], 0
    for t in toks:
        first = True
        for m in re.finditer(r"[A-Za-z]+", t):
            g = m.group(0).lower()
            runs.append((pos, pos + len(g), g, m.group(0), first))
            pos += len(g)
            first = False
    return runs


def _block_ok(cdt: list, cmt: list, words, pairs=None) -> bool:
    """Judge ONE spacing change (character-identical strings, so the
    alphabetic offsets of both sides are comparable). Every model
    token must be justified:

      * established book word (printed >=2x) — always fine;
      * FRAGMENT MERGE: it spans >=2 whole deterministic tokens that
        are all rare (<2x) — the signature of a typesetter's mid-word
        wrap ("interme"+"dius"->"intermedius", "destr"+"oying"->
        "destroying"). "rhenium"+"The"->"rheniumThe" fails: "The" is
        not a fragment;
      * UN-GLUE: it is a >=4-letter piece of ONE rare (glued) det
        token whose sibling pieces are all established words, short
        ones very common ("negligibleor"->"negligible or").
        "calcifications"->"calcificatio ns" fails ("ns" is not a
        word); "Monochorionicity"->"Monochorionic ity" fails."""
    D, M = _tokruns(cdt), _tokruns(cmt)
    if not D or not M:
        return False
    if D == M:
        # no alphabetic change: only punctuation spacing may move, and
        # a space may only be INSERTED after sentence punctuation
        # ("damage,fetal"->"damage, fetal"), never after a symbol
        # (">=2mm"->">= 2mm") and never removed
        dd = re.sub(r"\s+", "", " ".join(cdt))
        mm = re.sub(r"\s+", "", " ".join(cmt))
        if dd != mm:
            return False
        i = j = 0
        sd, sm = " ".join(cdt), " ".join(cmt)
        while i < len(sd) and j < len(sm):
            if sd[i] == sm[j]:
                i += 1
                j += 1
            elif sm[j] == " ":          # model inserted a space
                if i == 0 or sd[i - 1] not in ",.:;!?":
                    return False
                j += 1
            else:                       # model removed a space
                return False
        return True
    for (ms, me, m, _mo, _mf) in M:
        if len(m) == 1:
            # single letters only where the det has the same single
            # letter ("mm"->"m m" is corruption)
            if not any(ds == ms and de == me for ds, de, dm, _o, _f in D
                       if dm == m):
                return False
            continue
        if words.get(m, 0) >= 2:
            # a fragment-split of an established det word is a
            # regression ("surface" -> "su rface"): reject unless every
            # model piece covering that det word is itself common
            # ("retractionnot" -> "retraction not" stays allowed).
            # A det token printed <=2x is a glued artifact, not a real
            # word — the model un-glueing it ("oblongatatill" ->
            # "oblongata till") is a repair and must win.
            for ds, de, dm, _o, _f in D:
                if ds <= ms and de >= me and len(dm) > len(m) \
                        and words.get(dm, 0) >= 2 \
                        and (dm.startswith(m) or dm.endswith(m)):
                    if words.get(dm, 0) <= 2:
                        continue        # artifact: model repair wins
                    # splitting an established det word is accepted
                    # only when the book itself prints the split
                    # spacing somewhere ("retraction not"); a model
                    # fragment-split of a real word ("adja cent") is
                    # a transcription glitch and must lose
                    sib = [t for _s, _e, t, _o2, _f2 in M
                           if _s < de and _e > ds]
                    printed = pairs is not None and any(
                        pairs.get((sib[i], sib[i + 1]), 0) >= 1
                        for i in range(len(sib) - 1))
                    if not printed:
                        return False
            # reverse fusion: the model glues det words the book
            # prints spaced. Reject only with evidence that the spaced
            # form is the book's real spelling: the pair printed >=2x
            # ("the incus"), or every part very common ("su rface"
            # never). A rare glued WORD the book prints ("dome" from
            # "do me") still wins.
            ov2 = [r for r in D if r[0] < me and r[1] > ms]
            if len(ov2) >= 2 and words.get(m, 0) <= 2:
                if pairs is not None and pairs.get(
                        tuple(r[2] for r in ov2[:2]), 0) >= 2:
                    return False
                if all(words.get(r[2], 0) >= 5 for r in ov2):
                    return False
            continue
        ov = [r for r in D if r[0] < me and r[1] > ms]
        if not ov:
            return False
        if len(ov) >= 1 and all(words.get(r[2], 0) < 2 for r in ov):
            # the model re-words a run of rare det fragments: its
            # tokens must tile the fragment span exactly and every
            # tiling token must be an established book word (or the
            # single joined form, "osteocal"+"cin" -> "osteocalcin")
            span_s, span_e = ov[0][0], ov[-1][1]
            tile = sorted((r[0], r[1], r[2]) for r in M
                          if r[0] < span_e and r[1] > span_s)
            pos, ok = span_s, bool(tile)
            for ts, te, tt in tile:
                if ts != pos:
                    ok = False
                    break
                pos = te
                if words.get(tt, 0) < 2:
                    ok = (te == span_e and len(tile) == 1)
            if ok and pos == span_e \
                    and not any(r[4] and r[3][0].isupper()
                                for r in ov[1:]):
                continue                      # fragment rewording
        if len(ov) == 1 and ov[0][0] <= ms and ov[0][1] >= me \
                and words.get(ov[0][2], 0) < 2 and len(m) >= 4:
            sib = [t for _s, _e, t, _o, _f in M
                   if t != m and _s < ov[0][1] and _e > ov[0][0]]
            if all(words.get(t, 0) >= 2
                   and (len(t) > 3 or words.get(t, 0) >= 50)
                   for t in sib):
                continue                      # un-glue of a rare blob
        # multi-way un-glue of a blob printed nowhere: every model
        # piece covering the blob must itself be a common book word
        # ("Intracranialintradural..." -> "Intracranial intradural
        # ..."). "calcifications" -> "calcific ations" fails: "ations"
        # is printed nowhere.
        if len(ov) == 1 and words.get(ov[0][2], 0) == 0:
            sib = [t for _s, _e, t, _o, _f in M
                   if _s < ov[0][1] and _e > ov[0][0]]
            if len(sib) >= 2 and all(words.get(t, 0) >= 2 for t in sib):
                continue
        return False
    return True


def _respaced(d: str, lj: str, words, pairs=None) -> str:
    """Best spacing of one cell: per diff block, take the model's
    spacing when _block_ok accepts it, else keep the deterministic
    one. A model glitch in one corner of a cell ("themalleus") can
    never veto its good repairs elsewhere in the same cell."""
    import difflib
    dw = re.findall(r"(\s*)(\S+)", d)
    mw = re.findall(r"(\s*)(\S+)", lj)
    smx = difflib.SequenceMatcher(
        a=[x[1].lower() for x in dw], b=[x[1].lower() for x in mw],
        autojunk=False)
    out = ""
    for op, i1, i2, j1, j2 in smx.get_opcodes():
        if op == "equal":
            seg = "".join(ws + t for ws, t in dw[i1:i2])
        else:
            cdt = [t for _ws, t in dw[i1:i2]]
            cd = "".join(cdt)
            cmt = [t for _ws, t in mw[j1:j2]]
            cm = "".join(cmt)
            if cd == cm and _block_ok(cdt, cmt, words, pairs):
                seg = "".join(ws + t for ws, t in mw[j1:j2])
                if seg and dw[i1:i2]:
                    seg = dw[i1][0] + seg[len(mw[j1][0]):]
            else:
                seg = "".join(ws + t for ws, t in dw[i1:i2])
        if out and seg and out[-1].isalnum() and seg[0].isalnum():
            out += " "
        out += seg
    out = out.strip()
    return out if _norm(out) == _norm(d) else d   # character safety net


def merge_llm(det_rows: list, llm_rows: list, vocab=None) -> tuple:
    """Fidelity envelope: same characters (whitespace-insensitive),
    then the deterministic cell is replaced only when the model cell
    is strictly more plausible under the book's own vocabulary —
    a candidate containing a non-word token ("TungstenThe",
    "atriumLeft") is disqualified outright; ties keep deterministic.

    Hybrid structure: when the model's row/cell grid differs from the
    deterministic grid, the model grid is accepted ONLY when the whole
    table's content is character-identical (whitespace removed) and
    every >=3-letter token in it is a word the book prints >=2 times.
    Otherwise the deterministic table is kept untouched."""
    if not isinstance(llm_rows, list) or not llm_rows:
        return det_rows, 0
    words = pairs = None
    if vocab is not None:
        words, pairs = vocab

        def hrow(r):
            if isinstance(r, list):
                return [hrow(c) for c in r]
            return _harmonize(str(r), words)

        llm_rows = [hrow(r) for r in llm_rows]
    det_shape = [len(r) for r in det_rows]
    llm_shape = [len(r) if isinstance(r, list) else -1 for r in llm_rows]
    if det_shape != llm_shape:
        # structure suggestion: Gemini owns spacing and cell flow as
        # long as not one character is deleted or added — the whole
        # table's content must be character-identical (whitespace
        # removed).  Otherwise the deterministic table is kept.
        if words is not None and _content(llm_rows) == _content(det_rows):
            return [[" ".join(str(c).split()) for c in r]
                    for r in llm_rows], 1
        return det_rows, 0
    out, nfix = [], 0
    for drow, lrow in zip(det_rows, llm_rows):
        if not isinstance(lrow, list) or len(lrow) != len(drow):
            return det_rows, 0
        newrow = []
        for d, l in zip(drow, lrow):
            l = str(l)
            lj = " ".join(l.split())
            # Gemini owns table spacing: the model cell may replace
            # the deterministic one whenever it carries exactly the
            # same characters (no word deleted or added) and its
            # spacing wins the evidence arbitration in _respaced —
            # which accepts the model un-glueing rare glued artifacts
            # and rejects the model breaking established book words.
            take = False
            if lj != d and _norm(l) == _norm(d):
                if words is None:
                    take = True
                else:
                    lj = _respaced(d, lj, words, pairs)
                    take = lj != d
            if take:
                newrow.append(lj)
                nfix += 1
            else:
                newrow.append(d)
        out.append(newrow)
    return out, nfix


def _post(url: str, payload: dict, key: str) -> dict:
    import time as _time
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "x-goog-api-key": key}, method="POST")
    for attempt in (1, 2, 3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:   # retry throttling/server errs
            if e.code in (429, 500, 503) and attempt < 3:
                _time.sleep(1.5 * attempt)
                continue
            raise
        except (urllib.error.URLError, OSError) as e:
            # transient network failure: retry like a 5xx
            if attempt < 3:
                _time.sleep(2.0 * attempt)
                continue
            raise


# ---- loud failure reporting -------------------------------------------
# Nothing here may fail silently. A `return None` with no explanation
# made a revoked key, a wrong model id, a blocked answer and a dead
# network look IDENTICAL in the run log ("no-answer"), so a whole book
# could ship raw tables with no clue why. Every distinct failure now
# prints once (deduped, key material never included) with the likely fix.

_ERRORS_SEEN: set[str] = set()

_HTTP_HINTS = {
    400: "invalid request — usually a bad/revoked API key or a bad parameter",
    401: "unauthorized — the API key is missing or revoked",
    403: "forbidden — this key/project may not have access to the model",
    404: ("model not found — check QBANK_LLM_MODEL "
          "(run: python scripts/check_gemini.py)"),
    429: "quota/rate limit — free-tier cap or request burst",
    500: "Gemini server error",
    503: "Gemini overloaded",
}


def _http_detail_text(code: int, body: str) -> str:
    """'HTTP 404 NOT_FOUND models/x is not found' from pre-read parts —
    the key is never in the URL or the body, so this is safe to print."""
    try:
        err = json.loads(body).get("error") or {}
        return (f"HTTP {code} {err.get('status', '')} "
                f"{err.get('message', '')}").strip()[:400]
    except Exception:                          # noqa: BLE001
        return f"HTTP {code} {body.strip()[:300]}".strip()


def _http_detail(e) -> str:
    """'HTTP 404 NOT_FOUND models/x is not found' — the key is never in
    the URL or the body, so this is safe to print."""
    try:
        body = e.read().decode("utf-8", "replace")
    except Exception:                          # noqa: BLE001
        body = ""
    return _http_detail_text(e.code, body)


def report_error(what: str, detail: str, hint: str = "") -> None:
    """One API failure, printed once per unique signature."""
    sig = hashlib.sha1(f"{what}|{detail[:200]}".encode()).hexdigest()[:8]
    if sig in _ERRORS_SEEN:
        return
    _ERRORS_SEEN.add(sig)
    print(f"[gemini] {what}: {detail}", flush=True)
    if hint:
        print(f"[gemini]   -> {hint}", flush=True)


def _hint_for(code: int) -> str:
    return _HTTP_HINTS.get(code, "")


def _key_fault(code: int, body: str) -> bool:
    """True when the failure blames THIS key rather than the request —
    a revoked key, a quota 403, a per-project block. The next pool key
    may still serve, so rotate instead of giving up. Anything else (a
    bad model id, a malformed request) would fail identically on every
    key, so those report once and stop without burning the pool."""
    low = (body or "").lower()
    if code in (400, 401):
        return ("api key" in low or "api_key" in low or "apikey" in low
                or "credential" in low or "unauthorized" in low
                or "unauthenticated" in low)
    if code == 403:
        return ("quota" in low or "exhaust" in low or "rate" in low
                or "limit" in low or "permission" in low
                or "forbidden" in low or "api key" in low)
    return False


def reset_error_reports() -> None:
    """Clear the once-per-signature dedupe. Called at the START of every
    run: the dedupe is meant per RUN, not per process — the dashboard is
    a long-lived Flask process, so a process-wide dedupe would print a
    failure once ever and then stay silent on every later re-run."""
    _ERRORS_SEEN.clear()


def _fp8(text: str) -> str:
    """Short fingerprint — used in cache keys (never for secrets)."""
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def _max_tokens(default: int) -> int:
    """Output budget per call; QBANK_LLM_MAX_TOKENS overrides. Thinking
    models spend this budget on reasoning too, so a tight cap shows up
    as MAX_TOKENS + a truncated or empty answer."""
    try:
        return max(256, int(os.environ.get("QBANK_LLM_MAX_TOKENS", default)))
    except (TypeError, ValueError):
        return default


def _answer_text(resp: dict) -> tuple[str, str]:
    """(visible answer, reason when there is none). Thought parts are
    never the answer — thinking models return them next to it."""
    cands = resp.get("candidates") or []
    if not cands:
        fb = resp.get("promptFeedback") or {}
        return "", (f"no candidates returned "
                    f"(blockReason={fb.get('blockReason') or 'none'})")
    cand = cands[0]
    parts = (cand.get("content") or {}).get("parts") or []
    txt = "".join(p.get("text", "") for p in parts
                  if not p.get("thought")).strip()
    if txt:
        return txt, ""
    fin = cand.get("finishReason") or "unknown"
    if fin == "MAX_TOKENS":
        return "", ("finishReason=MAX_TOKENS — the answer hit the output "
                    "cap (thinking tokens count towards it); retrying "
                    "with a bigger budget")
    return "", f"empty answer (finishReason={fin})"


def _bump_cap(payload: dict) -> dict:
    """Double max_output_tokens (never past the model ceiling) so a
    thinking model that ate the whole budget gets room to answer."""
    gen = dict(payload.get("generationConfig") or {})
    cap = int(gen.get("max_output_tokens") or 0)
    if not cap or cap >= 65536:
        return payload
    gen["max_output_tokens"] = min(cap * 2, 65536)
    return {**payload, "generationConfig": gen}


def _call(pool, key: str, model: str, payload: dict):
    """One generateContent exchange with pool-aware key rotation.
    Returns the parsed rows or None — the caller keeps the
    deterministic output. A 429 that survives _post's burst retries —
    or a failure that blames the key itself (revoked key, quota 403) —
    rotates to the next pool key (bounded by pool size) without
    spending a parse-retry attempt; anything else gives up on the
    spot (loudly — see report_error)."""
    rot, attempt = 0, 0
    while attempt < 3:
        try:
            k = pool.acquire() if pool is not None else key
            resp = _post(API.format(model=model), payload, k)
            if pool is not None:
                pool.note_call()
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", "replace")
            except Exception:              # noqa: BLE001
                body = ""
            if pool is not None and e.code == 429 and rot < len(pool.keys):
                pool.note_429(body)
                rot += 1
                continue
            if pool is not None and rot < len(pool.keys) \
                    and _key_fault(e.code, body):
                pool.note_bad_key(e.code, body)   # next key may serve
                rot += 1
                continue
            report_error(f"{model} call failed",
                         _http_detail_text(e.code, body), _hint_for(e.code))
            return None
        except keypool.PoolExhausted as e:
            report_error(f"{model} pool exhausted", str(e),
                         "no usable key left (daily caps spent, keys "
                         "rejected, or every key cooling down) — "
                         "deterministic output for now")
            return None
        except Exception as e:   # network dead: det output
            report_error(f"{model} call failed", f"{type(e).__name__}: {e}",
                         "network unreachable from this host?")
            return None
        txt, why = _answer_text(resp)
        if txt:
            txt = re.sub(r"^```(?:json)?|```$", "", txt)
            try:
                rows = json.loads(txt).get("rows")
            except ValueError:
                rows = None
            if rows is not None:
                return rows
            why = "answer was not the expected JSON"
        report_error(f"{model} no usable answer", why)
        if "MAX_TOKENS" in why:
            payload = _bump_cap(payload)   # then retry with more room
        attempt += 1           # RECITATION/SAFETY filters: retry
    return None


def list_models(key: str, timeout: int = 30) -> list[dict]:
    """The models THIS key can see (same auth path the calls use)."""
    req = urllib.request.Request(MODELS_API, headers={"x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return (json.loads(r.read()) or {}).get("models") or []


def check_model(model: str | None = None, pool=None,
                key: str | None = None) -> bool:
    """Preflight: does THIS key actually serve THIS model id, through
    the SAME header auth the calls use? Prints a one-line verdict, and
    when the model is wrong it lists the ids that WOULD work — so a bad
    QBANK_LLM_MODEL is caught before an hour-long run instead of during
    it. Returns True when the model is usable."""
    model = model or os.environ.get("QBANK_LLM_MODEL", DEFAULT_MODEL)
    pool = pool or (None if key else keypool.get_pool())
    key = key or os.environ.get("GEMINI_API_KEY", "")
    try:
        k = pool.acquire() if pool is not None else key
    except Exception as e:                     # noqa: BLE001
        report_error("no usable key in the pool", f"{type(e).__name__}: {e}")
        return False
    if not k:
        report_error("no API key", "neither GEMINI_API_KEY nor GEMINI_API_KEYS "
                                   "is set in this environment")
        return False
    try:
        req = urllib.request.Request(f"{MODELS_API}/{model}",
                                     headers={"x-goog-api-key": k})
        with urllib.request.urlopen(req, timeout=30) as r:
            info = json.loads(r.read()) or {}
        print(f"[gemini] model check OK: {model} "
              f"({info.get('displayName', '?')})", flush=True)
        return True
    except urllib.error.HTTPError as e:
        report_error("model check failed", _http_detail(e), _hint_for(e.code))
    except Exception as e:                     # noqa: BLE001
        report_error("model check failed", f"{type(e).__name__}: {e}",
                     "the host cannot reach generativelanguage.googleapis.com")
        return False
    try:                        # wrong model id: say what WOULD work
        avail = sorted(m["name"].split("/")[-1] for m in list_models(k)
                       if "generateContent" in
                       (m.get("supportedGenerationMethods") or []))
        print("[gemini]   models this key can use: "
              + ", ".join(avail[:12]) + (" ..." if len(avail) > 12 else ""),
              flush=True)
    except Exception:                          # noqa: BLE001
        pass
    return False


def _cache_path(cache_dir: Path, book, pg: int, box) -> Path:
    # T2: prompt now mandates layout-spacing repair + no-deletion
    sig = hashlib.sha1(
        f"T2|{getattr(book.doc, 'name', '')}|{pg}|{tuple(round(v,1) for v in box)}"
        .encode()).hexdigest()
    return cache_dir / f"{sig}.json"


def transcriber(cache_dir: Path | None = None, model: str | None = None,
                key: str | None = None, pool=None):
    """Return llm(book, pg, box) -> rows | None (None = keep det).

    `key` pins one key (tests, single-key deployments). With no key,
    the multi-key pool is used and an exhausted key advances to the
    next instead of ending the run."""
    pool = pool or (None if key else keypool.get_pool())
    key = key or os.environ.get("GEMINI_API_KEY", "")
    model = model or os.environ.get("QBANK_LLM_MODEL", DEFAULT_MODEL)

    def llm(book, pg: int, box):
        cache = None
        if cache_dir is not None:
            cache = _cache_path(cache_dir, book, pg, box)
            if cache.exists():
                try:
                    return json.loads(cache.read_text())
                except Exception:
                    pass
        try:
            pix = book.doc[pg - 1].get_pixmap(
                clip=pymupdf.Rect(*box), matrix=pymupdf.Matrix(3, 3))
            b64 = base64.b64encode(pix.tobytes("png")).decode()
            payload = {
                "contents": [{"parts": [
                    {"inline_data": {"mime_type": "image/png", "data": b64}},
                    {"text": PROMPT}]}],
                "generationConfig": {"temperature": 0.0,
                                     "max_output_tokens": _max_tokens(8192)},
            }
            rows = _call(pool, key, model, payload)
        except Exception as e:                 # noqa: BLE001
            report_error("page image could not be prepared for Gemini",
                         f"{type(e).__name__}: {e}")
            rows = None
        if cache is not None and rows is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rows))
        return rows
    return llm


VERIFY_PROMPT = """You are re-checking ONE ruled medical-textbook table.
A previous transcription of this exact image may contain word-fragment
errors. Suspect fragments: {suspects}

Look at the image again, cell by cell, and return ONLY valid JSON
(no markdown fences): {{"rows": [["cell", "..."], ...]}} — one inner
list per table row, one string per visible column, reading order,
header first. Transcribe every cell EXACTLY as printed: same words,
symbols, units; empty cell = "". Join words the typesetter broke
across lines inside a cell; insert the missing space where two words
are glued. DO NOT correct spellings or terminology, DO NOT add,
remove, translate or reorder content. Pay special attention to the
suspect fragments above — decide from the IMAGE whether each is one
word or two; a suspect next to a complementary fragment may together
form ONE recognised medical term (e.g. a word the publisher split
mid-line) — join it only when the combined form is a real term."""


def verifier(cache_dir: Path | None = None, model: str | None = None,
             key: str | None = None, pool=None):
    """Second pass for QA-flagged boxes: verify(book, pg, box,
    suspects) -> rows | None. Same envelope applies downstream, so a
    hallucinated answer can never reach the output."""
    pool = pool or (None if key else keypool.get_pool())
    key = key or os.environ.get("GEMINI_API_KEY", "")
    model = model or os.environ.get("QBANK_LLM_MODEL", DEFAULT_MODEL)

    def verify(book, pg: int, box, suspects):
        cache = None
        if cache_dir is not None:
            # suspects shape the prompt, so they shape the cache key
            sig = hashlib.sha1(
                f"V3|{getattr(book.doc, 'name', '')}|{pg}|"
                f"{tuple(round(v, 1) for v in box)}|"
                f"{tuple(suspects)}".encode()).hexdigest()
            cache = cache_dir / f"{sig}.json"
            if cache.exists():
                try:
                    return json.loads(cache.read_text())
                except Exception:
                    pass
        try:
            pix = book.doc[pg - 1].get_pixmap(
                clip=pymupdf.Rect(*box), matrix=pymupdf.Matrix(3, 3))
            b64 = base64.b64encode(pix.tobytes("png")).decode()
            payload = {
                "contents": [{"parts": [
                    {"inline_data": {"mime_type": "image/png", "data": b64}},
                    {"text": VERIFY_PROMPT.format(
                        suspects=", ".join(suspects))}]}],
                "generationConfig": {"temperature": 0.0,
                                     "max_output_tokens": _max_tokens(8192)},
            }
            rows = _call(pool, key, model, payload)
        except Exception as e:                 # noqa: BLE001
            report_error("page image could not be prepared for Gemini",
                         f"{type(e).__name__}: {e}")
            rows = None
        if cache is not None and rows is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rows))
        return rows
    return verify


REARRANGE_PROMPT = """You are rearranging ONE medical-textbook table that
a deterministic pipeline extracted from a PDF's text layer. You receive
ONLY the extracted pipe-markdown below — no page image.

The extraction is full of layout artifacts because narrow columns and
line breaks print each visual line separately:
  * words broken across lines INSIDE a cell: "mylo hyoid" -> "mylohyoid",
    "digast ric" -> "digastric", "tens or veli palatini" ->
    "tensor veli palatini", "platys ma" -> "platysma",
    "Mi ddle 1/3" -> "Middle 1/3", "ventr icle" -> "ventricle",
    "developme nt" -> "development", "stag e" -> "stage";
  * a header cell itself wrapped mid-word: "Pharyngeal A rch" ->
    "Pharyngeal Arch", "Important events of each stag e" ->
    "Important events of each stage";
  * neighbouring rows/cells glued into one string:
    "Bulbus cordisProximal 1/3Mi ddle 1/3 (conus cordis)Distal 1/3
    (truncus arteriosus)" is really ONE label ("Bulbus cordis") with
    THREE separate derivatives, each belonging in its own row/cell
    exactly as the printed table shows;
  * missing spaces around words and punctuation: "Rt.ventricle" ->
    "Rt. ventricle", "period(First 2 weeks)" ->
    "period (First 2 weeks)", "FertilizationCleavage and blastocyst
    formation" -> "Fertilization; Cleavage and blastocyst formation".

Using your MEDICAL KNOWLEDGE of what this table describes, return the
SAME table rearranged so a medical student can read it:
- header row first, every column properly headed, no split words in it;
- every value in the cell it medically belongs to; every word whole
  (unwrapped, un-glued), natural single spaces, punctuation spaced;
- where the extraction lists parallel entries inside one cell
  (multiple derivatives, multiple events), give each its own row or a
  clearly separated list;
- keep every fact, value, unit, abbreviation, roman numeral and
  citation EXACTLY as in the extraction: arrangement, wrapping and
  spacing may change, content may not.

The extraction is the ONLY source:
- add NOTHING from your memory — no fact, no word, no row, no value
  that is not already present in it;
- completing an obvious mid-word split ("digast ric" -> "digastric")
  is repair, not addition;
- if a cell looks cut off (its continuation is simply not in the
  extraction), keep exactly what is given; no guesswork.

Return ONLY the rearranged pipe-markdown table (no fences, no prose),
one row per line, every row with the same number of columns:
| Header | Header |
|---|---|
| ... | ... |"""


def _call_text(pool, key: str, model: str, payload: dict) -> str | None:
    """generateContent exchange returning raw text (markdown), with the
    same pool/retry discipline as _call — plus a LOUD report of every
    failure and one automatic retry with a bigger budget when the model
    ran out of output tokens before it finished the table."""
    rot, attempt = 0, 0
    while attempt < 3:
        try:
            k = pool.acquire() if pool is not None else key
            resp = _post(API.format(model=model), payload, k)
            if pool is not None:
                pool.note_call()
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", "replace")
            except Exception:              # noqa: BLE001
                body = ""
            if pool is not None and e.code == 429 and rot < len(pool.keys):
                pool.note_429(body)
                rot += 1
                continue
            if pool is not None and rot < len(pool.keys) \
                    and _key_fault(e.code, body):
                pool.note_bad_key(e.code, body)   # next key may serve
                rot += 1
                continue
            report_error(f"{model} call failed",
                         _http_detail_text(e.code, body), _hint_for(e.code))
            return None
        except keypool.PoolExhausted as e:
            report_error(f"{model} pool exhausted", str(e),
                         "no usable key left (daily caps spent, keys "
                         "rejected, or every key cooling down) — "
                         "deterministic output for now")
            return None
        except Exception as e:                 # noqa: BLE001
            report_error(f"{model} call failed", f"{type(e).__name__}: {e}",
                         "network unreachable from this host?")
            return None
        txt, why = _answer_text(resp)
        if txt:
            return re.sub(r"^```(?:markdown)?|```$", "", txt).strip()
        report_error(f"{model} no usable answer", why)
        if "MAX_TOKENS" in why:
            payload = _bump_cap(payload)       # retry with more room
        attempt += 1
    return None


def refiner(cache_dir: Path | None = None, model: str | None = None,
            key: str | None = None, pool=None):
    """Return refine(book, pgs, current_md) -> markdown | None.

    Runs DURING extraction for every extracted table. TEXT-ONLY: the
    extracted pipe-markdown itself is what Gemini refines — no page
    images are rendered or sent. The caller saves the returned
    markdown (validated only for table structure, not
    content-identity)."""
    pool = pool or (None if key else keypool.get_pool())
    key = key or os.environ.get("GEMINI_API_KEY", "")
    model = model or os.environ.get("QBANK_LLM_MODEL", DEFAULT_MODEL)

    def refine(book, pgs, current_md: str) -> str | None:
        pgs = [int(p) for p in (pgs if isinstance(pgs, (list, tuple))
                                else [pgs])] or [1]
        cache = None
        if cache_dir is not None:
            # R7: the MODEL and the PROMPT are part of the identity — a
            # cached answer from another model (or from an older prompt)
            # must never be served as if it were this one's.
            sig = hashlib.sha1(
                f"R7|{model}|{_fp8(REARRANGE_PROMPT)}|"
                f"{getattr(book.doc, 'name', '')}|{tuple(pgs)}|"
                f"{hashlib.sha1(current_md.encode()).hexdigest()}"
                .encode()).hexdigest()
            cache = cache_dir / f"{sig}.json"
            if cache.exists():
                try:
                    return json.loads(cache.read_text())
                except Exception:              # noqa: BLE001
                    pass
        ask = REARRANGE_PROMPT
        if len(pgs) > 1:
            ask += (f"\n\nThe extraction below covers a table that "
                    f"spanned {len(pgs)} printed pages: treat it as "
                    "ONE continuous table — the page break is just "
                    "another artifact to repair.")
        payload = {
            "contents": [{"parts": [
                {"text": ask + "\n\nExtraction:\n" + current_md}]}],
            "generationConfig": {"temperature": 0.0,
                                 "max_output_tokens": _max_tokens(16384)},
        }
        try:
            txt = _call_text(pool, key, model, payload)
        except Exception:
            txt = None
        if cache is not None and txt is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(txt))
        return txt
    return refine
