#!/usr/bin/env python3
"""End-to-end Gemini check — run this BEFORE blaming the pipeline.

Answers, in order, the four things that can silently kill the table
pass, printing the raw evidence for each:

  1. KEY      which keys the process can see (fingerprints only, never
              the key material) and whether the key is accepted;
  2. MODEL    whether the configured model id exists FOR THIS KEY,
              through the same header auth the real calls use — and
              which ids would work instead;
  3. CALL     a REAL generateContent call with the real
              REARRANGE_PROMPT on a small broken table: shows the raw
              answer, the finishReason, and whether the pipeline's
              validator would accept it;
  4. VERDICT  what will actually happen during extraction.

Usage (inside the Railway shell / container, where the env vars live):

    python scripts/check_gemini.py
    python scripts/check_gemini.py --model gemini-2.5-flash
    python scripts/check_gemini.py --offline      # 1 + 2 only

Exit code 0 = every step passed, 1 = something is wrong (and printed).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qbank import keypool                      # noqa: E402
from qbank import llm                          # noqa: E402
from qbank import refine as refine_mod         # noqa: E402

# deliberately broken the way the real books are broken: a glued word
# and a table whose rows need rearranging
SAMPLE = ("| Pharyngeal | Arch |\n|---|---|\n"
          "| 1 | Maxillary artery;Tri geminal |\n"
          "| 2 | Stapedial artery |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QBANK_LLM_MODEL",
                                                      llm.DEFAULT_MODEL))
    ap.add_argument("--offline", action="store_true",
                    help="keys + model check only, no generateContent call")
    a = ap.parse_args()
    ok = True

    print("=" * 68)
    print("1) KEYS")
    keys = keypool.discover_keys()
    if not keys:
        print("   ✗ no key found — set GEMINI_API_KEY (or GEMINI_API_KEYS / "
              "GEMINI_API_KEY_1..20)")
        print("   ⚠ QBANK_LLM_TABLES=0 also disables the pass even with keys")
        return 1
    print(f"   ✓ {len(keys)} key(s): "
          + ", ".join(f"key{i + 1}#{keypool._fp(k)}"
                      for i, k in enumerate(keys)))
    if os.environ.get("QBANK_LLM_TABLES", "1") == "0":
        print("   ✗ QBANK_LLM_TABLES=0 — the pass is switched OFF by env")
        ok = False

    print("=" * 68)
    print(f"2) MODEL  {a.model}")
    good = llm.check_model(a.model, key=keys[0])
    ok = ok and good
    if not good:
        print("   ✗ this key cannot use this model id — see the list above")

    if a.offline:
        print("=" * 68)
        print("VERDICT:", "ready for extraction" if ok else "FIX THE ABOVE FIRST")
        return 0 if ok else 1

    print("=" * 68)
    print("3) REAL CALL  (REARRANGE_PROMPT on a sample table)")
    payload = {
        "contents": [{"parts": [{"text": llm.REARRANGE_PROMPT
                                 + "\n\nExtraction:\n" + SAMPLE}]}],
        "generationConfig": {"temperature": 0.0,
                             "max_output_tokens": llm._max_tokens(16384)},
    }
    txt = llm._call_text(None, keys[0], a.model, payload)
    if txt is None:
        print("   ✗ the call returned NOTHING (reason printed above by "
              "[gemini] lines) — tables would ship raw")
        ok = False
    else:
        print("   ✓ answer received, "
              f"{len(txt)} chars. First 3 lines:")
        for line in txt.strip().splitlines()[:3]:
            print("     " + line[:120])
        acc = refine_mod.valid_rearrangement(txt)
        print(f"   {'✓' if acc else '✗'} validator: "
              + ("accepted as an even pipe table"
                 if acc else "REJECTED — not an even pipe-markdown table "
                             "(the original is kept)"))
        ok = ok and acc

    print("=" * 68)
    print("VERDICT:", "ready for extraction" if ok else "FIX THE ABOVE FIRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
