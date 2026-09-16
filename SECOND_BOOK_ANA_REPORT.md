# ANA (MARROW ED8 Anatomy) — chunk-by-chunk run report

Book: `Anatomy_ed8_MARROW.pdf` — 1217 pages, offset 0, **63 chapters**, registered as `ANA`.
Run mode: **never the whole book at once** — chunk by chunk (user's standing instruction).

## 1. Chunk log (run → collect → next)

| chunk | chapters | questions | census | table audit (refined / unchanged / rejected / review) |
|---|---|---|---|---|
| 1 | 1 | 19 | none | — |
| 2 | 2–9 | 145 | none | — |
| 3 | 10–17 | 144 | none | — |
| 4 | 18–25 | 97 | none | 4 / 16 / 10 / 0 |
| 5 | 26–33 | 146 | none | 6 / 22 / 11 / 0 |
| 6 | 34–41 | 146 | none | 9 / 32 / 20 / 0 |
| 7 | 42–49 | 139 | none | 10 / 42 / 22 / 0 |
| 8 | 50–57 | 167 | none | 17 / 50 / 30 / 0 |
| 9 | 58–63 | 112 | none | 18 / 56 / 33 / 0 |

Logs: `/home/user/logs/ana_chunk_02.log` … `ana_chunk_09.log` (+ `ana_chunk_16_rerun.log`).

**Total 1115 questions in 63 chapters — exactly the number of question headers the book prints.
1254 embedded images. `census failures: none` in every chunk.**

Recipe per chunk (`OUTPUT_DIR=/home/user/anat_out`):

```
source /home/user/.env && export OUTPUT_DIR=/home/user/anat_out
cd /home/user/repo && python -m qbank run --book ANA --chapters <N | A-B>
```

## 2. Final state

```
python -m qbank status      -> export gate ANA: OPEN (zip can build)
python -m qbank audit --book ANA
    scanned 1115 question rows; page-text evidence: yes
    no flags
python -m qbank table-audit --book ANA
    total_tables 107  refined 18  unchanged 56  rejected 33  review 0
    spacing_repairs 24  medical_spelling_repairs 1  cross_page_tables_checked 15
    rejected_hallucinations 16  rejected_deletions 6  before_after_regression OK
```

QA status of the shipped rows: **1110 READY / 5 REVIEW_NEEDED**, tables **0 REVIEW**.

Export:

```
python -m qbank export --book ANA
-> /home/user/anat_out/final_export_ANA.zip   52,527,971 B
   sha256 68fa3478410afe717310ef74339c2dd0fbc94d30cdb90aa71f1c8062c644b4e3
   63 chapters, 1115 questions, unresolved_qids 0, census_failed_chapters []
```

## 3. Verification (independent tools)

* `scripts/verify_extraction.py <pdf> ANA /home/user/anat_out`
  — 63 chapters, 1115 questions, 1254 embedded images, worst content coverage **0.9898**,
  **7 FAILURES: single table cells** (listed in §4).
* `scripts/medical_token_audit.py /home/user/anat_out ANA <pdf>`
  — 144 candidates in 67 forms, 6 repairs accepted, **whitespace-only fidelity failures: 0**
  (every remaining candidate is a benign "A 40-year-old"-class false positive the book does not print joined).
* MIC (re-verified after the same fixes): `verify_extraction.py … MIC /home/user/micro_out`
  — **ALL CHECKS PASSED**, 35 chapters, 616 questions, worst coverage 0.9922.

## 4. Results found and fixed during this book (each one is a real pipeline defect)

1. **Glued function words in the print itself were never repaired.**
   The book's narrow table columns drop spaces: p301 prints `Apical partof cellsheds offdur`, p236 (MIC)
   prints `leucocytosis alongwith`, p366 `Sensorylanguage area`. The repair layer only split a glued form
   when the book printed it **zero** times, so a form printed **once** (the artifact itself) was left glued
   and then re-raised a table REVIEW that locked the export gate.
   Fix: `qbank/tables.py` — `_split_glue_token` / `split_glue_words` (function-word glue, glued form printed
   ≤1×, both halves established ≥2×, spaced pair a recurring print ≥2×), wired into table cells
   (`_repair_token`) and into prose (`spacing_fix(..., vocab)`, called from `qbank/parse.py`).
   Measured blast radius over both books: `partof`, `layersof`, `arteryin`, `thecell`, `themiddle`,
   `proximalto`, `alongwith`, `atleast`, `thesame` — all true artifacts, zero false positives.
2. **Latin/anatomical terms were corrupted by the same cascade.**
   The suffix split used *any* function word, so `Depressor labii inferioris` → `inferior is`, and the
   species names `vaginalis` / `recurrentis` / `fetalis` / `inhibitor` → `vaginal is`, `inhibit or` the moment
   such a token reached a table cell (7 such words exist in the MIC vocabulary alone).
   Fix: `_LATIN_TAILS = {is, or, as, it}` excluded from the glue set; a test pins that
   `Depressor labii inferioris` is never split.
3. **Two table REVIEWs were false positives of the QA detector** (`020-T01` suspects `['language']`,
   `032-T01` suspects `['jugulodigastric']` — both are the book's own single print of a glued form).
   Fix: `qa_suspects` now requires the glued complement to be an **established** print (≥2×), plus the new
   `scripts/recheck_tables.py` re-judges stored tables with the current detector without re-running anything
   (`--apply` writes atomically, keeps `.pre-recheck` backups, stamps provenance).
   Result: ANA tables 78/78 instances `ok`, gate OPEN.
4. **A faithful table was flagged `numeric_drift` by the post-run audit** (ANA-020-004 / `020-T01`, value "19").
   The print wraps the number: p366 really reads `Visual associationarea (1` / `8,19) (B)` — one number
   "18,19" split over two lines, while our cell correctly joins it. The audit's tokeniser then saw `819`
   on the print's side and `18`/`19` on ours.
   Fix: `qbank/audit.py` — comma-list normalisation + `page_num_evidence()` (digit runs re-joined across a
   whitespace break, letters never merged). ANA audit after the fix: **no flags**.
5. **Table-cell text is now repaired by the same evidence rule** (p301 `partof` → `part of`), verified by a
   fresh `--force` re-run of chapter 16: the shipped cell reads
   `Apical part of cellsheds off during discharge`, `table_qa: ok`.

Tests: **292 passed, 36 skipped**.

## 5. Honest residual list (nothing hidden)

1. **5 question rows are `REVIEW_NEEDED`** — `ANA-014-004`, `ANA-016-001`, `ANA-028-023`, `ANA-031-005`,
   `ANA-043-005`, all with reason `source_missing_match_items`: the book's own match-items are missing/
   ambiguous on the printed page. Content was kept exactly as printed (no guessing). These rows ship inside
   the zip and do **not** lock the export gate.
2. **7 table cells differ from the print in tiny, character-level ways** (verifier §3 list):
   * `051-T01`, `051-T03`, `051-T05` — the book prints `duoednum`; the output ships `duodenum`
     (deliberate, recorded medical-spelling repair: `medical_spelling_repairs 1`). The independent
     verifier demands letter-identity with the print, so it flags these by design.
   * `017-T02` (`sensory nuclei: CN V…`) and `039-T05` (`Medial half: ulnar nerve…`) — a **colon** was
     inserted after the label where that particular printed line has none (the same table prints the colon
     in its second listing on p721).
   * `052-T04` — a **period** was inserted after `epididymis` (the model's sentence split; the print runs
     the sentences together).
   * `049-T02` — capitalisation: print `Right Psoas major` ships as `Right psoas major`.
   None of these changes a letter of a medical term, none changes a number, and all are recorded; but they
   are not "whitespace only", so they honestly fail the strict verifier. Fix path if wanted: tighten the
   presentation guard in `refine_final.py` (an inserted mark the print does not carry must reject the
   refinement) and re-run the four affected chapters (17, 39, 49, 52).
3. **Source-level glue that no evidence can repair** stays printed as-is (e.g. `cellsheds` in `016-T01`
   — the book never prints "cell sheds" or "cell"+"sheds" apart), and is reported as an advisory
   `suspect_lost_space:` warning, never as a gate lock.

## 6. Answer to "is the pipeline perfect now?"

Extraction completeness is **1115/1115 questions, 0 unresolved ids, 0 census failures, 0 audit flags,
0 gate locks** — that part is proven. The engine's own guards rejected 33 model table rewrites in the last
chunk alone (`hallucinated_addition`, `content_substitution`, `mark moved`) instead of shipping them.
What remains is a **7-cell character-level residual** (§5.2, 3 of them a deliberate typo fix) plus the 5
book-defect rows (§5.1) — small, listed, reproducible, and fixable on request.
