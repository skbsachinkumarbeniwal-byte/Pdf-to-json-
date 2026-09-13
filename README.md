# Jdon Extract — deterministic text-layer pipeline (v2)

Extracts question banks from MARROW ED8 PDFs **with a corrected text
layer** into the v1 `final_export.zip` contract (`FORMAT.md`).

v2 replaces the old OCR/Gemini/review-layer pipeline. The text layer of
the corrected books is authoritative, so extraction is deterministic in
its matrix: nothing is ever guessed — a row is written only when the
printed page proves it. Gemini is used inside extraction for tables
only: every extracted table is sent to the model once, which rearranges
it per medical knowledge, and that rearranged markdown is what ships
(provenance kept on the record). The export gate is the human-review
layer: the zip builds the moment the book's REVIEW tables are all
decided; census anomalies and unresolved q_ids ride along in the
receipt as advisories instead of blocking the build.

## Prerequisites

- Python ≥ 3.10
- Dependencies: `PyMuPDF`, `Pillow` (`pip install -r requirements.txt`;
  `pytest` is only needed for the test suite)
- **Corrected text layer required.** v2 reads the PDF's internal text
  layer exclusively. A scanned or image-only PDF yields no baselines —
  the TOC parse raises and the run stops. v2 intentionally fails rather
  than guess; it never falls back to OCR.

## Workspace Structure

```text
.
├── books.json             # Registry: subject code -> pdf path, page_offset
├── pdfs/                  # (gitignored) place corrected book PDFs here
├── qbank/                 # Core pipeline package (run: python -m qbank)
├── scripts/
│   └── verify_extraction.py   # independent ground-truth verifier
├── tests/                 # pytest suite (21 tests, synthetic PDFs only)
├── FORMAT.md              # Output contract specification (frozen)
├── glyph_artifacts.md     # Glyph artifact census + repair audit (BIO)
├── requirements.txt
├── Dockerfile
└── qbank_output/          # Generated (gitignored): split/, assets/,
                           #   data/, subjects/, final_export_<CODE>.zip
```

## What v2 does differently

| v1 (deleted)                  | v2 (this tree)                                    |
| ----------------------------- | ------------------------------------------------- |
| Gemini OCR + vision fallback  | none — text layer only (`provenance: TEXT_LAYER`) |
| Flask dashboard, review queue | pure CLI (`python -m qbank`)                      |
| LLM table reconstruction      | ruled-grid detection + verbatim cell reads        |
| heuristic glyph regex         | font-proven sentinels + ordered, audited rules    |
| `REVIEW_NEEDED` by LLM flag   | `REVIEW_NEEDED` only for unresolvable glyphs      |

The output contract is unchanged: `split/{SUBJ}/{SUBJ}-NNN/` with
`questions.jsonl`, `answers.jsonl`, `solutions.jsonl`,
`unresolved_qids.jsonl`, `orphans.jsonl`, `image_manifest.jsonl`,
`chapter_completeness.json`; `assets/questions/{SUBJ}/*.webp`;
`data/chapters.json`, `data/image_ownership.jsonl`;
`subjects/{SUBJ}/chapters.json`; `final_export_<CODE>.zip` +
`REVIEW_RECEIPT.json`. See `FORMAT.md`.

## Quickstart

```bash
pip install -r requirements.txt

# put the book PDFs in pdfs/ (gitignored) or point QBANK_PDFS_DIR at a dir
python -m qbank run --book BIO              # extract (resumable per chapter)
python -m qbank status                      # progress + export-gate report
python -m qbank export --book ENT           # ENT-only zip, ENT-only gate
```

`books.json` registers books:

```json
{ "BIO": { "path": "pdfs/Biochemistry_ed8_CLEAN_corrected.pdf",
           "page_offset": "auto" } }
```

`page_offset: "auto"` proves the file-page ↔ printed-page offset from
the books' own footer numbers; pass `--page-offset N` to override.
One-off runs without touching the registry:
`python -m qbank run --pdf path/to/book.pdf --subject OBG`.
Other flags: `--chapters 1,3-5` (subset), `--force` (ignore resume
state).

## Dashboard (upload → run → download)

`dashboard.py` is a thin web shell around the same CLI functions —
the pipeline core stays deterministic and untouched:

```bash
python dashboard.py        # binds 0.0.0.0:$PORT (default 8000)
```

- **Upload** a corrected book PDF (auto-registers in `books.json`;
  subject code defaults to the file name prefix)
- **Run / Re-run** in a background thread with a live chapter log;
  one book at a time; the export step runs automatically afterwards
- **Download** — every book gets its OWN `final_export_<CODE>.zip` with its
  OWN gate (only that book's completeness + review queue must be clear), so
  running a new book never reopens or blocks the old book's zip. There is
  no combined all-books export — one book, one gate, one zip

On Railway: mount a Volume at `/out` so the zip survives redeploys
(and optionally `/pdfs` for uploads). Railway injects `$PORT`; the
dashboard binds it and Railway exposes the public URL automatically.

## How extraction works

1. **Contents table** (`qbank/toc.py`) — parsed geometrically from word
   baselines; chapter file ranges come from the TOC + proven offset.
2. **Zones** (`qbank/zones.py`) — each chapter splits at the printed
   `Answer Key` baseline and the first `Solution to Question N:` header.
   The key itself is a ruled table, but no pipe syntax is involved:
   each `15 a` row is paired geometrically from words sharing one
   baseline. Before any of this, the printed page number standing alone
   in the bottom 12% of each page is lifted out of the text stream
   (exact string match against the proven offset — kept in a per-page
   footer audit, never parsed into stems or solutions).
3. **Blocks** (`qbank/parse.py`) — one `Question N:` header anchors one
   record; stem/options/solution text is reflowed from visual-order
   baselines (line-continuation fragments merge back by y/3 bucket +
   x-order).
4. **Tables** (`qbank/tables.py`) — a table exists **iff the book drew
   rules**: long horizontal rules stitched by verticals form a grid
   box (`textlayer._ruled_table_boxes`). Each box is rebuilt
   geometrically, never from flattened text:
   vertical rules give the columns, horizontal rules give the row
   bands; lines are assigned to (band, column) cells by bbox.
   - *In-cell line reconstruction*: a wrapped line is glued to its
     predecessor without a space only when the predecessor fills the
     column's measured fill edge AND that edge reaches the column's
     right rule (the typesetter ran out of room — "medial su"+"rface"
     = "medial surface", "C2,C"+"3" = "C2,C3", "Grad"+"e 1" =
     "Grade 1"); trailing hyphens join keeping the hyphen; every other
     wrap keeps its space ("middle"+"ear" = "middle ear"). Each join
     is layout-proven and counted (`line_joins`).
   - *Lost-space repair*: camel-boundary spaces are re-inserted for
     text-layer corruptions ("antihelixSome" → "antihelix Some"),
     guarded so pH/IgG/mOsm/B12 never split; counted
     (`camel_space_fixes`). Remaining suspects (long space-less
     tokens) are flagged in the table's `validation.warnings`, never
     silently rewritten.
   - *Cross-page merge*: same column geometry on the next page +
     repeated header (deduplicated) or last-box → first-box document
     flow merges into ONE logical table — one `table_id`,
     `source_pages` lists every contributing page.
   - Every logical table ships as pipe-markdown in the row's `tables`
     field AND as one pixel-exact 200 dpi WebP render per contributing
     page (manifest `xref = -1`); both carry the same `table_id`, so
     the structured and visual representations are linked, never
     accidental duplicates. Per-chapter aggregates land in
     `chapter_completeness.json` → `tables`.
   Unruled pseudo-tables and bullet lists stay verbatim prose.
5. **Glyphs** (`qbank/glyphs.py`) — the corrected books still carry two
   broken font mappings (Symbol `°`, AdobePiStd `■`). Sentinelisation
   is **per span**: font identity is read from the individual
   `get_text("dict")` span, so a broken-font glyph inside an otherwise
   clean line is caught and clean spans are never touched. Sentinels
   are then repaired by an ordered, enumerated rule table covering
   exactly the 67 artifacts
   catalogued for Biochemistry ED8. Every application is counted per
   question (`glyph_fix_counts` in `chapter_completeness.json`); real
   ArialMT degree signs are never touched; anything unmatched is
   restored verbatim and the row is flagged `REVIEW_NEEDED` — never
   guessed.
6. **Images** (`qbank/images.py`) — embedded figures are claimed by
   (page, y-center) inside the owning block or option interval;
   orphans go to `orphans.jsonl`. Nothing is dropped silently. Some
   publisher figures are stored as 2-4 interlocking image placements;
   touching placements are clustered per page and shipped as ONE clip
   render of their union bbox (pixel-exact stitch, seams and vector
   overlays included) instead of cut fragments — recorded in
   `chapter_completeness.json` as `merged_placements`.
7. **Optional Gemini table passes** (`qbank/llm.py`) — with
   `GEMINI_API_KEY` set: (a) each ruled box is rendered and sent for
   cell transcription (default model `gemini-3.5-flash-lite`, override
   with `QBANK_LLM_MODEL`); a model cell is accepted ONLY when it is
   character-identical to the deterministic cell after whitespace
   removal; (b) during extraction EVERY extracted table is sent once
   more for rearrangement — the model returns the table rearranged per
   medical knowledge (headers, cell placement, row order) and that
   markdown is what ships, with the original preserved under
   `validation.pre_gemini_markdown` (`QBANK_REFINE=flagged` narrows it
   to flagged tables, `=off` disables). Responses cached under
   `<output>/llm_cache/`. Without a key the pipeline stays zero-LLM.
8. **Gate** (`qbank/export.py`) — the zip builds as soon as every
   chapter is on disk and the book's REVIEW tables are all decided;
   census failures / unresolved q_ids ship as receipt advisories.

**Provenance.** Every text field ships as `TEXT_LAYER` because v2 reads
only the PDF's internal text layer. That makes it immune to OCR
hallucination — and strictly dependent on the publisher's corrected
text layer. If a PDF is purely image-based, v2 fails loudly instead of
inventing content.

## Onboarding a new subject (e.g. OBGYN ED8)

1. **Register the book** in `books.json` — new key, PDF path,
   `page_offset: "auto"` — or probe once with
   `python -m qbank run --pdf <path> --subject OBG`.
2. **Run the extraction.** `run` is chapter-atomic and resumable; a
   chapter whose census fails prints `<<< CENSUS FAILED` but does not
   stop the book.
3. **Verify before anything else**:
   `python scripts/verify_extraction.py <book.pdf> OBG qbank_output`.
   It re-derives an independent ground truth; investigate every failure
   line before trusting the output.
4. **Audit glyphs.** New subjects can introduce broken font mappings
   beyond the two catalogued for BIO. Look for
   `"unknown_glyph"` in each chapter's `glyph_fix_counts`
   (`chapter_completeness.json`) and for `REVIEW_NEEDED` rows — those
   mark sentinels no rule matched. If a pattern recurs, add an ordered
   rule to `qbank/glyphs.py`, add a regression case to
   `tests/test_glyphs.py`, and extend `glyph_artifacts.md` with a
   per-subject artifact table. Unknown glyphs must stay flagged — never
   paper over a flag with a guessed replacement.
5. **Check the structural assumptions.** The zone regexes in
   `qbank/zones.py` expect ED8-style printed headers
   (`Question N:`, `Answer Key`, `Solution to Question N:`) and a
   numeric contents table. A book that prints different furniture will
   show up as census failures — extend the regexes/whitelists, not the
   gate.

## Troubleshooting gate failures

The export gate now blocks only on the human-review layer: the zip is
refused while REVIEW tables are undecided or stale (the review
dashboard clears it). These are NOT blockers any more, but always
investigate them — they mean data you may be missing:

- **census failed** (advisory, in the receipt) — question headers, key
  rows or solution headers are non-contiguous or unequal in a chapter.
  Usually a header variant the zone regexes missed, or a genuinely
  misprinted book.
- **unresolved q_ids** (advisory, in the receipt) — printed
  key/solution anchors exist with no matching question header (see
  `unresolved_qids.jsonl`, which records the exact reason from a fixed
  vocabulary).
- **`REVIEW_NEEDED` rows** — a glyph sentinel no rule could resolve.
  See step 4 of the onboarding guide.

Debug files, per chapter under `split/{SUBJ}/{SUBJ}-NNN/`:

- `chapter_completeness.json` — the census numbers,
  `glyph_fix_counts`, image claim summary; written **last**, so its
  presence means the chapter is fully on disk.
- `unresolved_qids.jsonl` — anchored-but-unmatched questions, with
  `reason` and the anchors that were found.
- `orphans.jsonl` — embedded images no block/option interval claimed.
  Orphans do **not** lock the gate, but review them: each is either
  decorative furniture or a figure whose y-center fell outside every
  block interval (a parsing bug worth fixing).
- `image_manifest.jsonl` / `data/image_ownership.jsonl` — one row per
  shipped image with its claim evidence.

## Verification

`scripts/verify_extraction.py` re-derives an independent visual-order
ground truth from the PDF and checks, for every shipped row: stem,
options, key letter and solution text are exact substrings (up to the
documented glyph repairs), every table's markdown appears verbatim,
and every printed baseline is accounted for by some row (coverage).

```bash
python scripts/verify_extraction.py <book.pdf> BIO qbank_output
# verified 28 chapters, 582 questions, 232 embedded images + 39 table renders
# worst content coverage: 0.9942 — ALL CHECKS PASSED
```

## Tests

```bash
python -m pytest tests/ -q        # 21 tests, no fixtures needed
```

- `test_glyphs.py` — every artifact class of the frozen repair table.
- `test_ruled_tables.py` — grid detection/rejection + markdown order on
  synthetic PDFs.
- `test_mini_book.py` — end-to-end run on a synthetic two-chapter book
  (TOC, offset detection, zones, key pairing, Symbol-font repair,
  image claim, ruled-table markdown + render, split-file contract).

## Environment

`GEMINI_API_KEYS` (comma-separated keys from SEPARATE Google projects)
enables the optional Gemini table-vision pass — quota is per project, so
keys inside one project share one bucket. Without keys the pipeline is
deterministic and receipts say `llm_tables_repaired: 0` honestly.

`OUTPUT_DIR` overrides the output root (default `qbank_output/`);
`QBANK_PDFS_DIR` adds a PDF search directory; `QBANK_BOOKS` overrides
`books.json`.

## Web dashboard (Railway)

**Persistence (one-time setup):** the container FS is ephemeral —
without a volume every deploy wipes runs, ledgers and exports. Add ONE
Railway Volume mounted at `/out` and set env vars
`OUTPUT_DIR=/out` and `QBANK_BOOKS=/out/books.json`. (Optionally keep PDFs
on the same volume: `QBANK_PDFS_DIR=/out/pdfs`.) The app prints a
startup warning when OUTPUT_DIR is still on the ephemeral FS.

`dashboard.py` is the production web shell (Flask): upload/fetch a book
PDF, run extraction, watch the log, download the review-gated per-book `final_export_<CODE>.zip`
(one book, one gate, one zip). Two more surfaces hang off it:

- `/review` — the human review dashboard: every REVIEW-flagged table,
  printed-page crop beside the extracted JSON, in-place edit + approve.
  Decisions persist in append-only ledgers and hold the export gate
  until the queue is resolved.
- `/api/audit` — the post-run content audit report (numeric drift,
  duplicates, thin options, bad answer key); advisory flags only.

Both also run standalone for local work via
`review_dashboard/server.py`.
