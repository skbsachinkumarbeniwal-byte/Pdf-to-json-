"""Free the volume after a book has shipped: delete ONE subject's
extracted data (split/, subjects/, assets/, crops/, its review-ledger
rows, its resume state) while keeping its final_export_<CODE>.zip so
the shipped product stays downloadable. Deterministic, per-subject —
a new book's run can call this for every already-shipped old book so
the container's volume (and the review dashboard) only ever hold the
book being worked on."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import review
from . import state as state_mod


def _ledger_subject(row: dict) -> str:
    """Subject of a ledger row: explicit field or BOOK|q_id|tid key."""
    k = row.get("book") or row.get("subject") or ""
    if not k and "|" in str(row.get("key", "")):
        k = str(row["key"]).split("|", 1)[0]
    return str(k).upper()


def _filter_ledger(p: Path, subject: str) -> int:
    """Drop rows belonging to subject; returns rows removed."""
    if not p.exists():
        return 0
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    keep = [r for r in rows if _ledger_subject(r) != subject]
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in keep))
    return len(rows) - len(keep)


def purge_subject(output_root, subject: str, keep_zip: bool = True) -> dict:
    """Delete everything the volume holds for subject except its zip.
    Never touches the source PDF (re-runs stay possible)."""
    out = Path(output_root)
    subject = subject.upper()
    removed = {}

    split = out / "split" / subject
    if split.is_dir():
        shutil.rmtree(split)
        removed["split"] = str(split)

    subj = out / "subjects" / subject
    if subj.is_dir():
        shutil.rmtree(subj)
        removed["subjects"] = str(subj)

    assets = out / "assets" / "questions" / subject
    if assets.is_dir():
        shutil.rmtree(assets)
        removed["assets"] = str(assets)

    crops = out / "crops"
    if crops.is_dir():
        hit = sorted(crops.glob(f"{subject.lower()}_*"))
        for f in hit:
            f.unlink()
        if hit:
            removed["crops"] = len(hit)

    removed["decisions"] = _filter_ledger(out / review.DECISIONS, subject)
    removed["edits"] = _filter_ledger(out / review.EDIT_LEDGER, subject)

    st = state_mod.load_state()
    if subject in st.get("pdf_progress", {}):
        del st["pdf_progress"][subject]
        state_mod.save_state(st)
        removed["resume_state"] = subject

    if not keep_zip:
        zp = out / f"final_export_{subject}.zip"
        if zp.exists():
            zp.unlink()
            removed["zip"] = str(zp)

    return removed
