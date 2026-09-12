"""
Text-layer extraction primitives.

One Book object caches per-page structured data so every later stage
(zones, parse, images) reads the SAME lines and the SAME geometry:

  PageData.lines : list[Line]  text lines in reading order, artifact
                               glyphs already replaced by sentinels
  PageData.words : list[Word]  word boxes (used for the answer-key and
                               TOC tables, whose columns extract as
                               separate lines in raw text mode)
  PageData.images: list[Img]   embedded-image placements (bbox + xref)

Footer lines (the printed page number alone at the bottom of the page)
are dropped here, once, for everybody.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

from . import glyphs

SENT_BY_FONT_CHAR = {
    # (font-substring, extracted char) -> sentinel
    ("symbol", "\u00b0"): glyphs.SENT_DEGREE,
    ("pi", "\u25a0"): glyphs.SENT_SQUARE,        # AdobePiStd / AdobePi
}

# horizontal gap that splits a baseline into table columns. Justified
# prose never exceeds ~10pt between spans; the ED8 tables use >=25pt.
CLUSTER_GAP = 22.0


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    line_key: tuple          # (block_no, line_no) from pymupdf


@dataclass
class Line:
    page: int                # 1-based PDF file page
    y0: float
    y1: float
    x0: float
    x1: float
    text: str                # sentinel-tagged, NOT stripped
    sizes: tuple = ()
    fonts: tuple = ()
    spans: tuple = ()        # ((text, x0, x1), ...) sentinel-tagged
    clusters: tuple = ()     # ((x0, x1), ...) span runs split by big gaps
                             # (>CLUSTER_GAP pt): 1 for prose, >=2 for a
                             # table baseline

    @property
    def pos(self):
        """Linear reading position used for zone/block interval math."""
        return (self.page, self.y0)

    def cell_texts(self) -> list[str]:
        """One string per x-cluster (table cell text, in visual order)."""
        out = []
        for (cx0, cx1) in self.clusters:
            parts = [t for (t, sx0, sx1) in self.spans
                     if (sx0 + sx1) / 2 >= cx0 - 1 and (sx0 + sx1) / 2 <= cx1 + 1]
            out.append("".join(parts).strip())
        return out


@dataclass
class Img:
    page: int                # 1-based PDF file page
    xref: int
    bbox: tuple              # (x0, y0, x1, y1)

    @property
    def pos(self):
        return (self.page, (self.bbox[1] + self.bbox[3]) / 2.0)


@dataclass
class PageData:
    page: int                # 1-based
    height: float
    width: float
    lines: list = field(default_factory=list)
    words: list = field(default_factory=list)
    images: list = field(default_factory=list)
    footers: list = field(default_factory=list)   # dropped lines (audit)
    table_boxes: list = field(default_factory=list)  # [bbox] of RULED tables


def _ruled_table_boxes(p) -> list:
    """Bounding boxes of RULED tables: clusters of long horizontal
    rules (with or without vertical separators). The ED8 books draw
    real tables with rules and set pseudo-tables/bullets as plain
    text, so 'the book drew a grid' is the deterministic table
    signal — no width/alignment guessing."""
    h, v = [], []
    for it in p.get_drawings():
        for item in it["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                seg = (min(p1.x, p2.x), min(p1.y, p2.y),
                       max(p1.x, p2.x), max(p1.y, p2.y))
            elif item[0] == "re":
                r = item[1]
                seg = (r.x0, r.y0, r.x1, r.y1)
            else:
                continue
            if seg[3] - seg[1] < 2.5 and seg[2] - seg[0] > 60:
                h.append(seg)
            elif seg[2] - seg[0] < 2.5 and seg[3] - seg[1] > 8:
                v.append(seg)
    # cluster segments that TOUCH (bbox intersection with small
    # tolerance). Row rules can be far apart; the full-height verticals
    # stitch them into one grid, so union every box a segment bridges.
    TOL = 3.0
    segs = sorted([(s, "h") for s in h] + [(s, "v") for s in v],
                  key=lambda sk: (sk[0][1], sk[0][0]))
    boxes = []        # [x0, y0, x1, y1, nh, nv]

    def _hits(seg):
        return [bx for bx in boxes
                if (seg[0] <= bx[2] + TOL and seg[2] >= bx[0] - TOL
                    and seg[1] <= bx[3] + TOL and seg[3] >= bx[1] - TOL)]

    for seg, kind in segs:
        hits = _hits(seg)
        if not hits:
            boxes.append([seg[0], seg[1], seg[2], seg[3],
                          1 if kind == "h" else 0,
                          1 if kind == "v" else 0])
            continue
        base = hits[0]
        for other in hits[1:]:
            base[0] = min(base[0], other[0]); base[1] = min(base[1], other[1])
            base[2] = max(base[2], other[2]); base[3] = max(base[3], other[3])
            base[4] += other[4]; base[5] += other[5]
            boxes.remove(other)
        base[0] = min(base[0], seg[0]); base[1] = min(base[1], seg[1])
        base[2] = max(base[2], seg[2]); base[3] = max(base[3], seg[3])
        base[4 if kind == "h" else 5] += 1
    return [(b[0], b[1], b[2], b[3]) for b in boxes
            if b[4] >= 3 or (b[4] >= 2 and b[5] >= 2)]


def _tag_span_text(text: str, font: str) -> str:
    fl = font.lower()
    for (fsub, ch), sent in SENT_BY_FONT_CHAR.items():
        if fsub in fl and ch in text:
            text = text.replace(ch, sent)
    return text


class Book:
    """Lazily-extracted structured view of one PDF."""

    # pages are heavy (words + spans); a 1000+ page book would fill
    # the container's RAM if every touched page stayed cached, so the
    # cache is bounded (sequential access re-parses, which is cheap).
    MAX_CACHED_PAGES = 32

    def __init__(self, path: str, page_offset: int = 0):
        self.doc = pymupdf.open(path)
        self.path = path
        self.page_offset = page_offset     # file_page - printed_page
        self._cache: dict[int, PageData] = {}

    def __len__(self):
        return len(self.doc)

    def drop_cache(self) -> None:
        """Free all cached PageData (call between pipeline phases)."""
        self._cache.clear()

    @property
    def total_pages(self):
        return len(self.doc)

    def page(self, file_page: int) -> PageData:
        """file_page is 1-based."""
        if file_page in self._cache:
            return self._cache[file_page]
        p = self.doc[file_page - 1]
        pd = PageData(page=file_page, height=p.rect.height, width=p.rect.width)

        # --- text lines (with fonts/sizes; artifact glyphs sentinelised)
        raw_lines = []
        for b in p.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            for ln in b["lines"]:
                parts, sizes, fonts, span_info = [], set(), set(), []
                for s in ln["spans"]:
                    txt = _tag_span_text(s["text"], s["font"])
                    parts.append(txt)
                    sizes.add(round(s["size"], 1))
                    fonts.add(s["font"])
                    if txt.strip():
                        span_info.append((txt, s["bbox"][0], s["bbox"][2]))
                text = "".join(parts)
                if text.strip():
                    span_info.sort(key=lambda si: si[1])
                    clusters = []
                    for (_t, sx0, sx1) in span_info:
                        if clusters and sx0 - clusters[-1][1] > CLUSTER_GAP:
                            clusters.append((sx0, sx1))
                        elif clusters:
                            clusters[-1] = (clusters[-1][0],
                                            max(clusters[-1][1], sx1))
                        else:
                            clusters.append((sx0, sx1))
                    raw_lines.append(Line(
                        page=file_page,
                        y0=ln["bbox"][1], y1=ln["bbox"][3],
                        x0=ln["bbox"][0], x1=ln["bbox"][2],
                        text=text,
                        sizes=tuple(sorted(sizes)),
                        fonts=tuple(sorted(fonts)),
                        spans=tuple(span_info),
                        clusters=tuple(clusters)))
        # Reading order: baselines first (bucketed with ~3pt tolerance —
        # line-end fragments in these PDFs sit ~0.3pt above the line
        # start, and an exact-y sort puts them BEFORE their own line),
        # then left-to-right inside a baseline.
        raw_lines.sort(key=lambda l: (round(l.y0 / 3.0), l.x0))

        # --- footer: printed page number alone near the page bottom
        printed = file_page - self.page_offset
        kept = []
        for l in raw_lines:
            is_footer = (
                l.text.strip() == str(printed)
                and l.y1 > pd.height * 0.88
            )
            (pd.footers if is_footer else kept).append(l)
        pd.lines = kept

        # --- words (tight boxes; geometry for the key/TOC tables and
        #     the table-cell reconstruction). Each word inherits the
        #     glyph tagging of the span that contains it, so table
        #     cells stay sentinel-safe.
        font_spans = []
        for b in p.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            for ln in b["lines"]:
                for s in ln["spans"]:
                    font_spans.append((s["bbox"], s["font"]))
        for w in p.get_text("words"):
            x0, y0, x1, y1, txt, bno, lno, _wno = w
            if not txt.strip():
                continue
            # drop footer page numbers here too
            if txt.strip() == str(printed) and y1 > pd.height * 0.88:
                continue
            cx = (x0 + x1) / 2
            for (sx0, sy0, sx1, sy1), font in font_spans:
                if sx0 - 1 <= cx <= sx1 + 1 and sy0 - 2 <= y0 <= sy1 + 2:
                    txt = _tag_span_text(txt, font)
                    break
            pd.words.append(Word(txt, x0, y0, x1, y1, (bno, lno)))

        # --- embedded image placements
        for info in p.get_image_info(xrefs=True):
            pd.images.append(Img(
                page=file_page,
                xref=info.get("xref", 0),
                bbox=tuple(info["bbox"])))

        # --- ruled tables
        pd.table_boxes = _ruled_table_boxes(p)

        self._cache[file_page] = pd
        while len(self._cache) > self.MAX_CACHED_PAGES:
            self._cache.pop(next(iter(self._cache)))
        return pd

    def set_offset(self, offset: int) -> None:
        """Set the proven page offset and drop cached pages (footer
        stripping depends on it)."""
        self.page_offset = offset
        self._cache.clear()

    def chapter_lines(self, first_page: int, last_page: int) -> list[Line]:
        out = []
        for pg in range(first_page, last_page + 1):
            out.extend(self.page(pg).lines)
        return out

    def close(self):
        self.doc.close()


# ---------------------------------------------------------------------------
# reflow: physical lines -> field text
# ---------------------------------------------------------------------------

_NEW_GROUP = re.compile(
    r"^(?:\u2022|\(\d+\)|\d+\.\s|Option\s+[A-D]\s*:)", re.I)


def reflow(lines: list[Line]) -> str:
    """Join physical lines the way the book reads.

    Wrapped sentence lines join with a single space; structural lines
    (bullets, (1)-style sub-items, 'Option A:' commentary) keep their
    own line; a vertical gap clearly bigger than a wrap gap starts a
    NEW LINE (paragraph / bullet) — the book's own layout is the
    authority. Deterministic — no content is added or removed.
    """
    gaps = sorted(b.y0 - a.y1 for a, b in zip(lines, lines[1:])
                  if a.page == b.page and b.y0 - a.y1 > 0)
    if gaps:
        med = gaps[len(gaps) // 2]
        # wrapped (continuation) lines hug the previous line; fresh
        # lines (new paragraph / bullet) sit a clear gap below. When
        # the block is ALL wraps the median itself is tiny, so scale
        # up instead of down.
        thr = med * 0.5 if med >= 3.0 else med * 1.5
    else:
        thr = None
    out, prev = [], None
    for ln in lines:
        t = ln.text.strip()
        if not t:
            continue
        if out:
            if (thr is not None and prev is not None
                    and prev.page == ln.page
                    and ln.y0 - prev.y1 > thr):
                out.append("\n")
            else:
                out.append("\n" if _NEW_GROUP.match(t) else " ")
        out.append(t)
        prev = ln
    return "".join(out)


def word_rows(page_data: PageData, tol: float = 5.0) -> list[list[Word]]:
    """Words clustered into visual rows (one baseline). Used for the
    answer-key and TOC tables, whose columns extract as separate text
    lines at slightly different y offsets (the ED8 TOC prints its
    numbers ~2pt below the title baseline)."""
    ws = sorted(page_data.words, key=lambda w: (w.y0, w.x0))
    rows, cur, cur_y = [], [], None
    for w in ws:
        if cur and w.y0 - cur_y > tol:
            rows.append(sorted(cur, key=lambda x: x.x0))
            cur, cur_y = [], None
        if cur_y is None:
            cur_y = w.y0
        cur.append(w)
    if cur:
        rows.append(sorted(cur, key=lambda x: x.x0))
    return rows
