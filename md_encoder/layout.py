"""Stage 2 – layout analysis.

Turns the OCR lines of one page into an ordered list of semantic blocks:
headings, paragraphs, bullet items, endnotes, margin notes ("Marginalien"),
tinted info boxes, figures and tables (cropped as images, tables additionally
rebuilt as Markdown tables).

Everything works in normalised page coordinates (0..1, origin top-left).
The thresholds were tuned on a classic textbook layout: one body column, margin
notes on the outer edge, running header, page number + publisher line at the
bottom, captions *below* figures/tables. Books without margin notes or boxes simply
produce headings, paragraphs, lists, figures and tables. The word lists below cover
German and English; extend them for other languages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median

import numpy as np

HEADER_Y = 0.065       # everything above is running header
FOOTER_Y = 0.935       # everything below is page number / publisher line
SPLIT_GAP = 0.03       # horizontal gap between words that splits a line in two
DEFAULT_PITCH = 0.0163  # typical baseline-to-baseline distance

CAPTION_WORDS = r"Abbildung|Abb\.|Tabelle|Tab\.|Figure|Fig\.|Table|Exhibit"
# "Abbildung 3.1: …", "Figure 1.2. …" – the number must be complete, so "Abbildung 3.1 zeigt" is no caption
CAPTION_RE = re.compile(rf"^({CAPTION_WORDS})\s*(\d+(?:\.\d+)*)(?!\.?\d)\s*[:.]\s*(.*)$")
CHAPTER_RE = re.compile(r"^(?:Chapter|Kapitel|Part|Teil|Lesson|Lektion)\s+(\d{1,2})\b\s*[:.\-–]?\s*(.*)$", re.I)
BIG_NUM_TITLE_RE = re.compile(r"^(\d{1,2})\.?\s+([A-ZÄÖÜ].{2,80})$")
NUM_HEAD_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,4})\.?\s+([A-ZÄÖÜ0-9„\"].*)$")
BULLET_RE = re.compile(r"^[•■▪●◦○□►▸]\s*")
ENDNOTE_RE = re.compile(r"^(\d{1,3})\s+(\S.*)$")
UPPER_TITLE_RE = re.compile(r"^[A-ZÄÖÜ][A-ZÄÖÜ \-&]{5,}$")
# "Markt- und Umwelt": a hyphen before these words is a suspended compound, not a line-break hyphen
CONJ = {"und", "oder", "bzw", "sowie", "als", "bis", "noch", "wie", "u."}
# lines containing this are licence stamps / watermarks, not book text (see --drop)
WATERMARK_RE = re.compile(r"Lizenziert für|Licensed to|Licensed for")

TOC_TITLES = {"inhaltsverzeichnis", "inhalt", "contents", "table of contents"}
ENDNOTE_TITLES = {"literaturhinweise und anmerkungen", "anmerkungen", "endnoten", "notes", "endnotes", "notes and references"}
BACK_MATTER = {
    "literaturverzeichnis": "Literaturverzeichnis",
    "quellenverzeichnis": "Quellenverzeichnis",
    "stichwortverzeichnis": "Stichwortverzeichnis",
    "sachverzeichnis": "Sachverzeichnis",
    "register": "Register",
    "glossar": "Glossar",
    "bibliography": "Bibliography",
    "references": "References",
    "glossary": "Glossary",
    "index": "Index",
    "subject index": "Subject Index",
}
INDEX_TITLES = {"Stichwortverzeichnis", "Sachverzeichnis", "Register", "Index", "Subject Index"}
INTRO_TITLES = {"vorwort", "einleitung", "zusammenfassung", "preface", "foreword", "introduction", "summary"}


# --------------------------------------------------------------------------- data


@dataclass
class Seg:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    words: list = field(default_factory=list)  # [(text, x0, x1)]
    tint: bool = False
    wide: bool = False  # part of a figure/table that spans into the margin

    @property
    def h(self):
        return self.y1 - self.y0

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def yc(self):
        return (self.y0 + self.y1) / 2

    @property
    def xc(self):
        return (self.x0 + self.x1) / 2


@dataclass
class Block:
    kind: str  # heading | para | bullet | endnote | margin | figure | table | boxtitle
    lines: list = field(default_factory=list)
    level: int = 0
    y0: float = 0.0
    y1: float = 0.0
    x0: float = 0.0
    tint: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return join_lines(self.lines)


@dataclass
class PageResult:
    index: int
    kind: str = "content"  # content | chapter_start | toc | blank
    header: str = ""
    chapter_num: int | None = None
    new_chapter: bool = False  # a chapter heading on a normal content page starts a new chapter
    chapter_title: str = ""
    section_title: str = ""  # big unnumbered heading on the page (e.g. back matter)
    blocks: list = field(default_factory=list)
    crops: list = field(default_factory=list)  # (block, (x0,y0,x1,y1))
    body: tuple = (0.1, 0.9)
    low_conf: int = 0


# --------------------------------------------------------------------------- text helpers


def join_lines(lines: list[str]) -> str:
    lines = [l.strip() for l in lines if l and l.strip()]
    if not lines:
        return ""
    out = lines[0]
    for nxt in lines[1:]:
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha():
            first = nxt.split(" ", 1)[0].rstrip(".,;:")
            if nxt[:1].islower() and first not in CONJ:
                out = out[:-1] + nxt          # Dis- ziplin -> Disziplin
            elif nxt[:1].islower():
                out = out + " " + nxt         # Markt- und -> Markt- und
            else:
                out = out + nxt               # Strategy-as-Practice- Forschung
        else:
            out = out + " " + nxt
    return re.sub(r"\s{2,}", " ", out)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


# --------------------------------------------------------------------------- segment building


def _seg_from_words(words, y0, y1) -> Seg:
    return Seg(
        " ".join(w[0] for w in words),
        words[0][1],
        y0,
        words[-1][2],
        y1,
        list(words),
    )


def split_line(line: dict) -> list[Seg]:
    x0, y0, x1, y1 = line["box"]
    ws = line.get("words") or []
    if not ws or any(w["box"] is None for w in ws):
        return [Seg(line["text"], x0, y0, x1, y1, [(line["text"], x0, x1)])]
    words = [(w["t"], w["box"][0], w["box"][2]) for w in ws]
    groups = [[words[0]]]
    for a, b in zip(words, words[1:]):
        if b[1] - a[2] > SPLIT_GAP:
            groups.append([b])
        else:
            groups[-1].append(b)
    return [_seg_from_words(g, y0, y1) for g in groups]


def split_at_column(seg: Seg, left: float, right: float) -> list[Seg]:
    """Split a segment that straddles the body/margin boundary."""
    if len(seg.words) < 2:
        return [seg]
    if seg.x0 < left - 0.01 and seg.x1 > left + 0.03:
        idx = next((i for i, w in enumerate(seg.words) if w[1] >= left - 0.004), None)
        if idx:
            return [_seg_from_words(seg.words[:idx], seg.y0, seg.y1), _seg_from_words(seg.words[idx:], seg.y0, seg.y1)]
    if seg.x0 < right - 0.03 and seg.x1 > right + 0.012:
        idx = next((i for i, w in enumerate(seg.words) if w[1] > right + 0.002), None)
        if idx:
            return [_seg_from_words(seg.words[:idx], seg.y0, seg.y1), _seg_from_words(seg.words[idx:], seg.y0, seg.y1)]
    return [seg]


def estimate_body(segs: list[Seg]) -> tuple[float, float] | None:
    long = [s for s in segs if s.w > 0.42]
    if len(long) < 4:
        return None
    return median(s.x0 for s in long), median(s.x1 for s in long)


# --------------------------------------------------------------------------- pixel helpers


def _px(img, x0, y0, x1, y1):
    H, W = img.shape[:2]
    return img[max(0, int(y0 * H)): max(1, int(np.ceil(y1 * H))), max(0, int(x0 * W)): max(1, int(np.ceil(x1 * W)))]


def background(img, s: Seg) -> np.ndarray:
    r = _px(img, s.x0, s.y0, s.x1, s.y1).reshape(-1, 3).astype(int)
    if len(r) == 0:
        return np.array([255, 255, 255])
    lum = r.mean(1)
    return np.median(r[lum >= np.percentile(lum, 75)], 0)


def is_tinted(rgb) -> bool:
    return (rgb.max() - rgb.min()) > 10 and rgb.mean() > 170


def is_red_page(img) -> bool:
    hits = 0
    for x, y in ((0.06, 0.5), (0.94, 0.5), (0.5, 0.015), (0.06, 0.9)):
        c = _px(img, x - 0.01, y - 0.005, x + 0.01, y + 0.005).reshape(-1, 3).mean(0)
        if c[0] > 150 and c[1] < 110 and c[2] < 110:
            hits += 1
    return hits >= 3


def ink_bbox(img, x0, y0, x1, y1, masks=()):
    sub = _px(img, x0, y0, x1, y1).astype(int)
    if sub.size == 0:
        return None
    # ink = anything that differs from the background (white page or tinted box)
    bg = np.median(sub.reshape(-1, 3), axis=0)
    ink = np.abs(sub - bg).max(axis=2) > 14
    H, W = img.shape[:2]
    oy, ox = int(y0 * H), int(x0 * W)
    for m in masks:  # blank out margin notes etc.
        ink[max(0, int(m.y0 * H) - oy - 2): int(m.y1 * H) - oy + 2, max(0, int(m.x0 * W) - ox - 2): int(m.x1 * W) - ox + 2] = False
    rows = np.where(ink.sum(1) > 1)[0]
    cols = np.where(ink.sum(0) > 1)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return (x0 + cols[0] / W, y0 + rows[0] / H, x0 + (cols[-1] + 1) / W, y0 + (rows[-1] + 1) / H)


# --------------------------------------------------------------------------- figures & tables


def _is_prose(s: Seg, left, right) -> bool:
    return (
        len(s.text) >= 45
        and abs(s.x1 - right) < 0.015
        and left - 0.012 <= s.x0 <= left + 0.05
    )


def find_figures(body: list[Seg], captions: list[Seg], heads: set, left, right, pitch):
    """For every caption, walk upwards to find where the figure/table starts."""
    regions = []
    cap_ids = {id(c) for c in captions}
    for cap in sorted(captions, key=lambda s: s.y0):
        above = sorted((s for s in body if s.y1 <= cap.y0 + 0.004 and s is not cap), key=lambda s: -s.y1)
        top = HEADER_Y
        for s in above:
            if id(s) in cap_ids or id(s) in heads:
                top = s.y1
                break
            boxed = s.tint and s.w > 0.6 * (right - left) and len(s.text) >= 45  # running text of an info box
            if boxed or _is_prose(s, left, right):
                # the paragraph may end with a short line just above the figure
                cont = s
                for t in sorted((t for t in body if t.y0 > s.y0 + 0.002 and t.y1 <= cap.y0 + 0.004), key=lambda t: t.y0):
                    if s.x0 - 0.012 <= t.x0 <= s.x0 + 0.05 and t.y0 - cont.y1 < 0.007 and (t.yc - cont.yc) < 1.3 * pitch:
                        cont = t
                    else:
                        break
                top = cont.y1
                break
        regions.append((cap, top + 0.002))
    return regions


def _longest_run(row: np.ndarray) -> int:
    if not row.any():
        return 0
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return int((ends - starts).max())


def _cell_text(segs: list[Seg]) -> str:
    """Order the pieces of one table cell line by line, left to right."""
    lines: list[list[Seg]] = []
    for s in sorted(segs, key=lambda s: s.yc):
        if lines and abs(s.yc - lines[-1][-1].yc) < 0.5 * min(s.h, lines[-1][-1].h):
            lines[-1].append(s)
        else:
            lines.append([s])
    return join_lines([" ".join(t.text for t in sorted(l, key=lambda t: t.x0)) for l in lines])


def table_markdown(img, segs: list[Seg], box) -> str | None:
    x0, y0, x1, y1 = box
    sub = _px(img, x0, y0, x1, y1).astype(int)
    if sub.size == 0 or not segs:
        return None
    H = img.shape[0]
    red = ((sub[:, :, 0] > 170) & (sub[:, :, 1] < 120) & (sub[:, :, 2] < 120)).mean(1) > 0.5
    ink = sub.min(axis=2) < 235
    # a rule is one *continuous* run of ink across most of the table (text rows have gaps)
    rule = np.array([_longest_run(row) for row in ink]) > 0.7 * ink.shape[1]
    red_rows = np.where(red)[0]
    if len(red_rows) == 0:
        return None
    # header band = first contiguous run of red rows
    hb_end = red_rows[0]
    while hb_end + 1 < len(red) and red[hb_end + 1]:
        hb_end += 1
    hy0, hy1 = y0 + red_rows[0] / H, y0 + (hb_end + 1) / H
    head = sorted((s for s in segs if hy0 - 0.003 <= s.yc <= hy1 + 0.003), key=lambda s: s.x0)
    if len(head) < 2:
        return None
    cols = []
    for s in head:
        if cols and s.x0 - cols[-1] < 0.02:
            continue
        cols.append(s.x0)
    if len(cols) < 2:
        return None
    seps = [y0 + i / H for i in np.where(rule)[0] if y0 + i / H > hy1 + 0.003]
    bands = []
    prev = hy1
    for y in seps + [y1 + 0.01]:
        if y - prev > 0.006:
            bands.append((prev, y))
        prev = y

    def col_of(s):
        idx = 0
        for i, c in enumerate(cols):
            if s.x0 >= c - 0.012:
                idx = i
        return idx

    body = [s for s in segs if s.yc > hy1 + 0.002]
    rows = []
    if len(bands) >= 2:
        for a, b in bands:
            cells = [[] for _ in cols]
            for s in (s for s in body if a <= s.yc < b):
                cells[col_of(s)].append(s)
            if any(cells):
                rows.append([_cell_text(c) for c in cells])
    else:  # no rules: a new row starts whenever the first column gets text
        for s in sorted(body, key=lambda s: (s.y0, s.x0)):
            c = col_of(s)
            if not rows or (c == 0 and rows[-1][0]):
                rows.append([[] for _ in cols])
            rows[-1][c].append(s)
        rows = [[_cell_text(c) for c in r] for r in rows]
    if not rows:
        return None

    def cell(t):
        return clean_text(t).replace("|", "\\|")

    header_cells = [[] for _ in cols]
    for s in head:
        header_cells[col_of(s)].append(s)
    out = ["| " + " | ".join(cell(_cell_text(c)) for c in header_cells) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- text cleanup

_SUP = "®©™°º^¹²³⁴⁵⁶⁷⁸⁹⁰"
REF_WORDS = r"(Abschnitt|Abbildung|Abb\.|Tabelle|Tab\.|Kapitel|Section|Figure|Fig\.|Table|Chapter)"


def clean_text(t: str, md: bool = True) -> str:
    t = t.replace("­", "")
    # cross-reference arrows (▸ Abschnitt 2.4) are OCR'd as bullets / odd glyphs
    t = re.sub(rf"(^|[\s(])[•►▸▶■●»]\s*(?={REF_WORDS}\b)", r"\1→ ", t)
    # footnote markers that OCR read as symbols or digits: "Markt.® Die", "gesagt.®3", "Ziel:\"°", "nutzen.^ Dem"
    t = re.sub(rf"(?<=[\w.,;:!?\"“”'’)])[{_SUP}]+\d{{0,3}}", "", t)
    t = re.sub(r"(?<=[a-zäöüß)][.,;])['’?*](?=\s+[A-ZÄÖÜ„(]|$)", "", t)
    t = re.sub(r"(?<=[a-zäöüß)][.,;:!?])([\"“”]?)\d{1,3}(?=\s|$)", r"\1", t)
    t = re.sub(r"(?<=[a-zäöüß]{4})\d{1,3}(?=[\s,.;:]|$)", "", t)
    # German quotes: „…" -> „…“
    t = re.sub(r"„([^„\"“]{0,400}?)\"", r"„\1“", t)
    # markdown-sensitive characters
    if md:
        t = t.replace("*", r"\*").replace("_", r"\_").replace("<", "&lt;")
    return re.sub(r"\s{2,}", " ", t).strip()


# --------------------------------------------------------------------------- page analysis


def _chapter_heading(body: list, med_h: float, current: int | None):
    """(number, title, segments) of a chapter heading in the upper part of the page, or None."""
    top = sorted((s for s in body if s.y0 < 0.45), key=lambda s: s.y0)
    for k, s in enumerate(top):
        txt = s.text.strip()
        m = CHAPTER_RE.match(txt)
        if m and s.h > 1.25 * med_h:
            num, title, used = int(m.group(1)), m.group(2).strip(), [s]
            if not title:  # title on the following big line(s)
                for t in top[k + 1:]:
                    if t.y0 - used[-1].y1 > 0.06 or t.h < 1.25 * med_h:
                        break
                    used.append(t)
                title = join_lines([t.text for t in used[1:]])
            return (num, title or f"{num}", used)
        m = BIG_NUM_TITLE_RE.match(txt)
        if m and s.h > 1.6 * med_h and int(m.group(1)) == (current or 0) + 1:
            return int(m.group(1)), m.group(2).strip(), [s]
    return None


def choose_body(segs: list[Seg], fallback: tuple) -> tuple[float, float]:
    """Body column edges. The book-wide value wins whenever lines on this page confirm it;
    boxes are inset, so a page full of boxes would otherwise shrink the column."""
    gl, gr = fallback
    long = [s for s in segs if s.w > 0.42]
    est = estimate_body(segs)
    if not est:
        return fallback
    # any segment confirms an edge (endnote numbers or short lines start at the body edge,
    # margin notes never do)
    left = gl if any(abs(s.x0 - gl) < 0.012 for s in segs) else est[0]
    right = gr if any(abs(s.x1 - gr) < 0.012 for s in segs) else est[1]
    return left, right


def _page_segs(raw: list[dict]) -> tuple[list[Seg], list[dict]]:
    header, segs = [], []
    for l in raw:
        x0, y0, x1, y1 = l["box"]
        txt = l["text"].strip()
        if (y1 - y0) > 2.5 * (x1 - x0) and (y1 - y0) > 0.05:
            continue  # vertical licence watermark
        if WATERMARK_RE.search(txt) or x1 < 0.036 or (len(txt) <= 1 and not txt.isalnum()):
            continue  # watermark / icons ("?", "!")
        if y1 < HEADER_Y:
            header.append(l)
            continue
        if y0 > FOOTER_Y:
            continue
        segs.extend(split_line(l))
    return segs, header


def _merge_wide(segs: list[Seg], full_segs: list[Seg], left, right) -> tuple[list[Seg], dict]:
    """Column-wise OCR cuts figures/tables that reach into the margin. For those
    (recognised by a caption outside the body column) use the full-page OCR instead."""
    caps = [s for s in full_segs if CAPTION_RE.match(s.text.strip())]
    if not caps:
        return segs, {}
    split_full = [p for s in full_segs for p in split_at_column(s, left, right)]
    body_full = [s for s in split_full if not (s.x1 < left - 0.004 or s.x0 > right + 0.004)]
    body_full += [c for c in caps if c not in body_full]
    tops, ranges = {}, []
    for cap, top in find_figures(body_full, caps, set(), left, right, DEFAULT_PITCH):
        outside = cap.x0 < left - 0.02 or cap.x1 > right + 0.02
        straddle = sum(
            1 for s in full_segs
            if top <= s.yc <= cap.y0 and ((s.x0 < right - 0.015 and s.x1 > right + 0.015) or (s.x0 < left - 0.015 and s.x1 > left + 0.015))
        )
        if outside or straddle >= 2:
            tops[id(cap)] = top
            ranges.append((top - 0.003, cap.y1 + 0.003))
    if not ranges:
        return segs, {}

    def in_range(s):
        return any(a <= s.yc <= b for a, b in ranges)

    out = [s for s in segs if not in_range(s)]
    for s in full_segs:
        if in_range(s):
            s.wide = True
            out.append(s)
    return out, tops


def index_blocks(segs: list[Seg]) -> list[Block]:
    """Multi-column register pages: read column by column, one entry per line."""
    xs = sorted(s.x0 for s in segs)
    starts = [xs[0]] + [b for a, b in zip(xs, xs[1:]) if b - a > 0.12]
    def col(s):
        return max(i for i, c in enumerate(starts) if s.x0 >= c - 0.02)
    blocks: list[Block] = []
    order = 0
    for ci in range(len(starts)):
        cur = None
        for s in sorted((s for s in segs if col(s) == ci), key=lambda s: s.y0):
            t = s.text.strip()
            y = order + s.y0
            if norm(t) in BACK_MATTER:
                continue
            if re.fullmatch(r"[A-ZÄÖÜ]|Symbole|Zahlen", t):
                blocks.append(Block("heading", [t], level=3, y0=y, y1=y, extra={"letter": True}))
                cur = None
                continue
            cont = cur is not None and (
                s.x0 > starts[ci] + 0.012 or t[:1].islower() or cur.lines[-1].endswith("-") or re.match(r"^[\d,\s–-]+$", t)
            )
            if cont:
                cur.lines.append(t)
            else:
                cur = Block("bullet", [t], y0=y, y1=y, x0=s.x0)
                blocks.append(cur)
        order += 1
    return blocks


def analyse_page(page: dict, img: np.ndarray, fallback_body: tuple, state: dict, page_full: dict | None = None) -> PageResult:
    idx = page["page_index"]
    res = PageResult(index=idx)
    raw = page["lines"]
    res.low_conf = sum(1 for l in raw if l["conf"] < 0.5)

    segs, header = _page_segs(raw)
    full_segs = None
    if page_full is not None and page_full is not page and page_full.get("lines") is not raw:
        full_segs, header = _page_segs(page_full["lines"])  # header lives in pass 1 only
    res.header = " ".join(l["text"] for l in sorted(header, key=lambda l: l["box"][0])).strip()

    red = is_red_page(img)
    multi_col = state.get("section_kind") == "index"
    if full_segs is not None and (red or multi_col):
        segs = full_segs  # openers and multi-column index pages: full-page OCR reads them best
        full_segs = None
    if not segs:
        res.kind = "blank"
        return res

    # ---- chapter opener / red divider pages
    if red:
        big = sorted((s for s in segs if s.h > 0.026), key=lambda s: (s.y0, s.x0))
        nums = [s for s in big if re.fullmatch(r"\d{1,2}", s.text.strip())]
        title = join_lines([s.text for s in big if s not in nums])
        if not title or norm(title) in TOC_TITLES:
            res.kind = "toc"
            return res
        # chapter number: the big digit (often lost on the red background), else the
        # mini table of contents ("2.1", "2.2" …), else the caller counts on (+1)
        mini = [int(s.text.split(".")[0]) for s in segs if re.fullmatch(r"\d{1,2}\.\d{1,2}", s.text.strip())]
        res.kind = "chapter_start"
        res.chapter_num = int(nums[0].text) if nums else (max(set(mini), key=mini.count) if mini else None)
        res.chapter_title = title
        return res
    if norm(res.header) in TOC_TITLES or any(norm(s.text) in TOC_TITLES and s.h > 0.022 for s in segs):
        res.kind = "toc"
        return res

    if multi_col:
        res.blocks = index_blocks(segs)
        return res

    # ---- body column & margin notes
    left, right = choose_body(segs, fallback_body)
    res.body = (left, right)
    known_tops: dict = {}
    if full_segs and not multi_col:
        segs, known_tops = _merge_wide(segs, full_segs, left, right)
    if not multi_col:
        segs = [p for s in segs for p in (split_at_column(s, left, right) if not s.wide else [s])]
    margin, body = [], []
    for s in segs:
        if not multi_col and not s.wide and (s.x1 < left - 0.004 or s.x0 > right + 0.004):
            margin.append(s)
        else:
            body.append(s)
    for s in body:
        s.tint = is_tinted(background(img, s))

    heights = [s.h for s in body if len(s.text) > 30]
    med_h = median(heights) if heights else 0.015
    pitches = []
    srt = sorted(body, key=lambda s: s.yc)
    for a, b in zip(srt, srt[1:]):
        d = b.yc - a.yc
        if 0.01 < d < 0.024:
            pitches.append(d)
    pitch = median(pitches) if pitches else DEFAULT_PITCH

    # ---- chapter heading on a normal page: "Chapter 3: Markets", "Kapitel 3" + title line, big "3 Markets"
    #      (books whose chapters open on separate coloured pages are handled above)
    if state.get("section_kind") in ("chapter", "front"):
        found = _chapter_heading(body, med_h, state.get("chapter"))
        if found:
            num, title, used = found
            body = [s for s in body if s not in used]
            if num != state.get("chapter"):
                res.new_chapter, res.chapter_num, res.chapter_title = True, num, title

    # ---- numbered headings ("2.4.1 Process models")
    heads: dict[int, Block] = {}
    for s in sorted(body, key=lambda s: s.y0):
        txt = s.text.strip()
        m = NUM_HEAD_RE.match(txt)
        if (
            m
            and abs(s.x0 - left) < 0.02
            and len(txt) < 100
            and not txt.endswith(("-", ","))
            and s.h >= 0.85 * med_h
            and not CAPTION_RE.match(txt)
        ):
            chap = state.get("chapter")
            if chap is None or int(m.group(1).split(".")[0]) == chap:
                heads[id(s)] = Block("heading", [txt], level=min(m.group(1).count(".") + 1, 6), y0=s.y0, y1=s.y1, x0=s.x0, extra={"seg": s})
    # numbered headings that wrap onto a second line
    for b in list(heads.values()):
        hs = b.extra["seg"]
        text_x = hs.words[1][1] if len(hs.words) > 1 else hs.x0
        for t in sorted(body, key=lambda t: t.y0):
            if t.y0 > b.y1 - 0.004 and t.y0 - b.y1 < 0.006 and id(t) not in heads and abs(t.x0 - text_x) < 0.012 and t.h >= 0.9 * hs.h and t.x1 < right - 0.05:
                b.lines.append(t.text)
                b.y1 = t.y1
                heads[id(t)] = b
                break

    # ---- figures & tables (caption below; walk upwards to find the top)
    captions = [s for s in body if CAPTION_RE.match(s.text.strip()) and id(s) not in heads]
    consumed: set[int] = set()
    blocks: list[Block] = []
    found = find_figures(body, captions, set(heads), left, right, pitch)
    for n, (cap, top) in enumerate(found):
        if id(cap) in known_tops:
            found[n] = (cap, known_tops[id(cap)] + 0.002)
    for cap, top in found:
        m = CAPTION_RE.match(cap.text.strip())
        cap_lines = [cap.text]
        consumed.add(id(cap))
        if cap.x1 > right - 0.03:  # caption wrapped onto a 2nd line
            for t in sorted(body, key=lambda t: t.y0):
                if 0 <= t.y0 - cap.y1 < 0.006 and id(t) not in heads and id(t) not in consumed:
                    cap_lines.append(t.text)
                    consumed.add(id(t))
                    break
        kind = "table" if m.group(1).startswith("Tab") else "figure"
        inside = [s for s in body if top - 0.002 <= s.yc <= cap.y0 and id(s) not in consumed and id(s) not in heads]
        margins_in = [s for s in margin if top <= s.yc <= cap.y0]
        if cap.wide:
            bbox = ink_bbox(img, 0.04, top, 0.97, cap.y0 - 0.003, masks=margins_in)
        else:
            bbox = ink_bbox(img, max(0.04, left - 0.06), top, min(0.97, right + 0.06), cap.y0 - 0.003, masks=margins_in)
        blk = Block(kind, [join_lines(cap_lines)], y0=top, y1=cap.y1, x0=left)
        blk.extra["number"] = m.group(2)
        blk.extra["figtext"] = [s.text for s in sorted(inside, key=lambda s: (round(s.yc / 0.008), s.x0))]
        if bbox and (bbox[3] - bbox[1]) > 0.012:
            pad = 0.006
            crop = (max(0, bbox[0] - pad), max(top, bbox[1] - pad), min(1, bbox[2] + pad), min(cap.y0 - 0.002, bbox[3] + pad))
            res.crops.append((blk, crop))
            if kind == "table":
                blk.extra["md_table"] = table_markdown(img, inside, bbox)
        consumed.update(id(s) for s in inside)
        blocks.append(blk)

    # ---- big unnumbered headings ("Preface to the 3rd edition", "Literaturhinweise und Anmerkungen")
    known = ENDNOTE_TITLES | set(BACK_MATTER) | INTRO_TITLES
    big_run = [
        s for s in sorted(body, key=lambda s: s.y0)
        if id(s) not in heads and id(s) not in consumed
        and len(s.text) < 70 and s.x1 < right - 0.04
        and not s.text[:1].islower() and not s.text.rstrip().endswith(("-", ","))
        and not re.fullmatch(r"[\d\W]+", s.text)
        and (s.h > 1.6 * med_h or (norm(s.text) in known and s.h > 1.15 * med_h and abs(s.x0 - left) < 0.02))
    ]
    groups: list[list[Seg]] = []
    for s in big_run:
        if groups and s.y0 - groups[-1][-1].y1 < 0.02:
            groups[-1].append(s)
        else:
            groups.append([s])
    for g in groups:  # second line of a big heading set slightly smaller by OCR ("… und Autoren")
        last = g[-1]
        for t in sorted(body, key=lambda t: t.y0):
            if (t.y0 > last.y1 - 0.003 and t.y0 - last.y1 < 0.012 and abs(t.x0 - g[0].x0) < 0.012
                    and t.h > 1.25 * med_h and len(t.text) < 50 and t not in g and id(t) not in consumed
                    and not re.search(r"[.!?]$", t.text)):
                g.append(t)
                break
    for g in groups:
        txt = join_lines([s.text for s in g])
        b = Block("heading", [txt], level=2, y0=g[0].y0, y1=g[-1].y1, x0=g[0].x0, extra={"unnumbered": True})
        for s in g:
            heads[id(s)] = b
        if not res.section_title:
            res.section_title = txt

    # ---- rows: segments on the same baseline belong together
    rest = sorted((s for s in body if id(s) not in consumed), key=lambda s: s.y0)
    rows: list[list[Seg]] = []
    for s in rest:
        if id(s) in heads:
            rows.append([s])
            continue
        for r in reversed(rows[-3:]):
            if id(r[0]) in heads:
                continue
            ov = min(r[0].y1, s.y1) - max(r[0].y0, s.y0)
            if ov > 0.6 * min(r[0].h, s.h) and abs(r[0].yc - s.yc) < 0.5 * pitch:
                r.append(s)
                break
        else:
            rows.append([s])

    endnotes = state.get("endnotes", False)
    tinted = [s for s in body if s.tint and id(s) not in consumed]
    tint_left = min((s.x0 for s in tinted), default=left)
    tint_right = max((s.x1 for s in tinted), default=right)
    emitted_heads: set[int] = set()
    cur: Block | None = None
    prev_row = None

    def flush():
        nonlocal cur
        if cur is not None and cur.lines:
            blocks.append(cur)
        cur = None

    for r in rows:
        r.sort(key=lambda s: s.x0)
        s0 = r[0]
        if id(s0) in heads:
            hb = heads[id(s0)]
            flush()
            if id(hb) not in emitted_heads:
                emitted_heads.add(id(hb))
                blocks.append(hb)
                if norm(hb.text) in ENDNOTE_TITLES:
                    endnotes = True
            prev_row = None
            continue
        text = join_lines([s.text for s in r])
        x0, x1 = s0.x0, max(s.x1 for s in r)
        y0, y1 = min(s.y0 for s in r), max(s.y1 for s in r)
        tint = sum(s.tint for s in r) * 2 > len(r)
        lft, rgt = (tint_left, tint_right) if tint else (left, right)

        arrow = bool(re.match(rf"^{REF_WORDS}\s+\d", text)) and x0 > lft + 0.008
        if arrow:  # "▸ Abschnitt 3.2" at line start: the arrow was not recognised, only its indent
            text, x0 = "→ " + text, lft
        new_kind = None
        if BULLET_RE.match(text) and not re.match(rf"^\S\s*{REF_WORDS}\s+\d", text):
            new_kind, text = "bullet", BULLET_RE.sub("", text)
        elif endnotes and ENDNOTE_RE.match(text) and abs(x0 - left) < 0.02:
            new_kind = "endnote"
        elif UPPER_TITLE_RE.match(text) and len(text) < 40:
            new_kind = "boxtitle"
        elif tint and (cur is None or not cur.tint) and len(text) < 60 and x1 < rgt - 0.1 and not re.search(r"[.,;:!?]$", text):
            new_kind = "boxtitle"  # first, short line of an info box = its title

        start_new = new_kind is not None or cur is None
        if not start_new:
            p_text, p_x0, p_x1, p_yc = prev_row
            if (y0 + y1) / 2 - p_yc > 1.35 * pitch:
                start_new = True
            elif tint != cur.tint:
                start_new = True
            elif cur.kind in ("bullet", "endnote") and x0 < cur.extra.get("indent", x0) - 0.008:
                start_new = True  # back at the left edge: list item ended
            elif cur.kind == "boxtitle":
                start_new = True
            elif p_x1 < rgt - 0.04 and re.search(rf"[.!?:“”\")][{_SUP}'’*\"]{{0,2}}\d{{0,3}}$", p_text):
                start_new = True  # previous line was the short last line of a paragraph
            elif not endnotes and x0 > lft + 0.015 and p_x0 <= lft + 0.008 and cur.kind == "para":
                start_new = True  # indentation starts: a list follows
        if start_new:
            flush()
            kind = new_kind or "para"
            implicit = False
            if kind == "para" and not endnotes and x0 > lft + 0.015:
                kind, implicit = "bullet", True  # indented block = list item whose bullet glyph was not recognised
            cur = Block(kind, [], y0=y0, y1=y1, x0=x0, tint=tint)
            if kind == "bullet":
                w = r[0].words
                if implicit:
                    cur.extra.update(indent=x0, implicit=True)
                else:
                    cur.extra["indent"] = w[1][1] if len(w) > 1 and len(w[0][0]) == 1 else x0 + 0.012
            elif kind == "endnote":
                cur.extra["indent"] = x0 + 0.02
        cur.lines.append(text)
        cur.y1 = y1
        prev_row = (text, x0, x1, (y0 + y1) / 2)
    flush()
    state["endnotes"] = endnotes

    # ---- margin notes: group lines that belong together
    mnotes: list[list[Seg]] = []
    for s in sorted(margin, key=lambda s: s.y0):
        if mnotes and s.y0 - mnotes[-1][-1].y1 < 0.008 and abs(s.xc - mnotes[-1][-1].xc) < 0.15:
            mnotes[-1].append(s)
        else:
            mnotes.append([s])
    for g in mnotes:
        txt = join_lines([s.text for s in g])
        if not txt or re.fullmatch(r"[\W\d_]+", txt):
            continue
        if re.fullmatch(r"[A-ZÄÖÜ]", txt):  # alphabet letter in a bibliography
            blocks.append(Block("heading", [txt], level=3, y0=g[0].y0, y1=g[-1].y1, extra={"letter": True}))
            continue
        blocks.append(Block("margin", [txt], y0=g[0].y0, y1=g[-1].y1))

    # ---- reading order: margin notes go in front of the block they sit next to
    ordered = sorted((b for b in blocks if b.kind != "margin"), key=lambda b: b.y0)

    def sort_key(b: Block):
        if b.kind == "margin":
            host = next((o for o in ordered if o.y1 > b.y0 + 0.004), None)
            return ((host.y0 if host else b.y0) - 1e-6, 0)
        return (b.y0, 1)

    res.blocks = sorted(blocks, key=sort_key)
    return res
