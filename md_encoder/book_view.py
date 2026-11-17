"""Stage 5 – "book look" variant.

Writes one Markdown file whose pages imitate the printed book: page frames with
running header and page number, margin notes in the outer margin column, red
section headings, pink info boxes, red square bullets, chapter-opener pages
and a table of contents.

The layout is built from inline-styled HTML, which Markdown allows. Inline
styles (not a <style> block) are used because Obsidian and the VS Code preview
both keep them. GitHub strips styles, so it shows plain text there.
Each page is one HTML block without blank lines, so no Markdown parser splits it.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import lang
from layout import BACK_MATTER, CAPTION_WORDS, ENDNOTE_RE, Block, clean_text, norm
from render import repair_endnotes

RED = "#d9282b"
PINK = "#ffebeb"
INK = "#1a1a1a"
SERIF = "'Lucida Bright','Charter','Iowan Old Style',Georgia,serif"
SANS = "'Frutiger','Myriad Pro','Avenir Next','Helvetica Neue',Arial,sans-serif"
COND = "'Myriad Pro Condensed','Avenir Next Condensed','Helvetica Neue',Arial,sans-serif"
PW, PH = 820, 1160  # page box in px (A4 proportions)

# outer margin column + gap, in fractions of the page width (measured on the scans)
MARGIN_W = 0.177
GAP = {0: 0.014, 1: 0.017}

PAGE = (
    "position:relative;box-sizing:border-box;width:100%;max-width:{w}px;min-height:{h}px;margin:0 auto 32px;"
    "background:{bg};color:{ink};box-shadow:0 3px 16px rgba(0,0,0,.28);overflow:hidden;"
    "padding:{pt}px {pr}% 96px {pl}%;font:14px/1.38 {serif};text-align:justify;hyphens:auto;-webkit-hyphens:auto"
)


def esc(t: str) -> str:
    s = html.escape(clean_text(t, md=False), quote=False)
    return s.replace("→ ", f'<span style="color:{RED}">▸</span>&nbsp;')


def _geometry(idx: int, body: tuple[float, float]) -> dict:
    even = idx % 2 == 0
    left, right = body
    if even:
        pl, pr = left - MARGIN_W - GAP[0], 1 - right
    else:
        pl, pr = left, 1 - (right + GAP[1] + MARGIN_W)
    cw = 1 - pl - pr
    return {
        "even": even, "pl": pl * 100, "pr": pr * 100,
        "mc": MARGIN_W / cw * 100, "gap": GAP[0 if even else 1] / cw * 100,
        "body_w": right - left, "left": left,
    }


def _frame(idx: int, g: dict, header: str, label: str, inner: str, bg: str = "#fff") -> str:
    style = PAGE.format(w=PW, h=PH, bg=bg, ink=INK, pt=88, pr=f"{g['pr']:.1f}", pl=f"{g['pl']:.1f}", serif=SERIF)
    head = ""
    if header:
        h = re.sub(r"\s*\|\s*", " ", header).strip()
        m = re.match(r"^(\d{1,2})\s+(.*)$", h)
        if g["even"] and m:
            h_html = f'<span style="color:{RED};font-size:25px;margin-right:12px">{m.group(1)}</span>{html.escape(m.group(2))}'
        else:
            h_html = html.escape(h)
        just = "flex-start" if g["even"] else "flex-end"
        head = (
            f'<div style="position:absolute;top:20px;left:4.7%;right:4.7%;display:flex;justify-content:{just};'
            f'align-items:baseline;border-bottom:1px solid {INK};padding-bottom:2px;font:700 12.5px/1.2 {SANS}">{h_html}</div>'
        )
    foot = ""
    if label:
        side = "left:5%" if g["even"] else "right:5%"
        bar = "border-right" if g["even"] else "border-left"
        foot = (
            f'<div style="position:absolute;bottom:30px;{side};font:11px {SERIF};{bar}:1px solid {INK};'
            f'padding:26px 8px 0 8px;line-height:1">{html.escape(label)}</div>'
        )
    return f'<div lang="{lang.code}" style="{style}">{head}{inner}{foot}</div>'


def _row(g: dict, margin_html: str, body_html: str) -> str:
    al = "right" if g["even"] else "left"
    m = (
        f'<div style="flex:0 0 {g["mc"]:.2f}%;max-width:{g["mc"]:.2f}%;text-align:{al};hyphens:none;'
        f'font:600 11.5px/1.25 {COND};color:{INK}">{margin_html}</div>'
    )
    gap = f'<div style="flex:0 0 {g["gap"]:.2f}%"></div>'
    b = f'<div style="flex:1;min-width:0">{body_html}</div>'
    return f'<div style="display:flex">{m + gap + b if g["even"] else b + gap + m}</div>'


def _heading(b: Block) -> str:
    t = clean_text(b.text, md=False)
    if b.extra.get("letter"):
        return f'<div style="font:700 17px {SANS};color:{RED};margin:12px 0 4px">{html.escape(t)}</div>'
    if b.extra.get("unnumbered"):
        return f'<div style="font:700 25px/1.2 {SANS};color:{RED};margin:6px 0 18px;text-align:left">{html.escape(t)}</div>'
    m = re.match(r"^([\d.]+)\s+(.*)$", t)
    num, rest = (m.group(1), m.group(2)) if m else ("", t)
    size, color, top = {2: (20, RED, 22), 3: (16, INK, 18), 4: (14.5, INK, 14)}.get(b.level, (14, INK, 12))
    return (
        f'<div style="font:700 {size}px/1.25 {SANS};color:{color};margin:{top}px 0 9px;text-align:left">'
        f'{html.escape(num)}&nbsp;&nbsp;{html.escape(rest)}</div>'
    )


def _bullet(text: str, sans: bool) -> str:
    return (
        f'<div style="display:flex;margin:0 0 4px"><span style="flex:0 0 15px;color:{RED};font-size:8px;'
        f'line-height:{19 if not sans else 18}px">■</span><span style="flex:1;min-width:0">{esc(text)}</span></div>'
    )


def _figure(b: Block, g: dict) -> str:
    cap = clean_text(b.text, md=False)
    m = re.match(rf"^((?:{CAPTION_WORDS})\s*\d+(?:\.\d+)*(?!\.?\d)\s*[:.])\s*(.*)$", cap)
    cap_html = (
        f'<b style="color:{RED}">{html.escape(m.group(1))}</b>&nbsp;{html.escape(m.group(2))}' if m else html.escape(cap)
    )
    img = ""
    if b.extra.get("image"):
        x0, _, x1, _ = b.extra.get("crop", (g["left"], 0, g["left"] + g["body_w"], 0))
        width = (x1 - x0) / g["body_w"] * 100
        shift = (x0 - g["left"]) / g["body_w"] * 100
        img = (
            f'<img src="{b.extra["image"]}" alt="{html.escape(cap)}" '
            f'style="display:block;width:{width:.1f}%;max-width:none;margin:0 0 0 {shift:.1f}%;background:#fff">'
        )
    return (
        f'<div style="margin:12px 0 14px">{img}<div style="font:12px/1.3 {SANS};margin-top:7px;text-align:left">'
        f"{cap_html}</div></div>"
    )


def _box(blocks: list[Block]) -> str:
    title = blocks[0] if blocks and blocks[0].kind == "boxtitle" else None
    rest = blocks[1:] if title else blocks
    t = title.text.strip() if title else ""
    summary = t.upper() == "ZUSAMMENFASSUNG"
    font = f"14px/1.38 {SERIF}" if summary else f"13px/1.4 {SANS}"
    inner = ""
    for b in rest:
        if b.kind == "bullet":
            inner += _bullet(b.text, True)
        else:
            inner += f'<p style="margin:0 0 7px">{esc(b.text)}</p>'
    if summary:
        head = (
            f'<div style="background:{RED};color:#fff;font:700 12px {SANS};letter-spacing:.14em;'
            f'padding:5px 14px;margin:-10px -14px 10px">{html.escape(t)}</div>'
        )
        return f'<div style="background:{PINK};padding:10px 14px 6px;margin:4px 0 14px;font:{font}">{head}{inner}</div>'
    if t.isupper():  # all-caps label, e.g. a chapter-opener tag
        head = f'<div style="color:{RED};font:700 13px {SANS};letter-spacing:.14em;margin-bottom:3px">{html.escape(t)}</div>'
        return (
            f'<div style="background:{PINK};border-left:4px solid {RED};padding:10px 14px 6px;margin:4px 0 16px;'
            f'font:{font}">{head}{inner}</div>'
        )
    head = f'<div style="font:700 13.5px {SANS};margin-bottom:3px">{esc(t)}</div>' if t else ""
    return f'<div style="background:{PINK};padding:10px 14px 6px;margin:4px 0 14px;font:{font}">{head}{inner}</div>'


def _margin_html(notes: list[Block], top_y: float) -> str:
    out, cursor = "", top_y
    for n in notes:
        gap = max(0.0, (n.y0 - cursor) * PH * 0.92)
        out += f'<div style="margin-top:{gap:.0f}px">{esc(n.text)}</div>'
        cursor = n.y1
    return out


def _content_page(rec: dict) -> str:
    g = _geometry(rec["idx"], rec["body"])
    blocks: list[Block] = rec["blocks"]
    if rec.get("section_kind") == "index":
        items = ""
        for b in blocks:
            if b.kind == "heading":
                items += f'<div style="font:700 15px {SANS};color:{RED};margin:8px 0 3px;break-after:avoid">{esc(b.text)}</div>'
            else:
                items += f'<div style="padding-left:12px;text-indent:-12px">{esc(b.text)}</div>'
        inner = f'<div style="column-count:2;column-gap:6%;font:12.5px/1.5 {SANS};text-align:left">{items}</div>'
        return _frame(rec["idx"], g, rec["header"], rec["label"], inner)

    rows, pending = [], []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b.kind == "margin" or (b.kind == "heading" and b.extra.get("letter") and rec.get("section_kind") != "index"):
            pending.append(b)
            i += 1
            continue
        if b.tint and b.kind in ("para", "bullet", "boxtitle", "endnote"):
            run = []
            def in_box(j):
                return blocks[j].tint and blocks[j].kind in ("para", "bullet", "boxtitle", "endnote")

            def margin_inside(j):  # a margin note belongs to the box only if the box continues after it
                k = j
                while k < len(blocks) and blocks[k].kind == "margin":
                    k += 1
                return k < len(blocks) and in_box(k)

            while i < len(blocks) and (in_box(i) or (blocks[i].kind == "margin" and margin_inside(i))):
                (pending if blocks[i].kind == "margin" else run).append(blocks[i])
                i += 1
            rows.append(_row(g, _margin_html(pending, run[0].y0), _box(run)))
            pending = []
            continue
        if b.kind == "heading":
            body = _heading(b)
        elif b.kind in ("figure", "table"):
            body = _figure(b, g)
        elif b.kind == "bullet":
            body = _bullet(b.text, False)
        elif b.kind == "endnote":
            m = ENDNOTE_RE.match(b.text)
            num, txt = (m.group(1), m.group(2)) if m else ("", b.text)
            body = (
                f'<div style="display:flex;margin:0 0 4px;text-align:left"><span style="flex:0 0 7%">{num}</span>'
                f'<span style="flex:1;min-width:0;text-align:justify">{esc(txt)}</span></div>'
            )
        elif b.kind == "boxtitle":
            body = f'<div style="font:700 13.5px {SANS};margin:10px 0 4px">{esc(b.text)}</div>'
        else:
            body = f'<p style="margin:0 0 8px">{esc(b.text)}</p>'
        rows.append(_row(g, _margin_html(pending, b.y0), body))
        pending = []
        i += 1
    if pending:
        rows.append(_row(g, _margin_html(pending, pending[0].y0), ""))
    return _frame(rec["idx"], g, rec["header"], rec["label"], "".join(rows))


def _leader_line(num: str, title: str, page: str, bold: bool, indent: int, numw: int,
                 numcolor: str = INK, size: float = 14) -> str:
    w = 700 if bold else 400
    return (
        f'<div style="display:flex;align-items:baseline;font:{w} {size}px/1.8 {SANS};padding-left:{indent}px;text-align:left">'
        f'<span style="flex:0 0 {numw}px;color:{numcolor}">{html.escape(num)}</span><span>{html.escape(title)}</span>'
        f'<span style="flex:1;min-width:12px;border-bottom:2px dotted #444;margin:0 6px;position:relative;top:-4px"></span>'
        f'<span style="flex:0 0 34px;text-align:right">{html.escape(page)}</span></div>'
    )


def _red_page(panel: str, number: str = "") -> str:
    style = PAGE.format(w=PW, h=PH, bg=RED, ink=INK, pt=70, pr="22", pl="10", serif=SERIF)
    num = (
        f'<div style="position:absolute;top:66px;right:8%;font:700 84px/1 {SANS};color:#fff">{html.escape(number)}</div>'
        if number else ""
    )
    return f'<div lang="{lang.code}" style="{style}"><div style="background:#fff;padding:30px 32px 32px">{panel}</div>{num}</div>'


def _panel_title(t: str) -> str:
    return (
        f'<div style="font:700 30px/1.2 {SANS};color:{RED};border-bottom:2px solid {RED};padding-bottom:10px;'
        f'margin-bottom:22px;text-align:left">{html.escape(t)}</div>'
    )


def _toc_entries(records: list[dict]) -> list[tuple]:
    """(kind, number, title, page, chapter) for every heading worth listing."""
    out, chapter = [], None
    for r in records:
        if r["kind"] == "chapter_start":
            chapter = r["chapter_num"]
            out.append(("chapter", str(r["chapter_num"]), r["chapter_title"], r["label"], chapter))
            continue
        for b in r["blocks"]:
            if b.kind != "heading" or b.extra.get("letter"):
                continue
            t = clean_text(b.text, md=False)
            if b.extra.get("unnumbered"):
                if norm(t) in BACK_MATTER:
                    out.append(("back", "", t, r["label"], None))
                elif chapter is None:
                    out.append(("front", "", t, r["label"], None))
                else:
                    out.append(("notes", "", t, r["label"], chapter))
            elif b.level in (2, 3):
                m = re.match(r"^([\d.]+)\s+(.*)$", t)
                if m:
                    out.append((f"h{b.level}", m.group(1), m.group(2), r["label"], chapter))
    return out


def build(records: list[dict], out_path: Path, title: str, cover: str | None = None, subtitle: str = "") -> Path:
    repair_endnotes([b for r in records for b in r["blocks"]])
    toc = _toc_entries(records)
    parts = [f"<!-- {title}: {lang.L['layout_note']} -->"]

    if cover:
        parts.append(
            f'<div style="max-width:{PW}px;margin:0 auto 32px;box-shadow:0 3px 16px rgba(0,0,0,.28)">'
            f'<img src="{cover}" alt="Cover" style="display:block;width:100%"></div>'
        )

    # table of contents in the style of the book's red contents page
    lines = []
    for kind, num, t, page, _ in toc:
        if kind in ("front", "back"):
            lines.append(_leader_line("", t, page, True, 0, 0))
        elif kind == "chapter":
            lines.append('<div style="height:10px"></div>' + _leader_line(f"{lang.L['chapter']} {num}", t, page, True, 0, 92, RED))
        elif kind == "h2":
            lines.append(_leader_line(num, t, page, False, 0, 46))
        elif kind == "h3":
            lines.append(_leader_line(num, t, page, False, 46, 50, size=13))
        elif kind == "notes":
            lines.append(_leader_line("", t, page, False, 46, 0, size=13))
    parts.append(_red_page(_panel_title(lang.L["toc"]) + "".join(lines)))

    for n, r in enumerate(records):
        if r["kind"] == "chapter_start":
            entries = []
            for kind, num, t, page, ch in toc:
                if ch == r["chapter_num"] and kind in ("h2", "notes"):
                    entries.append(_leader_line(num, t, page, True, 0, 46))
            parts.append(_red_page(_panel_title(r["chapter_title"]) + "".join(entries), str(r["chapter_num"])))
        else:
            parts.append(_content_page(r))

    out_path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return out_path
