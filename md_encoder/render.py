"""Stage 3 – assemble pages into sections (chapters) and write Markdown."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import lang
from layout import CAPTION_WORDS, ENDNOTE_RE, ENDNOTE_TITLES, Block, clean_text, join_lines, norm

TERMINAL = re.compile(r"[.!?:;“”\")\]]$")


@dataclass
class Section:
    key: str
    title: str
    number: int | None = None
    items: list = field(default_factory=list)  # Block | ("page", label)
    order: int = 0  # running number for back-matter sections

    @property
    def heading(self) -> str:
        return f"{lang.L['chapter']} {self.number}: {self.title}" if self.number is not None else self.title

    @property
    def filename(self) -> str:
        if self.number is not None:
            return f"{self.number:02d}-{slug(self.title)}.md"
        prefix = "00" if self.key == "front" else f"{90 + self.order:02d}"
        return f"{prefix}-{slug(self.title or self.key)}.md"


def slug(s: str) -> str:
    s = s.lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss"), ("&", lang.L["and"])):
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:60] or "section"


def gh_anchor(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


# --------------------------------------------------------------------------- continuation across pages


def last_text_block(items):
    for it in reversed(items):
        if isinstance(it, Block):
            if it.kind in ("para", "bullet", "endnote"):
                return it
            if it.kind in ("heading", "boxtitle"):
                return None
    return None


def merge_continuation(sec: Section, blocks: list[Block], body_left: float) -> list[Block]:
    """Glue a paragraph that runs over a page break back together. Boxes, margin
    notes and figures at the top of the new page are skipped over."""
    prev = last_text_block(sec.items)
    if prev is None:
        return blocks
    first_i = None
    for i, b in enumerate(blocks):
        if b.kind in ("margin", "figure", "table"):
            continue
        if b.tint != prev.tint and b.tint and not prev.tint:
            continue  # an info box interrupts the running paragraph
        first_i = i
        break
    if first_i is None:
        return blocks
    first = blocks[first_i]
    if first.tint != prev.tint:
        return blocks
    if first.kind == "bullet" and first.extra.get("implicit") and prev.kind == "bullet":
        pass  # indented continuation of a list item from the previous page
    elif first.kind != "para":
        return blocks
    ptxt, ftxt = prev.text, first.text
    indented = first.x0 > body_left + 0.008
    if prev.kind == "endnote":
        ok = not TERMINAL.search(ptxt) or ftxt[:1].islower()
    elif prev.kind == "bullet":
        ok = indented or not TERMINAL.search(ptxt) or ftxt[:1].islower()
    else:
        ok = ptxt.endswith("-") or ftxt[:1].islower() or not TERMINAL.search(ptxt)
    if not ok:
        return blocks
    prev.lines.extend(first.lines)
    return blocks[:first_i] + blocks[first_i + 1:]


# --------------------------------------------------------------------------- markdown


def _escape_start(t: str) -> str:
    if re.match(r"^(#|>|[-+*] |\d+[.)] )", t):
        return "\\" + t
    return t


def _caption(b: Block) -> str:
    t = clean_text(b.text)
    m = re.match(rf"^((?:{CAPTION_WORDS})\s*\d+(?:\.\d+)*(?!\.?\d)\s*[:.])\s*(.*)$", t)
    return f"**{m.group(1)}** {m.group(2)}" if m else t


def render_block(b: Block, sec: Section) -> str:
    k = b.kind
    if k == "heading":
        txt = clean_text(b.text)
        if b.extra.get("unnumbered"):
            if norm(txt) == norm(sec.title):
                return ""
            return f"## {txt}"
        return "#" * max(2, b.level) + " " + txt
    if k == "boxtitle":
        return f"**{b.text.strip().title()}**"
    if k == "para":
        return _escape_start(clean_text(b.text))
    if k == "bullet":
        return "- " + clean_text(b.text)
    if k == "endnote":
        m = ENDNOTE_RE.match(b.text)
        return f"- **{m.group(1)}** {clean_text(m.group(2))}" if m else "- " + clean_text(b.text)
    if k == "margin":
        return f"> **{clean_text(b.text)}**"
    if k in ("figure", "table"):
        out = []
        cap = clean_text(b.text)
        if b.extra.get("image"):
            alt = cap.replace("[", "(").replace("]", ")")
            out.append(f"![{alt}]({b.extra['image']})")
        out.append(_caption(b))
        if k == "table" and b.extra.get("md_table"):
            out.append(b.extra["md_table"])
        elif b.extra.get("figtext"):
            label = lang.L["table_text"] if k == "table" else lang.L["fig_text"]
            txt = " · ".join(clean_text(t) for t in b.extra["figtext"] if t.strip())
            out.append(f"<details><summary>{label}</summary>\n\n{txt}\n\n</details>")
        return "\n\n".join(out)
    return clean_text(b.text)


def repair_endnotes(items: list) -> None:
    """An endnote whose number OCR missed shows up as a plain paragraph between
    note n-1 and n+1 – give it its number back."""
    blocks = [b for b in items if isinstance(b, Block)]
    in_notes, last = False, None
    for i, b in enumerate(blocks):
        if b.kind == "heading":
            in_notes, last = norm(b.text) in ENDNOTE_TITLES, None
            continue
        if not in_notes:
            continue
        if b.kind == "endnote":
            m = ENDNOTE_RE.match(b.text)
            last = int(m.group(1)) if m else last
        elif b.kind == "para" and last is not None:
            nxt = next((o for o in blocks[i + 1:] if o.kind in ("endnote", "heading")), None)
            m = ENDNOTE_RE.match(nxt.text) if nxt is not None and nxt.kind == "endnote" else None
            if m is None or int(m.group(1)) == last + 2:
                b.kind = "endnote"
                b.lines[0] = f"{last + 1} {b.lines[0]}"
                last += 1


def render_section(sec: Section, page_markers: bool = True) -> str:
    repair_endnotes(sec.items)
    parts: list[str] = [f"# {sec.heading}"]
    callout: list[str] = []
    deferred_pages: list[str] = []
    prev_kind = None

    def close_callout():
        nonlocal callout
        if callout:
            body = "\n".join(callout).rstrip("\n")
            quoted = "\n".join(("> " + l) if l else ">" for l in body.split("\n"))
            parts.append("> [!NOTE]\n" + quoted)
            callout = []
        for p in deferred_pages:
            parts.append(p)
        deferred_pages.clear()

    items = sec.items
    for n, it in enumerate(items):
        if isinstance(it, Block) and it.kind == "margin" and callout:
            nxt = next((o for o in items[n + 1:] if isinstance(o, Block) and o.kind != "margin"), None)
            if nxt is not None and nxt.tint:
                callout += ["", render_block(it, sec)]  # margin note beside a box → nested quote
                continue
        if not isinstance(it, Block):
            if page_markers:
                marker = f"<!-- {lang.L['page']} {it[1]} -->"
                (deferred_pages if callout else parts).append(marker)
            prev_kind = None
            continue
        md = render_block(it, sec)
        if not md:
            continue
        listy = it.kind in ("bullet", "endnote")
        if it.tint and it.kind in ("para", "bullet", "boxtitle", "endnote"):
            if callout and not (listy and prev_kind == it.kind):
                callout.append("")
            callout.append(md)
            prev_kind = it.kind
            continue
        close_callout()
        if listy and prev_kind == it.kind and parts:
            parts[-1] += "\n" + md
        else:
            parts.append(md)
        prev_kind = it.kind
    close_callout()
    return "\n\n".join(parts).strip() + "\n"


def write_book(sections: list[Section], out_dir: Path, title: str, source: str, page_markers: bool = True,
               ocr: bool = True) -> list[Path]:
    written = []
    full = [f"# {title}\n"]
    how = lang.L["how_ocr"] if ocr else lang.L["how_text"]
    toc = [
        f"# {title}",
        "",
        lang.L["generated"].format(source=source, date=date.today().isoformat(), how=how),
        "",
        lang.L["page_note"],
        "",
        f"## {lang.L['contents']}",
        "",
    ]
    for sec in sections:
        if not any(isinstance(i, Block) for i in sec.items):
            continue
        md = render_section(sec, page_markers)
        path = out_dir / sec.filename
        path.write_text(md, encoding="utf-8")
        written.append(path)
        full.append(md)
        toc.append(f"- [{sec.heading}]({sec.filename})")
        for it in sec.items:
            if isinstance(it, Block) and it.kind == "heading" and not it.extra.get("unnumbered") and not it.extra.get("letter") and it.level == 2:
                h = clean_text(it.text)
                toc.append(f"  - [{h}]({sec.filename}#{gh_anchor(h)})")
    toc += ["", lang.L["one_file"], ""]
    (out_dir / "README.md").write_text("\n".join(toc), encoding="utf-8")
    (out_dir / "full-book.md").write_text("\n\n---\n\n".join(full), encoding="utf-8")
    return written
