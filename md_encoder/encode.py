#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pymupdf>=1.24",
#   "numpy",
#   "markdown",
#   "pyobjc-framework-Vision; sys_platform == 'darwin'",
# ]
# ///
"""md_encoder – turn a book PDF (digital or scanned) into structured Markdown.

    uv run md_encoder/encode.py "Book.pdf" -o mybook/

Pipeline:  PDF page ─► text layer, or bitmap ─► Apple Vision OCR (cached) ─► layout analysis
           ─► chapters / headings / paragraphs / margin notes / boxes /
              figures + tables (cropped images) ─► one .md per chapter + README + full-book.md
           + book.json (title, language, PDF) for the search and the reader in rag/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pymupdf  # noqa: E402

import book_view  # noqa: E402
import html_view  # noqa: E402
import lang  # noqa: E402
import layout  # noqa: E402
import ocr  # noqa: E402
from dataclasses import replace  # noqa: E402

from render import Section, merge_continuation, slug, write_book  # noqa: E402


def parse_ranges(spec: str | None, n: int) -> list[int]:
    if not spec:
        return []
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a or 0), min(int(b or n - 1), n - 1) + 1))
        else:
            out.add(int(part))
    return sorted(i for i in out if 0 <= i < n)


def page_image(doc, idx: int, dpi: int = 100) -> np.ndarray:
    pix = doc[idx].get_pixmap(dpi=dpi)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]


def page_labels(cache_dir: Path, pages: list[int]) -> dict[int, str]:
    """Book page numbers: OCR'd from the footer, gaps filled via the dominant offset."""
    found, offsets = {}, Counter()
    for i in pages:
        for l in ocr.load_page(cache_dir, i)["lines"]:
            t = l["text"].strip()
            if l["box"][1] > layout.FOOTER_Y and re.fullmatch(r"\d{1,4}", t):
                found[i] = int(t)
                offsets[int(t) - i] += 1
    off = offsets.most_common(1)[0][0] if offsets else 1
    return {i: str(found.get(i, i + off)) for i in pages}


def fallback_bodies(cache_dir: Path, pages: list[int]) -> dict[int, tuple]:
    """Median body column for even/odd pages – used on pages with too little text."""
    acc = {0: [], 1: []}
    for i in pages:
        segs = [s for l in ocr.load_page(cache_dir, i)["lines"] for s in layout.split_line(l)]
        b = layout.estimate_body(segs)
        if b:
            acc[i % 2].append(b)
    out = {}
    for k, v in acc.items():
        out[k] = (float(np.median([a for a, _ in v])), float(np.median([b for _, b in v]))) if v else (0.1, 0.9)
    return out


def column_crops(cache_dir: Path, pages: list[int], fb: dict) -> dict[int, list]:
    """Crop rectangles for pass 2: the body column plus the margin strip(s).
    Pages without a regular body column (openers, index, title pages) keep pass 1."""
    crops = {}
    for i in pages:
        segs = [s for l in ocr.load_page(cache_dir, i)["lines"] for s in layout.split_line(l)]
        est = layout.estimate_body(segs)
        if not est:
            continue  # no running text (index, title page, full-page figure): keep pass 1
        # union of this page's own text block and the book-wide column, so that inset
        # boxes or a wider first page of the bibliography are never cut
        L, R = min(est[0], fb[i % 2][0]), max(est[1], fb[i % 2][1])
        # header/footer and the vertical licence watermark stay out of the crops
        y0, y1 = layout.HEADER_Y - 0.01, layout.FOOTER_Y + 0.005
        out = [(max(0.0, L - 0.007), y0, min(1.0, R + 0.008), y1)]
        if L > 0.15:
            out.append((0.035, y0, L - 0.004, y1))
        if R < 0.85:
            out.append((R + 0.004, y0, 0.975, y1))
        crops[i] = out
    return crops


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Encode a book PDF (digital or scanned) into structured Markdown files.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="output folder (default: <pdf name>-md next to the PDF)")
    ap.add_argument("--title", help="book title for README / full-book.md")
    ap.add_argument("--pages", help="only process these PDF page indices, e.g. '15-60' (0-based)")
    ap.add_argument("--skip-pages", help="PDF page indices to leave out, e.g. '0-4,13' (cover, ads, imprint)")
    ap.add_argument("--dpi", type=int, default=250, help="render resolution for OCR (default 250)")
    ap.add_argument("--fig-dpi", type=int, default=200, help="resolution of cropped figures (default 200)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--no-figures", action="store_true", help="do not crop figure/table images")
    ap.add_argument("--no-page-markers", action="store_true", help="omit <!-- S. n --> page comments")
    ap.add_argument("--force-ocr", action="store_true", help="ignore the OCR cache")
    ap.add_argument("--cover-page", type=int, default=0, help="PDF page used as cover in the book-layout variant (-1: none)")
    ap.add_argument("--lang", default="auto", help="book language for labels: de, en or auto (default: detect)")
    ap.add_argument("--ocr-langs", help="OCR languages for scanned pages, e.g. 'fr-FR,en-US' (default: from --lang)")
    ap.add_argument("--always-ocr", action="store_true", help="OCR every page even if the PDF has a text layer")
    ap.add_argument("--drop", help="regex: drop lines matching it (licence stamps, watermarks)")
    ap.add_argument("--no-book-view", action="store_true", help="skip the book-layout Markdown variant")
    args = ap.parse_args(argv)

    pdf = args.pdf.resolve()
    if not pdf.exists():
        ap.error(f"{pdf} not found")
    if args.drop:
        layout.WATERMARK_RE = re.compile(f"{layout.WATERMARK_RE.pattern}|{args.drop}")
    doc = pymupdf.open(pdf)
    n = doc.page_count
    clean_stem = re.sub(r"\s*\(.*?\)\s*", " ", pdf.stem).strip()
    title = args.title or clean_stem
    out = (args.out or pdf.parent / f"{slug(clean_stem)}-md").resolve()
    cache = out / ".cache" / "ocr"
    if args.force_ocr and cache.exists():
        shutil.rmtree(cache)

    pages = parse_ranges(args.pages, n) or list(range(n))
    skip = set(parse_ranges(args.skip_pages, n))
    t0 = time.time()
    print(f"▶ {pdf.name}: {n} pages → {out}")

    # ---- stage 1: OCR (cached). Pass 1 = whole pages → learn the column layout;
    #      pass 2 = body column and margin column OCR'd separately (much cleaner).
    ocr_langs = (args.ocr_langs.split(",") if args.ocr_langs
                 else lang.LABELS[args.lang]["ocr"] if args.lang in lang.LABELS else None)
    try:
        ocr.run_ocr(pdf, cache, pages, args.dpi, args.workers, langs=ocr_langs, text_layer=not args.always_ocr)
        fb = fallback_bodies(cache, pages)
        ocr.run_ocr(pdf, cache, pages, args.dpi, args.workers, crops=column_crops(cache, pages, fb), langs=ocr_langs)
    except RuntimeError as e:
        ap.error(str(e))
    used_ocr = any(ocr.is_ocr_page(cache, i) for i in pages)
    code = args.lang if args.lang != "auto" else lang.detect(
        l["text"] for i in pages if i not in skip for l in ocr.load_page(cache, i)["lines"])
    lang.set_language(code)
    print(f"  language: {code}{' (detected)' if args.lang == 'auto' else ''}, "
          f"{'OCR' if used_ocr else 'text layer'}")

    # ---- stage 2: layout, page by page (sequential – chapters/endnotes carry state)
    labels = page_labels(cache, pages)
    # remove only what a previous run of this tool wrote (never touch other files)
    manifest = out / ".cache" / "manifest.txt"
    if manifest.exists():
        for rel in manifest.read_text().splitlines():
            f = out / rel
            if rel and f.is_file():
                f.unlink()
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    state = {"chapter": None, "endnotes": False, "section_kind": "front"}
    sections = [Section("front", "")]
    used_names: set[str] = set()
    stats = Counter()
    records: list[dict] = []  # per-page snapshot for the book-layout variant

    for idx in pages:
        if idx in skip:
            stats["skipped"] += 1
            continue
        pj = ocr.load_page(cache, idx, prefer_columns=True)
        pj_full = ocr.load_page(cache, idx)
        img = page_image(doc, idx)
        res = layout.analyse_page(pj, img, fb[idx % 2], state, pj_full)

        # back matter (bibliography / index) switches the section; index pages are multi-column
        bm = layout.BACK_MATTER.get(norm_title(res.section_title)) or layout.BACK_MATTER.get(norm_title(res.header))
        if bm and sections[-1].title != bm:
            kind = "index" if bm in layout.INDEX_TITLES else "back"
            state.update(section_kind=kind, endnotes=False, chapter=None)
            sections.append(Section(f"back-{slug(bm)}", bm, order=sum(1 for x in sections if x.key.startswith("back"))))
            res = layout.analyse_page(pj, img, fb[idx % 2], state, pj_full)

        if res.kind == "chapter_start":
            if res.chapter_num is None:
                res.chapter_num = (state["chapter"] or 0) + 1
            state.update(chapter=res.chapter_num, endnotes=False, section_kind="chapter")
            sections.append(Section(f"ch{res.chapter_num:02d}", res.chapter_title, res.chapter_num))
            records.append({"idx": idx, "kind": "chapter_start", "chapter_num": res.chapter_num,
                            "chapter_title": res.chapter_title, "label": labels[idx], "blocks": []})
            stats["chapters"] += 1
            continue
        if res.kind in ("toc", "blank", "divider"):
            stats[res.kind] += 1
            continue

        # chapter heading on a normal page ("Chapter 3: Markets")
        if res.new_chapter:
            num = res.chapter_num
            state.update(chapter=num, endnotes=False, section_kind="chapter")
            sections.append(Section(f"ch{num:02d}", res.chapter_title, num))
            stats["chapters"] += 1
            res = layout.analyse_page(pj, img, fb[idx % 2], state, pj_full)

        # chapter fallback: running header says we are in a new chapter
        m = re.match(r"^(\d{1,2})\s*\|?\s+(\D.*)$", res.header)
        if m and state["section_kind"] in ("chapter", "front") and int(m.group(1)) != state["chapter"]:
            num = int(m.group(1))
            state.update(chapter=num, endnotes=False, section_kind="chapter")
            sections.append(Section(f"ch{num:02d}", m.group(2).strip(), num))
            res = layout.analyse_page(pj, img, fb[idx % 2], state, pj_full)

        sec = sections[-1]
        if not sec.title and res.section_title:
            sec.title = res.section_title

        # figures → PNG crops
        page = doc[idx]
        W, H = page.rect.width, page.rect.height
        for blk, (x0, y0, x1, y1) in res.crops:
            stats[blk.kind] += 1
            blk.extra["crop"] = (x0, y0, x1, y1)
            if args.no_figures:
                continue
            base = (f"{lang.L['fig_prefix']}-" if blk.kind == "figure" else "tab-") + blk.extra["number"].replace(".", "-")
            name = base
            k = 2
            while name in used_names:
                name, k = f"{base}-{k}", k + 1
            used_names.add(name)
            pix = page.get_pixmap(dpi=args.fig_dpi, clip=pymupdf.Rect(x0 * W, y0 * H, x1 * W, y1 * H))
            pix.save(assets / f"{name}.png")
            blk.extra["image"] = f"assets/{name}.png"
        stats["tables_md"] += sum(1 for b, _ in res.crops if b.kind == "table" and b.extra.get("md_table"))
        stats["low_conf_lines"] += res.low_conf

        records.append({
            "idx": idx, "kind": "content", "label": labels[idx], "header": res.header,
            "body": fb[idx % 2], "section_kind": state["section_kind"],
            "blocks": [replace(b, lines=list(b.lines), extra=dict(b.extra)) for b in res.blocks],
        })
        blocks = merge_continuation(sec, res.blocks, res.body[0])
        sec.items.append(("page", labels[idx]))
        sec.items.extend(blocks)
        stats["pages"] += 1

    if not sections[0].title:
        sections[0].title = lang.L["front"]
    written = write_book(sections, out, title, pdf.name, page_markers=not args.no_page_markers, ocr=used_ocr)
    produced = [p.relative_to(out).as_posix() for p in written]
    html_view.build(out / "full-book.md", out / "full-book.html", title)
    cover = None
    if 0 <= args.cover_page < n:
        doc[args.cover_page].get_pixmap(dpi=150).save(assets / "cover.png")
        cover = "assets/cover.png"
        used_names.add("cover")
    produced += ["README.md", "full-book.md", "full-book.html"]
    if not args.no_book_view:
        book_view.build(records, out / lang.L["layout_file"], title, cover)
        produced.append(lang.L["layout_file"])
    produced += [f"assets/{n}.png" for n in sorted(used_names)]
    manifest.write_text("\n".join(produced) + "\n")
    write_book_info(out, title=title, language=code, pdf=pdf, pages=n)

    print(
        f"✔ {stats['pages']} pages → {len(written)} Markdown files, "
        f"{stats['figure']} figures, {stats['table']} tables ({stats['tables_md']} rebuilt as Markdown), "
        f"{stats['chapters']} chapters, skipped {stats['skipped'] + stats['toc'] + stats['blank']} pages "
        f"in {time.time() - t0:.0f}s"
    )
    if stats["low_conf_lines"]:
        print(f"  note: {stats['low_conf_lines']} OCR lines had low confidence")
    for p in written:
        print("   ", p.relative_to(out.parent) if out.parent in p.parents else p)
    print(f"   {(out / 'README.md')}")
    print(f"   {(out / 'full-book.html')}  ← reading view")
    if not args.no_book_view:
        print(f"   {(out / lang.L['layout_file'])}  ← book-layout variant")
    print(f"   {(out / 'book.json')}  ← used by rag/ (search, evaluation, reader)")
    return 0


def write_book_info(out: Path, **info) -> None:
    """book.json describes the book for rag/. Keys the encoder does not own are kept,
    e.g. a custom "query_task" for the embedding model."""
    path = out / "book.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    pdf: Path = info.pop("pdf")
    data.update(info, pdf=os.path.relpath(pdf, out))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm_title(s: str) -> str:
    return re.sub(r"^\d+\s*\|?\s*", "", layout.norm(s or ""))


if __name__ == "__main__":
    raise SystemExit(main())
