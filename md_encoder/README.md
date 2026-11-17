# md_encoder

Turns a book PDF into structured Markdown: one file per chapter, an index, a single-file version, a reading view in HTML and a variant that imitates the printed page. It handles both kinds of PDF:

- **Digital PDFs** (with a text layer): the text and word positions are read directly with PyMuPDF. This is exact, fast and works on any OS.
- **Scans, photos, screen recordings** (images only): the pages go through OCR with Apple's on-device Vision framework. This is free and offline, but needs macOS.

```bash
uv run md_encoder/encode.py "My Book.pdf" -o mybook
uv run md_encoder/encode.py "My Book.pdf" -o mybook --skip-pages 0-3,12   # leave out cover, ads, imprint
```

`uv` installs the Python dependencies on the first run. The book's language (German or English) is detected from the text. It decides the labels the encoder writes, such as "Chapter"/"Kapitel", "p."/"S." and the README text.

## Output

```
mybook/
  book.json              title, language, page count, path to the PDF (read by rag/)
  README.md              table of contents
  00-<title>.md          front matter: everything before chapter 1
  01-<chapter>.md …      one file per chapter
  90-<bibliography>.md   back matter (bibliography, glossary, index) from 90 on
  full-book.md           everything in one file
  full-book.html         reading view with a chapter sidebar
  book-layout.md         print-like variant (VS Code Markdown preview ⇧⌘V, or Obsidian)
  assets/                figures and tables as PNG
  .cache/ocr/            text + word boxes per page (also used by the reader for highlights)
```

Book page numbers appear in the Markdown as invisible comments, `<!-- p. 42 -->` (or `<!-- S. 42 -->` in German). They are taken from the page numbers printed in the footer, so they match the printed book rather than the PDF page index.

## Pipeline

1. **Text with positions** (`ocr.py`):
   - Pages with a text layer: words and boxes come from PyMuPDF.
   - Image-only pages, pass 1: whole-page OCR at 250 dpi, used to learn the column layout.
   - Image-only pages, pass 2: the body column and the margin strip are OCR'd separately, each padded with white space. This stops OCR from merging margin notes into body lines or skipping lines.
2. **Layout analysis** (`layout.py`) recognizes:
   - chapters: "Chapter 3: …" / "Kapitel 3", a big "3 Title" heading, or chapter openers on coloured pages
   - numbered headings (`2.4.1 …` → `###`)
   - paragraphs, rejoined across line and page breaks, with hyphenation removed
   - bullet lists, including ones whose bullet glyph was lost
   - margin notes (`> **…**`) and tinted info boxes (`> [!NOTE]`)
   - figures and tables with captions ("Figure 1.2: …", "Abbildung 3.1: …", "Table …", "Exhibit …"), cropped as PNG into `assets/`. Tables are also rebuilt as Markdown tables where possible.
   - endnotes, bibliography and a multi-column index
3. **Output** (`render.py`, `html_view.py`, `book_view.py`) writes the files above.

Results are cached in `<out>/.cache/`, so re-running after a tweak takes seconds. A new run deletes only the files the previous run wrote (listed in `.cache/manifest.txt`).

## Options

| flag | purpose |
|---|---|
| `-o mybook` | output folder (default: `<pdf name>-md` next to the PDF) |
| `--skip-pages 0-4,13` | leave out cover, ads, imprint (0-based PDF page indices) |
| `--pages 15-60` | process only part of the PDF (for testing) |
| `--title "…"` | book title (default: the PDF name without "(…)") |
| `--lang de\|en` | output language for labels (default: detected) |
| `--ocr-langs fr-FR,en-US` | OCR languages for scanned pages (default: from `--lang`) |
| `--always-ocr` | OCR every page even if the PDF has a text layer |
| `--drop "regex"` | drop lines that match, e.g. licence stamps or watermarks |
| `--no-figures` | skip figure/table image crops |
| `--no-page-markers` | omit the page comments |
| `--no-book-view` | skip `book-layout.md` |
| `--cover-page N` | PDF page used as the cover in `book-layout.md` (-1: none) |
| `--force-ocr` | throw away the cache |
| `--dpi`, `--fig-dpi`, `--workers` | OCR resolution, image resolution, parallelism |

## Limits

- The layout rules were developed on textbooks: one body column, optional margin notes, captions below figures. Two-column layouts, novels without numbered headings and heavily designed magazines will convert, but with less structure. The thresholds live at the top of `layout.py`.
- Bold and italic inside running text are not detected. Footnote reference numbers in the text are removed, not linked.
- Formulas come out flattened. The table image above each rebuilt table is the reliable version.
- Scanned pages need macOS. On Linux and Windows only PDFs with a text layer work.
