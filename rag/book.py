"""The book a command works on: a folder written by md_encoder.

Everything book-specific lives in that folder, nothing in the code:

    <book>/book.json                 title, language, PDF path (written by the encoder);
                                     optional "subject" for the embedding instruction
    <book>/NN-*.md                   chapters
    <book>/.cache/ocr/               word boxes from OCR / the PDF text layer (reader highlights)
    <book>/.cache/rag/               embedding cache, rendered page images
    <book>/rag/best_config.json      tuned settings, written by `rag.py eval`
    <book>/rag/questions/*.jsonl     your evaluation questions
    <book>/rag/report.md, misses.md  evaluation results

The folder is chosen with --book, else $BOOK_DIR, else ./markdown, else the current folder.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

PAGE_PREFIX = {"de": "S."}  # everything else: "p."


@dataclass
class Book:
    dir: Path
    title: str
    language: str = "en"
    pdf: Path | None = None
    subject: str = ""  # "a German textbook on strategic management" – used in model instructions

    @property
    def cache(self) -> Path:
        return self.dir / ".cache" / "rag"

    @property
    def ocr_dir(self) -> Path:
        return self.dir / ".cache" / "ocr"

    @property
    def data(self) -> Path:
        return self.dir / "rag"

    @property
    def questions_dir(self) -> Path:
        return self.data / "questions"

    @property
    def about(self) -> str:
        return self.subject or f'the book "{self.title}"'

    @property
    def query_task(self) -> str:
        return f"Given a question about {self.about}, retrieve the passages that answer it"

    @property
    def rerank_task(self) -> str:
        return f"Given a question about {self.about}, judge whether the passage contains information needed to answer it"

    def page_label(self, pages: list[str]) -> str:
        p = PAGE_PREFIX.get(self.language, "p.")
        if not pages:
            return f"{p} ?"
        return f"{p} {pages[0]}" if len(pages) == 1 else f"{p} {pages[0]}–{pages[-1]}"


def load(path: str | Path) -> Book:
    d = Path(path).expanduser().resolve()
    if not d.is_dir() or not (any(d.glob("[0-9][0-9]-*.md")) or (d / "book.json").exists()):
        raise SystemExit(f"{d} is not a book folder (expected the output of md_encoder: NN-*.md chapters + book.json)")
    info = json.loads((d / "book.json").read_text(encoding="utf-8")) if (d / "book.json").exists() else {}
    title = info.get("title") or _readme_title(d) or d.name
    language = info.get("language") or _guess_language(d)
    pdf = (d / info["pdf"]).resolve() if info.get("pdf") else None
    return Book(d, title, language, pdf, info.get("subject", ""))


def find(arg: str | Path | None = None) -> Book:
    if arg:
        return load(arg)
    if os.environ.get("BOOK_DIR"):
        return load(os.environ["BOOK_DIR"])
    for cand in (Path("markdown"), Path(".")):
        if cand.is_dir() and ((cand / "book.json").exists() or any(cand.glob("[0-9][0-9]-*.md"))):
            return load(cand)
    raise SystemExit("No book folder found. Pass --book <folder written by md_encoder> or set BOOK_DIR.")


_current: Book | None = None


def use(book: Book) -> Book:
    global _current
    _current = book
    return book


def current() -> Book:
    return _current if _current is not None else use(find())


def _readme_title(d: Path) -> str:
    readme = d / "README.md"
    if readme.exists():
        m = re.match(r"#\s+(.+)", readme.read_text(encoding="utf-8"))
        if m:
            return m.group(1).strip()
    return ""


def _guess_language(d: Path) -> str:
    sample = "".join(p.read_text(encoding="utf-8")[:20000] for p in sorted(d.glob("[0-9][0-9]-*.md"))[:3])
    return "de" if "<!-- S. " in sample or len(re.findall(r"\b(der|die|und)\b", sample)) > len(re.findall(r"\b(the|and)\b", sample)) else "en"
