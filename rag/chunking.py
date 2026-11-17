"""Load the book's Markdown chapters and chunk them with LangChain.

1. Clean the Markdown produced by md_encoder:
   - drop callout syntax and image links
   - turn margin notes into a line glued to the paragraph they annotate
   - keep page markers as sentinels
2. MarkdownHeaderTextSplitter splits along the chapter/section hierarchy, so a
   chunk never straddles two sections and knows its breadcrumb.
3. RecursiveCharacterTextSplitter cuts long sections into overlapping chunks
   (paragraph → line → sentence → word boundaries).
4. Each chunk gets metadata: chapter, section path, book pages, figures, its
   character span in the cleaned text (used by the evaluation), and a
   contextual header (breadcrumb) prepended to the text that is indexed.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import book

HEADERS = [("#", "chapter"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
PAGE_RE = re.compile(r"⟦S\.(\w+)⟧")
FIG_RE = re.compile(r"⟦F:([^⟧]+)⟧")
SENTINEL_RE = re.compile(r"⟦[^⟧]*⟧")
NOTES_TITLES = ("literaturhinweise und anmerkungen", "anmerkungen", "literaturhinweise", "notes", "endnotes")
# chapter files only; bibliography (90-) and index (91-) are excluded
FILE_GLOB = "[0-8][0-9]-*.md"


@dataclass(frozen=True)
class ChunkConfig:
    size: int = 1400
    overlap: int = 210
    include_notes: bool = False  # endnote sections ("Notes", "Literaturhinweise und Anmerkungen")

    @property
    def key(self) -> str:
        return f"c{self.size}-o{self.overlap}{'-notes' if self.include_notes else ''}"


# ---------------------------------------------------------------- cleaning
def clean_markdown(md: str) -> str:
    out: list[str] = []
    after_note = False  # next bold line is the title of a callout box
    glue = False  # previous line was a margin note: attach the next text line directly
    for raw in md.splitlines():
        m = re.match(r"^<!-- (?:S\.|p\.) (\w+) -->$", raw)
        if m:
            out += [f"⟦S.{m.group(1)}⟧", ""]
            continue
        line = raw
        while line.startswith(">"):  # strip (nested) blockquote markers
            line = line[1:].lstrip(" ") if line.startswith("> ") or line == ">" else line[1:]
        if line.strip() == "[!NOTE]":
            after_note = True
            continue
        if not line.strip():
            if not glue:
                out.append("")
            continue
        m = re.match(r"^\*\*(.+?)\*\*\s*$", line)
        if m and not re.match(r"^(Abbildung|Abb\.|Tabelle|Tab\.|Figure|Fig\.|Table|Exhibit)\s", m.group(1)):
            if after_note:
                out += [m.group(1), ""]
            else:  # margin note
                out.append(f"» {m.group(1)}")
                glue = True
            after_note = False
            continue
        after_note = glue = False
        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)\s*$", line)
        if m:
            out += [f"⟦F:{m.group(1)}⟧", ""]
            continue
        line = re.sub(r"<details><summary>(.*?)</summary>", r"(\1:)", line)
        line = line.replace("</details>", "")
        line = line.replace("**", "")
        line = line.replace(r"\*", "*").replace(r"\_", "_").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        out.append(line)
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


_QUOTES = str.maketrans({c: '"' for c in "„“”‟«»‹›‚‘’'`´"} | {c: "-" for c in "‐‑‒–—―−"})


def norm(t: str) -> str:
    """Normal form used to match evidence quotes against chunks."""
    t = SENTINEL_RE.sub(" ", t)
    t = unicodedata.normalize("NFKC", t).translate(_QUOTES).replace("»", " ")
    t = re.sub(r"[|*_#>]", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def breadcrumb(meta: dict) -> str:
    return " › ".join(meta[k] for k in ("chapter", "h2", "h3", "h4") if meta.get(k))


def chapter_of(meta: dict) -> str:
    m = re.match(r"(?:Kapitel|Chapter) (\d+)", meta.get("chapter", ""))
    return m.group(1) if m else "0"


# ---------------------------------------------------------------- chunking
def load_book(md_dir: Path) -> dict[str, str]:
    """file name → cleaned text."""
    return {p.name: clean_markdown(p.read_text(encoding="utf-8")) for p in sorted(md_dir.glob(FILE_GLOB))}


def chunk_book(md_dir: Path, cfg: ChunkConfig = ChunkConfig()) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(HEADERS, strip_headers=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.size,
        chunk_overlap=cfg.overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator="end",
        add_start_index=True,
    )
    docs: list[Document] = []
    for fname, text in load_book(md_dir).items():
        normed_file = norm(text)
        cursor = 0
        page = None
        for sec in header_splitter.split_text(text):
            # MarkdownHeaderTextSplitter joins blocks with "  \n"; restore paragraph breaks
            content = sec.page_content.replace("  \n", "\n\n")
            meta = dict(sec.metadata)
            sec_page = page
            markers = [(m.start(), m.group(1)) for m in PAGE_RE.finditer(content)]
            if markers:
                page = markers[-1][1]
            title = (meta.get("h2") or "").lower()
            if not cfg.include_notes and any(title.startswith(t) for t in NOTES_TITLES):
                continue
            carry = ""  # tiny pieces (e.g. a margin note split off a long paragraph) join the next chunk
            for piece in splitter.create_documents([content]) if content.strip() else []:
                part, start = piece.page_content, piece.metadata["start_index"]
                before = [p for pos, p in markers if pos <= start]
                # a marker counts only if text of the chunk follows it (not one that just precedes the next heading)
                inside = [p for pos, p in markers if start < pos < start + len(part)
                          and SENTINEL_RE.sub("", part[pos - start:]).strip()]
                first = before[-1] if before else sec_page
                if PAGE_RE.match(part.lstrip()) and inside:  # chunk starts with a marker
                    first = inside[0]
                pages = [p for p in [first, *inside] if p]
                pages = list(dict.fromkeys(pages))
                figs = FIG_RE.findall(part)
                body = re.sub(r"\n{3,}", "\n\n", SENTINEL_RE.sub("", part)).strip()
                if len(norm(body)) < 40:
                    carry = f"{carry}\n{body}".strip() if norm(body) else carry
                    continue
                if carry:
                    body, carry = f"{carry}\n{body}", ""
                nb = norm(body)
                pos = normed_file.find(nb, max(0, cursor - cfg.overlap * 2))
                if pos < 0:
                    pos = normed_file.find(nb)
                if pos >= 0:
                    cursor = pos
                crumb = breadcrumb(meta)
                docs.append(Document(
                    page_content=f"{crumb}\n\n{body}",
                    metadata={
                        "id": f"{fname[:2]}-{len(docs):04d}",
                        "file": fname,
                        "chapter": chapter_of(meta),
                        "section": crumb,
                        "pages": pages,
                        "figures": figs,
                        "body": body,
                        "span": [pos, pos + len(nb)] if pos >= 0 else None,
                    },
                ))
    return docs


def page_label(pages: list[str]) -> str:
    return book.current().page_label(pages)


if __name__ == "__main__":  # quick statistics
    import statistics
    import sys

    root = book.current().dir
    cfg = ChunkConfig(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
    chunks = chunk_book(root, cfg)
    lens = [len(d.metadata["body"]) for d in chunks]
    print(f"{len(chunks)} chunks, body chars: median {statistics.median(lens):.0f}, max {max(lens)}, min {min(lens)}")
    print("unmatched spans:", sum(d.metadata["span"] is None for d in chunks), " no page:", sum(not d.metadata["pages"] for d in chunks))
    for d in chunks[40:43]:
        print("—" * 60, d.metadata["id"], page_label(d.metadata["pages"]), d.metadata["figures"])
        print(d.page_content[:600])
