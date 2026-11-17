# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "langchain-core>=1.0",
#   "langchain-text-splitters>=1.0",
#   "langchain-community>=0.4",
#   "langchain-classic>=1.0",
#   "langchain-huggingface>=1.0",
#   "langchain-openai>=1.0",
#   "sentence-transformers>=5.0",
#   "faiss-cpu",
#   "rank-bm25",
#   "snowballstemmer",
#   "pymupdf",
# ]
# ///
"""Book reader with question answering: the original PDF pages in a PDF-viewer UI;
⌘F / Ctrl+F asks a question and highlights the relevant passages on the pages.

    uv run rag/reader.py --book mybook        # opens http://127.0.0.1:8765 in the browser
    uv run rag/reader.py --book mybook --port 9000 --no-browser

--book is a folder written by md_encoder (default: $BOOK_DIR, ./markdown or .); the PDF
comes from its book.json (or --pdf). Everything runs locally. Highlights are placed with
the word boxes recorded by the encoder (<book>/.cache/ocr), so they sit exactly on the page.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import threading
import time
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import book  # noqa: E402
import rag  # noqa: E402  (sets quiet logging/env defaults)
from chunking import chunk_book  # noqa: E402
from retrieval import PRECISE, PipelineConfig, build_retriever  # noqa: E402

DPI = 160
FOOTER_Y = 0.935  # same as md_encoder/layout.py: page numbers sit below this line


# ---------------------------------------------------------------- OCR word index per page
def _compact(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


class PageWords:
    """All OCR words of one page as one compact string, with a map from characters back to words."""

    def __init__(self, lines: list[dict]):
        self.lines = [line["text"] for line in lines]
        self.words: list[tuple[str, list[float], int]] = []  # compact text, box, line id
        for li, line in enumerate(lines):
            for w in line.get("words") or [{"t": line["text"], "box": line["box"]}]:
                c = _compact(w["t"])
                if c:
                    self.words.append((c, w["box"], li))
        self.text = "".join(w[0] for w in self.words)
        self.owner = [i for i, w in enumerate(self.words) for _ in w[0]]
        self.starts = list(itertools.accumulate((len(w[0]) for w in self.words), initial=0))
        self.line_size = Counter(w[2] for w in self.words)


def load_pages(ocr_dir: Path, n_pages: int) -> dict[int, PageWords]:
    pages = {}
    for i in range(n_pages):
        for name in (f"page_{i:04d}.cols.json", f"page_{i:04d}.json"):  # column OCR is cleaner
            p = ocr_dir / name
            if p.exists():
                pages[i] = PageWords(json.loads(p.read_text(encoding="utf-8"))["lines"])
                break
    return pages


def page_index_map(ocr_dir: Path, n_pages: int) -> dict[str, int]:
    """Printed page number → PDF page index, via the footer numbers – the same rule the encoder
    uses for the page labels in the Markdown (no numbers found: page 1 = first PDF page)."""
    offsets = Counter()
    for i in range(n_pages):
        p = ocr_dir / f"page_{i:04d}.json"
        if not p.exists():
            continue
        for line in json.loads(p.read_text(encoding="utf-8"))["lines"]:
            t = line["text"].strip()
            if line["box"][1] > FOOTER_Y and re.fullmatch(r"\d{1,4}", t):
                offsets[int(t) - i] += 1
    off = offsets.most_common(1)[0][0] if offsets else 1
    return {str(i + off): i for i in range(n_pages)}


def _segments(text: str) -> list[tuple[list[str], list[int]] | str]:
    """Search units of a passage: runs of words (3-word windows + token lengths) or short cells.

    Runs do not cross table cells or figure labels (`|`, `·`, line breaks), whose reading
    order differs between Markdown and OCR; a segment of at most 3 words is one cell."""
    out = []
    for seg in re.split(r"[|·\n]", text):
        toks = [c for c in (_compact(t) for t in seg.split()) if c]
        if len(toks) >= 4:
            out.append((["".join(toks[i:i + 3]) for i in range(len(toks) - 2)], [len(t) for t in toks]))
        elif toks and len("".join(toks)) >= 8 and not "".join(toks).isdigit():
            out.append("".join(toks))
    return out


def matched_words(text: str, page: PageWords, anchored: bool = False) -> tuple[list[bool], int]:
    """Which OCR words of the page belong to `text`, and how many of them were matched in chains.

    Every 3-word window of the passage is searched in the page's OCR text (compacted, so
    hyphenation and line breaks do not matter). A window only counts when the next or the
    previous window continues it seamlessly on the page: isolated hits of common phrases
    are coincidences. Short cells (figure labels, table cells) and long unchained windows are
    only accepted on a page where the passage is already anchored by such chains."""
    t = page.text
    covered = bytearray(len(t))
    cells, loose = [], []
    last = 0
    for seg in _segments(text):
        if isinstance(seg, str):
            cells.append(seg)
            continue
        wins, lens = seg
        prev = None  # (window index, position, string, part of a chain) of the last match
        for i, w in enumerate(wins):
            pos, linked = -1, False
            if prev and prev[0] == i - 1 and t.startswith(w, prev[1] + lens[i - 1]):
                pos, linked = prev[1] + lens[i - 1], True
                for a, b in ((prev[1], len(prev[2])), (pos, len(w))):
                    covered[a:a + b] = b"\x01" * b
                if not prev[3]:  # the chain starts at prev: extend it back over short windows ("In the …")
                    j, q = prev[0] - 1, prev[1]
                    while j >= 0 and t.startswith(wins[j], q - lens[j]):
                        q -= lens[j]
                        covered[q:q + len(wins[j])] = b"\x01" * len(wins[j])
                        j -= 1
            elif len(w) >= 12:  # a chain can only start with a distinctive window
                pos = t.find(w, last)
                if pos < 0:
                    pos = t.find(w)
            if pos >= 0:
                if not linked and len(w) >= 18:
                    loose.append((pos, w))
                prev, last = (i, pos, w, linked), pos
    chained = len(_hit_words(covered, page))
    if chained >= 3 or anchored:
        for pos, w in loose:
            covered[pos:pos + len(w)] = b"\x01" * len(w)
        for c in cells:
            pos = t.find(c)
            if pos >= 0:
                covered[pos:pos + len(c)] = b"\x01" * len(c)
    hit = [False] * len(page.words)
    for w in _hit_words(covered, page):
        hit[w] = True
    # bridge short gaps inside a line (small words that no chain covered)
    for i in range(len(hit)):
        if hit[i]:
            continue
        line = page.words[i][2]
        prev_w = next((j for j in range(i - 1, max(-1, i - 4), -1) if hit[j]), None)
        next_w = next((j for j in range(i + 1, min(len(hit), i + 4)) if hit[j]), None)
        if prev_w is not None and next_w is not None and page.words[prev_w][2] == line == page.words[next_w][2]:
            hit[i] = True
    return hit, chained


def _hit_words(covered: bytearray, page: PageWords):
    """Words with at least half of their characters covered."""
    counts = Counter(page.owner[i] for i, c in enumerate(covered) if c)
    return [w for w, n in counts.items() if n >= 0.5 * len(page.words[w][0])]


def locate(text: str, listed: list[int], pages: dict[int, PageWords], reach: int = 2) -> dict[int, list[bool]]:
    """Matched words of `text` per page.

    Searches the pages the chunk is labelled with plus `reach` pages either side: when a
    paragraph runs over a page break the Markdown keeps it in one piece, so the page label
    of a chunk that starts in its second half can be one page early."""
    found = {}
    labels_only = all(isinstance(seg, str) for seg in _segments(text))  # e.g. only margin notes
    for i in range(max(0, min(listed) - reach), max(listed) + reach + 1):
        if i not in pages:
            continue
        hit, chained = matched_words(text, pages[i], anchored=labels_only and i in listed)
        if (chained >= 3 and (i in listed or chained >= 6)) or (labels_only and i in listed and any(hit)):
            found[i] = hit
    return found


def rects_from_hits(hit: list[bool], page: PageWords) -> list[list[float]]:
    """Rectangles (normalised x0, y0, x1, y1) for the matched words, one per run on a line."""
    # one rectangle per run of highlighted words on the same line
    runs: list[tuple[int, list[int]]] = []
    for i, (_, _, line) in enumerate(page.words):
        if not hit[i]:
            continue
        if runs and runs[-1][0] == line and runs[-1][1][-1] == i - 1:
            runs[-1][1].append(i)
        else:
            runs.append((line, [i]))
    lines_hit = {line for line, _ in runs}
    rects = []
    for line, idx in runs:
        lonely = len(idx) < 2 and page.line_size[line] > 2 and not ({line - 1, line + 1} & lines_hit)
        if lonely:  # a single matched word in an otherwise unmatched line: probably a coincidence
            continue
        boxes = [page.words[i][1] for i in idx]
        rects.append([round(min(b[0] for b in boxes) - 0.003, 4), round(min(b[1] for b in boxes) - 0.0008, 4),
                      round(max(b[2] for b in boxes) + 0.003, 4), round(max(b[3] for b in boxes) + 0.0008, 4)])
    return rects


def span_rects(page: PageWords, a: int, b: int) -> list[list[float]]:
    """Rectangles for characters a..b of the page's compact text (an exact search hit).
    Partial words are cut proportionally, so "model" inside "models" is marked precisely."""
    runs: list[list] = []  # [box, line]
    for w in range(page.owner[a], page.owner[b - 1] + 1):
        word, (x0, y0, x1, y1), line = page.words[w]
        start, n = page.starts[w], len(word)
        ca, cb = max(a, start) - start, min(b, start + n) - start
        box = [x0 + (x1 - x0) * ca / n, y0, x0 + (x1 - x0) * cb / n, y1]
        if runs and runs[-1][1] == line:
            r = runs[-1][0]
            runs[-1][0] = [min(r[0], box[0]), min(r[1], box[1]), max(r[2], box[2]), max(r[3], box[3])]
        else:
            runs.append([box, line])
    return [[round(r[0] - 0.002, 4), round(r[1] - 0.0008, 4), round(r[2] + 0.002, 4), round(r[3] + 0.0008, 4)]
            for r, _ in runs]


# ---------------------------------------------------------------- the app
class Reader:
    def __init__(self, bk: book.Book, pdf: Path):
        import pymupdf

        self.book = bk
        self.doc = pymupdf.open(pdf)
        self.render_lock = threading.Lock()
        self.ask_lock = threading.Lock()
        self.n = self.doc.page_count
        self.sizes = [(p.rect.width, p.rect.height) for p in self.doc]
        self.label_to_index = page_index_map(bk.ocr_dir, self.n)
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}
        self.page_words = load_pages(bk.ocr_dir, self.n)
        if not self.page_words:
            print(f"note: no word boxes in {bk.ocr_dir} – search works, but nothing can be highlighted", flush=True)
        self.cfg, self.chunk_cfg = rag.load_best()
        self.chunks = chunk_book(bk.dir, self.chunk_cfg)
        self.by_id = {d.metadata["id"]: d for d in self.chunks}
        print(f"{bk.title}: loading search index …", flush=True)
        self.retriever = build_retriever(self.chunks, self.cfg)
        self.retriever.invoke("warm-up")  # the first query on the GPU is slow; do it before the user asks
        self.precise = None
        self._located: dict[str, dict[int, list[bool]]] = {}
        self.title = bk.title

    # -- pages
    def page_png(self, i: int) -> bytes:
        path = self.book.cache / "pages" / f"{i:04d}_{DPI}.jpg"
        if not path.exists():
            with self.render_lock:
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pix = self.doc[i].get_pixmap(dpi=DPI)
                    path.write_bytes(pix.tobytes("jpeg", jpg_quality=88))
        return path.read_bytes()

    def meta(self) -> dict:
        toc, seen = [], set()
        for d in self.chunks:
            parts = d.metadata["section"].split(" › ")
            for level, title in enumerate(parts[:3]):
                key = tuple(parts[: level + 1])
                if key in seen or not d.metadata["pages"]:
                    continue
                seen.add(key)
                toc.append({"title": title, "level": level, "page": self.label_to_index.get(d.metadata["pages"][0], 0)})
        return {"title": self.title, "language": self.book.language, "pages": [{"w": w, "h": h, "label": self.index_to_label.get(i, str(i))}
                                               for i, (w, h) in enumerate(self.sizes)], "toc": toc}

    # -- questions
    def locate(self, cid: str) -> dict[int, list[bool]]:
        if cid not in self._located:
            meta = self.by_id[cid].metadata
            listed = [self.label_to_index[x] for x in meta["pages"] if x in self.label_to_index]
            self._located[cid] = locate(meta["body"], listed, self.page_words) if listed else {}
        return self._located[cid]

    def find(self, query: str, limit: int = 500) -> dict:
        """Exact search (case, spaces, hyphenation and punctuation ignored), like Ctrl+F."""
        t = time.perf_counter()
        needle = _compact(query)
        hits, total = [], 0
        if len(needle) >= 2:
            for i in sorted(self.page_words):
                pw = self.page_words[i]
                pos = pw.text.find(needle)
                while pos >= 0:
                    total += 1
                    if len(hits) < limit:
                        hits.append({"page": i, "rects": span_rects(pw, pos, pos + len(needle)),
                                     "line": pw.lines[pw.words[pw.owner[pos]][2]]})
                    pos = pw.text.find(needle, pos + len(needle))
        return {"query": query, "hits": hits, "total": total, "ms": round(1000 * (time.perf_counter() - t))}

    def ask(self, question: str, precise: bool) -> dict:
        t = time.perf_counter()
        with self.ask_lock:
            if precise and self.precise is None:
                cfg = PipelineConfig(**{**self.cfg.__dict__, **PRECISE})
                self.precise = build_retriever(self.chunks, cfg)
            docs = (self.precise if precise else self.retriever).invoke(question)
        passages = rag.merge_passages(docs)
        out = []
        for rank, p in enumerate(passages, 1):
            merged: dict[int, list[bool]] = {}  # page -> matched words of all chunks of the passage
            for cid in p["chunks"]:
                for i, hit in self.locate(cid).items():
                    merged[i] = [a or b for a, b in zip(merged[i], hit)] if i in merged else list(hit)
            hls = [{"page": i, "rects": r} for i in sorted(merged) if (r := rects_from_hits(merged[i], self.page_words[i]))]
            labels = [self.index_to_label.get(h["page"], str(h["page"])) for h in hls]
            pages = self.book.page_label(list(dict.fromkeys(labels))) if labels else p["pages"]
            out.append({
                "rank": rank,
                "score": round(p["score"], 3),
                "section": p["section"].split(" › ")[-1],
                "crumbs": " › ".join(p["section"].split(" › ")[:-1]),
                "pages": pages,
                "text": p["text"],
                "highlights": hls,
            })
        return {"question": question, "precise": precise, "ms": round(1000 * (time.perf_counter() - t)),
                "passages": out}


def make_handler(reader: Reader):
    html_path = Path(__file__).resolve().parent / "reader.html"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, body: bytes, ctype: str, cache: bool = False, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: int = 200):
            self._send(json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8", status=status)

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path in ("/", "/index.html"):
                    self._send(html_path.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/meta":
                    self._json(reader.meta())
                elif url.path == "/api/ask":
                    qs = parse_qs(url.query)
                    q = (qs.get("q") or [""])[0].strip()
                    if not q:
                        return self._json({"error": "empty question"}, 400)
                    self._json(reader.ask(q, (qs.get("precise") or ["0"])[0] == "1"))
                elif url.path == "/api/find":
                    q = (parse_qs(url.query).get("q") or [""])[0].strip()
                    self._json(reader.find(q))
                elif m := re.fullmatch(r"/page/(\d+)\.jpg", url.path):
                    i = int(m.group(1))
                    if not 0 <= i < reader.n:
                        return self._send(b"", "text/plain", status=404)
                    self._send(reader.page_png(i), "image/jpeg", cache=True)
                else:
                    self._send(b"not found", "text/plain", status=404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", help="book folder written by md_encoder (default: $BOOK_DIR, ./markdown or .)")
    ap.add_argument("--pdf", type=Path, default=None, help="the book's PDF (default: from book.json)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    bk = book.use(book.find(args.book))
    pdf = args.pdf or bk.pdf
    if not pdf or not Path(pdf).exists():
        sys.exit(f"PDF not found ({pdf or 'book.json names no PDF'}) – pass --pdf")
    reader = Reader(bk, Path(pdf))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(reader))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Reader running: {url}   (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
