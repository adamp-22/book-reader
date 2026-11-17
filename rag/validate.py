"""Check an evaluation question file: every evidence quote must occur verbatim
(after normalisation) in the book text. Prints the page each quote was found on.

    uv run rag/rag.py --book <folder> validate        (checks <folder>/rag/questions/*.jsonl)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import book  # noqa: E402
from chunking import PAGE_RE, load_book, norm  # noqa: E402


def locate(book: dict[str, str]):
    """norm text per file + page lookup by normalised offset."""
    out = {}
    for f, text in book.items():
        pieces, pages, pos, page = [], [], 0, None
        for part in re.split(r"(⟦S\.\w+⟧)", text):
            m = PAGE_RE.fullmatch(part)
            if m:
                page = m.group(1)
                continue
            n = norm(part)
            if n:
                pages.append((pos, page))
                pieces.append(n)
                pos += len(n) + 1
        out[f] = (" ".join(pieces), pages)
    return out


def find(index, quote: str):
    q = norm(quote)
    hits = []
    for f, (text, pages) in index.items():
        start = text.find(q)
        while start >= 0:
            page = next((p for pos, p in reversed(pages) if pos <= start), None)
            hits.append((f, start, page))
            start = text.find(q, start + 1)
    return hits


def main(paths):
    index = locate(load_book(book.current().dir))
    bad = 0
    total = 0
    for path in paths:
        for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                q = json.loads(line)
                assert q["question"] and q["evidence"], "question/evidence missing"
            except Exception as e:  # noqa: BLE001
                print(f"{path}:{n}: invalid JSON line: {e}")
                bad += 1
                continue
            for ev in q["evidence"]:
                total += 1
                quote = ev["quote"]
                hits = find(index, quote)
                words = len(quote.split())
                if not hits:
                    print(f"{q['id']}: NOT FOUND: {quote!r}")
                    bad += 1
                elif words < 6:
                    print(f"{q['id']}: too short ({words} words): {quote!r}")
                    bad += 1
                elif len(hits) > 1:
                    print(f"{q['id']}: ambiguous ({len(hits)} hits, make it longer): {quote!r}")
                    bad += 1
                elif str(ev.get("page")) != str(hits[0][2]):
                    print(f"{q['id']}: page {ev.get('page')} given, found on {hits[0][2]}: {quote[:50]!r}")
    print(f"{total} quotes checked, {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
