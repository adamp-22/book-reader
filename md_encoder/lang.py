"""Language settings.

What the encoder *recognises* (caption words, back-matter and notes headings, …) covers
German and English at the same time, see layout.py. What it *writes* (chapter labels,
page markers, README text) follows the book's language, which is detected from the text
unless --lang is given. Other languages get the English labels; their OCR languages can be
set with --ocr-langs.
"""
from __future__ import annotations

import re
from collections import Counter

LABELS = {
    "de": {
        "chapter": "Kapitel",
        "page": "S.",
        "fig_text": "Text in der Abbildung",
        "table_text": "Text in der Tabelle",
        "front": "Vorspann",
        "generated": "_Automatisch aus `{source}` erzeugt am {date} ({how})._",
        "how_ocr": "OCR + Layout-Analyse",
        "how_text": "Textebene des PDFs + Layout-Analyse",
        "page_note": "Seitenzahlen der Buchvorlage stehen als unsichtbare Kommentare `<!-- S. 42 -->` im Text.",
        "contents": "Inhalt",
        "one_file": "Alles in einer Datei: [full-book.md](full-book.md)",
        "and": "und",
        "layout_file": "buch-layout.md",
        "fig_prefix": "abb",
        "layout_note": "Buch-Layout. Zum Lesen in der Markdown-Vorschau öffnen (VS Code: ⇧⌘V, oder Obsidian).",
        "toc": "Inhaltsverzeichnis",
        "ocr": ["de-DE", "en-US"],
    },
    "en": {
        "chapter": "Chapter",
        "page": "p.",
        "fig_text": "Text in the figure",
        "table_text": "Text in the table",
        "front": "Front matter",
        "generated": "_Generated automatically from `{source}` on {date} ({how})._",
        "how_ocr": "OCR + layout analysis",
        "how_text": "PDF text layer + layout analysis",
        "page_note": "Page numbers of the printed book are kept as invisible comments `<!-- p. 42 -->`.",
        "contents": "Contents",
        "one_file": "Everything in one file: [full-book.md](full-book.md)",
        "and": "and",
        "layout_file": "book-layout.md",
        "fig_prefix": "fig",
        "layout_note": "book layout. Open it in a Markdown preview to read it (VS Code: ⇧⌘V, or Obsidian).",
        "toc": "Contents",
        "ocr": ["en-US"],
    },
}

_STOPWORDS = {
    "de": {"der", "die", "das", "und", "ist", "nicht", "mit", "von", "den", "sich", "auch", "werden", "eine", "für", "auf", "dem", "wird"},
    "en": {"the", "and", "is", "of", "to", "in", "that", "with", "for", "are", "this", "be", "as", "on", "by", "which", "it"},
}

code = "de"
L = LABELS["de"]


def set_language(lang: str) -> None:
    """Select the output labels (unknown languages fall back to English labels)."""
    global code, L
    code = lang
    L = LABELS.get(lang, LABELS["en"])


def detect(texts) -> str:
    """'de' or 'en', whichever language's frequent words dominate the text."""
    counts = Counter()
    for t in texts:
        for w in re.findall(r"[a-zäöüß]+", t.lower()):
            for lang, words in _STOPWORDS.items():
                if w in words:
                    counts[lang] += 1
    return max(_STOPWORDS, key=lambda k: counts[k]) if counts else "en"
