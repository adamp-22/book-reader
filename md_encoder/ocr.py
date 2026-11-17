"""Stage 1 – text with positions, from the PDF's text layer or by OCR.

Pages that carry a text layer (normal digital PDFs) are read directly with PyMuPDF:
exact, fast and platform-independent. Pages without one (scans, screen recordings)
are rendered to a bitmap and run through Apple's on-device Vision text recognizer
(macOS; free, offline). Either way every line is stored with per-word boxes, so the
layout stage can split lines that accidentally merge a margin note with the body text.

Results are cached as one JSON file per page, so re-running the layout stage
(or the whole pipeline) never re-reads or re-OCRs a page.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pymupdf

DEFAULT_LANGS = ["de-DE", "en-US"]
MIN_TEXT_CHARS = 40  # fewer characters in the text layer: treat the page as a scan
LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})


def has_text_layer(page) -> bool:
    return len(page.get_text("text").strip()) >= MIN_TEXT_CHARS


def _text_layer_lines(page) -> list[dict]:
    """Lines with word boxes from the PDF's own text layer, in the same format as OCR."""
    W, H = page.rect.width, page.rect.height

    def norm_box(x0, y0, x1, y1):
        return [round(min(max(x0 / W, 0.0), 1.0), 5), round(min(max(y0 / H, 0.0), 1.0), 5),
                round(min(max(x1 / W, 0.0), 1.0), 5), round(min(max(y1 / H, 0.0), 1.0), 5)]

    grouped: dict[tuple, list] = {}
    for x0, y0, x1, y1, word, block, line, _ in page.get_text("words"):
        grouped.setdefault((block, line), []).append((x0, y0, x1, y1, word.translate(LIGATURES)))
    lines = []
    for ws in grouped.values():
        box = norm_box(min(w[0] for w in ws), min(w[1] for w in ws), max(w[2] for w in ws), max(w[3] for w in ws))
        lines.append({"text": " ".join(w[4] for w in ws), "conf": 1.0, "box": box,
                      "words": [{"t": w[4], "box": norm_box(*w[:4])} for w in ws]})
    return lines


def _vision_ocr(png: bytes, crop=(0.0, 0.0, 1.0, 1.0), langs=None) -> list[dict]:
    """OCR one bitmap; boxes are mapped from the crop back to full-page coordinates."""
    import Vision
    from Foundation import NSData

    data = NSData.dataWithBytes_length_(png, len(png))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(0)  # 0 = accurate
    req.setRecognitionLanguages_(langs or DEFAULT_LANGS)
    req.setUsesLanguageCorrection_(True)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(f"Vision OCR failed: {err}")

    cx0, cy0, cx1, cy1 = crop
    cw, ch = cx1 - cx0, cy1 - cy0

    def to_page(b):
        return [round(cx0 + b[0] * cw, 5), round(cy0 + b[1] * ch, 5), round(cx0 + b[2] * cw, 5), round(cy0 + b[3] * ch, 5)]

    lines = []
    for obs in req.results() or []:
        cand = obs.topCandidates_(1)[0]
        text = str(cand.string())
        words = _word_boxes(cand, text)
        for w in words:
            if w["box"] is not None:
                w["box"] = to_page(w["box"])
        lines.append({"text": text, "conf": float(cand.confidence()), "box": to_page(_box(obs.boundingBox())), "words": words})
    return lines


def _box(bb) -> list[float]:
    """Vision boxes are normalised with origin bottom-left -> convert to top-left x0,y0,x1,y1."""
    x0 = bb.origin.x
    y1 = 1.0 - bb.origin.y
    return [round(x0, 5), round(y1 - bb.size.height, 5), round(x0 + bb.size.width, 5), round(y1, 5)]


def _word_boxes(cand, text: str) -> list[dict]:
    from Foundation import NSMakeRange

    words, pos = [], 0
    for token in text.split(" "):
        if token:
            try:
                rect, _ = cand.boundingBoxForRange_error_(NSMakeRange(pos, len(token)), None)
                box = _box(rect.boundingBox()) if rect is not None else None
            except Exception:
                box = None
            words.append({"t": token, "box": box})
        pos += len(token) + 1
    return words


def _ocr_one(pdf_path: str, index: int, dpi: int, cache_file: str, crops, langs=None, text_layer=True) -> int:
    doc = pymupdf.open(pdf_path)
    page = doc[index]
    if text_layer and not crops and has_text_layer(page):
        _write(cache_file, {"page_index": index, "source": "text", "lines": _text_layer_lines(page)})
        return index
    if sys.platform != "darwin":
        raise RuntimeError(f"page {index} has no text layer: OCR needs Apple's Vision framework (macOS)")
    W, H = page.rect.width, page.rect.height
    lines = []
    for crop in crops or [(0.0, 0.0, 1.0, 1.0)]:
        x0, y0, x1, y1 = crop
        clip = pymupdf.Rect(x0 * W, y0 * H, x1 * W, y1 * H)
        pix = page.get_pixmap(dpi=dpi, clip=clip)
        if crops:
            # white border around the crop: Vision drops characters that touch the edge
            pad = 40
            big = pymupdf.Pixmap(pix.colorspace, pymupdf.IRect(pix.x - pad, pix.y - pad, pix.x + pix.width + pad, pix.y + pix.height + pad), False)
            big.clear_with(255)
            big.copy(pix, pix.irect)
            px, py = pad / pix.width * (x1 - x0), pad / pix.height * (y1 - y0)
            lines.extend(_vision_ocr(big.tobytes("png"), (x0 - px, y0 - py, x1 + px, y1 + py), langs))
        else:
            lines.extend(_vision_ocr(pix.tobytes("png"), crop, langs))
    _write(cache_file, {"page_index": index, "source": "ocr", "dpi": dpi, "crops": crops, "lines": lines})
    return index


def _write(cache_file: str, data: dict) -> None:
    tmp = cache_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    os.replace(tmp, cache_file)


def is_ocr_page(cache_dir: Path, index: int) -> bool:
    """True if pass 1 of this page came from OCR (old caches without "source" are OCR)."""
    with open(cache_path(cache_dir, index), encoding="utf-8") as fh:
        return json.load(fh).get("source", "ocr") == "ocr"


def cache_path(cache_dir: Path, index: int, columns: bool = False) -> Path:
    return cache_dir / (f"page_{index:04d}.cols.json" if columns else f"page_{index:04d}.json")


def run_ocr(pdf_path: Path, cache_dir: Path, pages: list[int], dpi: int, workers: int,
            crops: dict | None = None, log=print, langs=None, text_layer: bool = True) -> None:
    """Pass 1 (crops=None): whole pages – text layer where there is one, OCR otherwise.
    Pass 2: {page: [crop rects]} – column-wise OCR of the OCR'd pages, which stops Vision
    from merging margin notes into body lines or skipping lines."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cols = crops is not None
    todo = [i for i in pages if (not cols or i in crops) and not cache_path(cache_dir, i, cols).exists()]
    if cols:  # text-layer pages are exact already
        todo = [i for i in todo if is_ocr_page(cache_dir, i)]
    label = "OCR pass 2 (column-wise)" if cols else "Pass 1 (text layer or OCR, full pages)"
    if not todo:
        log(f"{label}: cached")
        return
    log(f"{label}: {len(todo)} page(s), {workers} worker(s)")
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [
            pool.submit(_ocr_one, str(pdf_path), i, dpi, str(cache_path(cache_dir, i, cols)),
                        crops.get(i) if cols else None, langs, text_layer)
            for i in todo
        ]
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 25 == 0 or done == len(todo):
                log(f"  {done}/{len(todo)}")


def load_page(cache_dir: Path, index: int, prefer_columns: bool = False) -> dict:
    p = cache_path(cache_dir, index, True)
    if not (prefer_columns and p.exists()):
        p = cache_path(cache_dir, index)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)
