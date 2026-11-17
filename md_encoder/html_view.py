"""Stage 4 – a comfortable reading view: full-book.md rendered as one HTML page
with a chapter sidebar, figures, info boxes, margin notes and page numbers."""
from __future__ import annotations

import html
import re
from pathlib import Path

import markdown

import lang

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1f;--muted:#6b6b70;--line:#e3e0d8;--accent:#c8102e;--box:#fdf0f0;--side:#f3f1ec}
@media (prefers-color-scheme:dark){:root{--bg:#18181a;--fg:#e8e6e1;--muted:#9a9890;--line:#33322f;--accent:#ff5a6e;--box:#2a1f21;--side:#202022}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:18px/1.65 Charter,"Iowan Old Style",Georgia,serif}
nav{position:fixed;top:0;left:0;bottom:0;width:300px;overflow-y:auto;padding:24px 18px;background:var(--side);border-right:1px solid var(--line);font:14px/1.45 -apple-system,system-ui,sans-serif}
nav .t{font-weight:700;font-size:15px;margin-bottom:12px}
nav ul{list-style:none;margin:0;padding-left:0}nav ul ul{padding-left:14px;margin:2px 0 6px}
nav a{color:var(--fg);text-decoration:none;display:block;padding:2px 0}nav a:hover{color:var(--accent)}
nav li li a{color:var(--muted)}
main{max-width:760px;margin:0 auto;padding:40px 28px 120px}
@media (min-width:1100px){main{margin-left:calc(300px + max(0px,(100vw - 300px - 760px)/2))}}
@media (max-width:1099px){nav{display:none}}
h1,h2,h3,h4{font-family:-apple-system,system-ui,sans-serif;line-height:1.25;scroll-margin-top:16px}
h1{font-size:2em;color:var(--accent);margin:2.2em 0 .6em}h2{font-size:1.45em;margin:1.8em 0 .5em}
h3{font-size:1.18em;margin:1.5em 0 .4em}h4{font-size:1.02em}
p{margin:.7em 0;hyphens:auto;text-align:justify}
img{max-width:100%;height:auto;display:block;margin:1.2em auto .4em;border-radius:4px;background:#fff}
blockquote{margin:1em 0;padding:0}
blockquote.margin{border-left:3px solid var(--accent);padding:2px 0 2px 12px;font:600 14px/1.4 -apple-system,system-ui,sans-serif;color:var(--accent)}
blockquote.margin p{margin:0;text-align:left}
blockquote.box{background:var(--box);border-radius:6px;padding:10px 18px;border-left:4px solid var(--accent)}
table{border-collapse:collapse;width:100%;font:14px/1.4 -apple-system,system-ui,sans-serif;margin:1em 0;display:block;overflow-x:auto}
th{background:var(--accent);color:#fff;text-align:left}th,td{border:1px solid var(--line);padding:6px 8px;vertical-align:top}
details{font:14px/1.4 -apple-system,system-ui,sans-serif;color:var(--muted);margin:.3em 0 1.2em}
.pg{font:12px -apple-system,system-ui,sans-serif;color:var(--muted);text-align:right;border-top:1px dashed var(--line);margin:1.6em 0 .4em;padding-top:2px}
hr{border:0;border-top:2px solid var(--line);margin:3em 0}
ul{padding-left:1.3em}li{margin:.25em 0}
"""


def build(md_path: Path, out_path: Path, title: str) -> Path:
    text = md_path.read_text(encoding="utf-8")
    # page markers become small visible labels
    text = re.sub(r"<!-- (S\.|p\.) (\w+) -->", r'<div class="pg" id="s\2">\1 \2</div>', text)
    md = markdown.Markdown(extensions=["tables", "toc", "sane_lists"], extension_configs={"toc": {"toc_depth": "1-2"}})
    body = md.convert(text)
    # > [!NOTE] callouts → info boxes, single-bold quotes → margin notes
    body = re.sub(r"<blockquote>\s*<p>\[!NOTE\]\s*", '<blockquote class="box"><p>', body)
    body = re.sub(r"<blockquote>\s*(<p><strong>[^<]*</strong></p>)\s*</blockquote>", r'<blockquote class="margin">\1</blockquote>', body)
    body = body.replace('<p></p>', "")
    toc = re.sub(rf'<li><a[^>]*>{re.escape(html.escape(title, quote=False))}</a></li>\s*', "", md.toc, count=1)  # book title
    page = f"""<!doctype html>
<html lang="{lang.code}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><nav><div class="t">{html.escape(title)}</div>{toc}</nav><main>{body}</main></body></html>"""
    out_path.write_text(page, encoding="utf-8")
    return out_path
