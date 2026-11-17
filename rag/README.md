# rag: ask the book

Ask a question and get back every passage of the book that is relevant to it, with chapter, section and page. Questions can be in another language than the book: the embedding model is multilingual. Everything runs locally and the book text is never sent anywhere.

The input is a book folder written by [md_encoder](../md_encoder/README.md).

```bash
uv run rag/rag.py --book mybook ask "What are the main stages of …?"
uv run rag/rag.py --book mybook ask                  # interactive: one question after another (fastest)
uv run rag/rag.py --book mybook ask --json "…"       # machine-readable
uv run rag/rag.py --book mybook ask --precise "…"    # + reranker: better recall, a few seconds of GPU per question
uv run rag/rag.py --book mybook ask --fixed 10 "…"   # exactly 10 chunks instead of the adaptive cut-off
```

Instead of `--book`, you can set `BOOK_DIR=mybook`, or run from a folder that contains `markdown/` or is itself the book folder.

## Reader: the book as a PDF, with ⌘F for questions

```bash
uv run rag/reader.py --book mybook        # opens http://127.0.0.1:8765
```

The original pages of the PDF, in a PDF-reader interface: page number, zoom, a contents sidebar. The interface follows the book's language (English or German).

- **⌘F** (Ctrl+F) opens the find bar. Type a **question** and press Enter: every relevant passage is highlighted on the page, the current one in orange.
  - `↵` / `⇧↵` (or ⌘G / ⇧⌘G) step through the hits.
  - Markers on the scrollbar show where the hits are.
  - **List** shows them ranked, with section and pages.
- **"A term in quotes"** searches the exact text, like Ctrl+F in any PDF reader: every occurrence, starting at the current page.
- **Precise** turns on the reranker. Off by default.
- Deep links: `/?q=question`, `/#page=74`, `/?q=…&list=1`.

The PDF is found through `book.json`; `--pdf` overrides it. On the first start the book is embedded (about 40 s for 300 pages); after that the reader starts in a few seconds.

How the highlights get onto the page:
- The encoder stored a box for every word (`<book>/.cache/ocr`), from the PDF's text layer or from OCR.
- Each retrieved chunk is matched against the words of its pages with 3-word windows. The text is compacted first, so hyphenation and line breaks don't matter.
- A window only counts if the next window continues it on the page. That rejects stray matches of common phrases.
- Figure labels and table cells are matched only on pages where the passage is already anchored.
- The two pages either side are searched too, because a paragraph that runs across a page break can carry the wrong page label.

On the 300-page reference book, 98% of the retrieved text is found on the page. The rest is mainly formula tables.

**Lightweight by default:**
- Answering a question takes one embedding of the question (about 20 ms on an Apple GPU) plus a FAISS lookup, so the fans stay off.
- A one-off `ask` takes about 10 s, most of it loading the model (about 1 GB RAM). In interactive mode and in the reader every further question is instant.

## How it works

1. **Chunking** (`chunking.py`, LangChain) cleans the Markdown:
   - margin notes are attached to their paragraph
   - info boxes are flattened
   - page markers are kept as metadata
   - bibliography, index and endnotes are left out

   Then `MarkdownHeaderTextSplitter` cuts the text along chapter and section boundaries, and `RecursiveCharacterTextSplitter` cuts it into overlapping chunks of 1,400 characters. Every chunk carries its breadcrumb (`Chapter 2 › 2.3 … › 2.3.1 …`) and page range. The breadcrumb is embedded together with the text.
2. **Vector search** (`retrieval.py`): a FAISS vector store built with `multilingual-e5-large-instruct`. The question is embedded with a task instruction that names the book ("Given a question about the book "…", retrieve the passages that answer it").
3. **Adaptive cut-off**: instead of a fixed top-k, every chunk whose similarity is close enough to the best hit is returned, up to 10. A definition question returns a few passages and a "list all …" question returns more.
4. **Gap filling**: if two hits from the same section are at most 2 chunks apart, the chunks between them are added too, so the middle of an enumeration is not skipped. Neighbouring chunks are merged into one passage for display.
5. **Optional `--precise`**: the 30 best candidates are rescored by a cross-encoder (`Qwen3-Reranker-0.6B`). Its order is blended 2:1 with the embedding order via `ContextualCompressionRetriever`.

The defaults were chosen by `rag.py eval` on a 300-page German textbook with 94 gold questions. To tune them for your own book, write questions for it and run `eval` (below). It writes the winner to `<book>/rag/best_config.json`, and `ask` and the reader use that file from then on.

## Results on the reference book

94 gold questions with 500 required passages; 19 of the questions are in English (the book is German) and 14 span several chapters.

| pipeline | recall@5 | recall@10 | MRR | GPU time / question |
|---|---:|---:|---:|---:|
| BM25 (keywords only) | 0.420 | 0.518 | 0.585 | 1 ms |
| multilingual-e5-small | 0.540 | 0.651 | 0.715 | 7 ms |
| bge-m3 | 0.670 | 0.756 | 0.831 | 21 ms |
| Qwen3-Embedding-0.6B | 0.662 | 0.772 | 0.825 | 48 ms |
| snowflake-arctic-embed-l-v2 | 0.681 | 0.786 | 0.862 | 21 ms |
| **multilingual-e5-large-instruct (default)** | **0.705** | **0.808** | **0.880** | **22 ms** |
| e5-large-instruct + Qwen3 reranker (`--precise`) | 0.757 | 0.853 | 0.888 | ~3,400 ms |

With the adaptive cut-off and gap filling, `ask` found **81% of all required passages** and returned about 9 chunks per question. Many of those chunks are neighbours, so they display as fewer passages.

Weak spots:
- Broad cross-chapter questions: recall@10 is about 0.69.
- The tail of long enumerations.

## Evaluation on your own book

```bash
uv run rag/rag.py --book mybook validate        # check the question files
uv run rag/rag.py --book mybook eval            # compare models, chunk sizes, cut-offs (~3 min for 300 pages)
uv run rag/rag.py --book mybook eval --rerankers qwen3-0.6b,bge-v2-m3 --max-latency-ms 0   # incl. rerankers (heavy)
```

- **Gold questions** go in `<book>/rag/questions/*.jsonl`. Each line is a question plus verbatim evidence quotes: *all* passages a complete answer needs. The quotes are matched against the text, so the labels survive changes to the chunking. A good set mixes definitions, lists spread over several pages, comparisons, scenarios, figure questions, cross-chapter questions and questions in other languages. Aim for 30 or more; with fewer the comparison is noisy and the built-in defaults are the safer choice.

  ```json
  {"id": "q-01", "question": "…", "lang": "en", "type": "definition", "evidence": [{"quote": "8–25 words copied from the book", "page": "74"}]}
  ```

  [`examples/questions.example.jsonl`](../examples/questions.example.jsonl) shows the format for the demo book.
- **Metrics**:
  - `R@k`: share of the required passages in the top k
  - `full@10`: all of them found
  - `MRR`: rank of the first relevant hit
  - `R@4k`: recall within a 4,000-character reading budget, used to compare chunk sizes fairly
  - for the adaptive cut-off: precision, recall and F3 (recall weighted 3×, because the goal is to find *all* relevant parts)
- **Choosing the winner**: highest mean of R@5, R@10 and R@4k, among pipelines no slower than `--max-latency-ms` (default 250 ms). Among pipelines within 0.01 of the best that also rank the first hit about as well (MRR), the fastest one wins.
- **Offline variants**: reranker fusion weights, candidate-pool sizes and cut-off thresholds are evaluated on stored scores, so they need no extra model calls.
- **Output**: `<book>/rag/report.md` (all pipelines, chunk sizes, question types and languages), `misses.md` (required passages that were not found) and `results.json`.

If the Markdown is regenerated, run `eval` again, because chunks and evidence locations change.

### Candidate models

Embedding models (`--embedders`):

| key | model | size |
|---|---|---|
| `e5-large-instruct` | intfloat/multilingual-e5-large-instruct | 560M |
| `arctic-l-v2` | Snowflake/snowflake-arctic-embed-l-v2.0 | 568M |
| `bge-m3` | BAAI/bge-m3 | 568M |
| `qwen3-0.6b` | Qwen/Qwen3-Embedding-0.6B | 596M |
| `e5-small` | intfloat/multilingual-e5-small | 118M |
| `openai-3-large`, `openai-3-small` | OpenAI text-embedding-3 | only with `OPENAI_API_KEY` set; **sends the book text to OpenAI** |

Rerankers (`--rerankers`; only with `--precise` or in the heavy eval):

| key | model | size |
|---|---|---|
| `qwen3-0.6b` | tomaarsen/Qwen3-Reranker-0.6B-seq-cls (Qwen3-Reranker-0.6B as a cross-encoder) | 596M |
| `bge-v2-m3` | BAAI/bge-reranker-v2-m3 | 568M |

## Where things are stored

Everything book-specific lives in the book folder, nothing in the code:

| path | contents |
|---|---|
| `<book>/book.json` | title, language, PDF path (written by the encoder). An optional `"subject"`, e.g. `"a German textbook on marketing"`, replaces the title in the embedding instruction. |
| `<book>/.cache/rag/` | embeddings (per model and text) and rendered page images. Delete it to start fresh. |
| `<book>/rag/questions/` | your evaluation questions |
| `<book>/rag/best_config.json` | tuned settings from `eval` (optional) |
| `<book>/rag/report.md`, `misses.md`, `results.json` | evaluation results |
