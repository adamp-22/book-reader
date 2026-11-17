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
# ]
# ///
"""Search a book: embedding retrieval with adaptive cut-off (optional BM25 / cross-encoder reranking).

  uv run rag/rag.py --book mybook ask "What are the five forces?"
  uv run rag/rag.py --book mybook ask         # interactive: type one question after another
  uv run rag/rag.py --book mybook eval        # benchmark models/pipelines, writes mybook/rag/report.md
  uv run rag/rag.py --book mybook build       # pre-compute the index (otherwise done on first ask)
  uv run rag/rag.py --book mybook validate    # check the evaluation questions in mybook/rag/questions/

--book is a folder written by md_encoder; default: $BOOK_DIR, else ./markdown, else the current folder.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import warnings
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import book  # noqa: E402
from chunking import ChunkConfig, chunk_book, page_label  # noqa: E402
from retrieval import DEFAULT, PRECISE, RERANKERS, PipelineConfig, available_embedders, build_retriever  # noqa: E402

RED, BOLD, DIM, RESET = "\033[31m", "\033[1m", "\033[2m", "\033[0m"


def load_best() -> tuple[PipelineConfig, ChunkConfig]:
    """The book's tuned settings (from `eval`), else the defaults tuned on the reference book."""
    path = book.current().data / "best_config.json"
    if not path.exists():
        return PipelineConfig(**DEFAULT), ChunkConfig()
    best = json.loads(path.read_text())
    return PipelineConfig(**best["pipeline"]), ChunkConfig(**best["chunking"])


# ---------------------------------------------------------------- output
def _join(a: str, b: str) -> str:
    """Concatenate two overlapping chunk bodies without repeating the overlap."""
    for n in range(min(len(a), len(b), 600), 20, -1):
        if a.endswith(b[:n]):
            return a + b[n:]
    return a + "\n\n" + b


def merge_passages(docs: list) -> list[dict]:
    """Neighbouring chunks of the same section become one passage, ranked by its best chunk."""
    ordered = sorted(docs, key=lambda d: (d.metadata["file"], (d.metadata["span"] or [0])[0]))
    groups: list[list] = []
    for d in ordered:
        prev = groups[-1][-1] if groups else None
        if (prev and prev.metadata["file"] == d.metadata["file"] and prev.metadata["section"] == d.metadata["section"]
                and prev.metadata["span"] and d.metadata["span"] and d.metadata["span"][0] <= prev.metadata["span"][1] + 5):
            groups[-1].append(d)
        else:
            groups.append([d])
    passages = []
    for g in groups:
        text = g[0].metadata["body"]
        for d in g[1:]:
            text = _join(text, d.metadata["body"])
        pages = list(dict.fromkeys(p for d in g for p in d.metadata["pages"]))
        passages.append({
            "score": max(d.metadata.get("score", 0) for d in g),
            "section": g[0].metadata["section"],
            "pages": page_label(pages),
            "text": text,
            "figures": [str(book.current().dir / f) for d in g for f in d.metadata["figures"]],
            "chunks": [d.metadata["id"] for d in g],
        })
    best_rank = {d.metadata["id"]: i for i, d in enumerate(docs)}
    return sorted(passages, key=lambda p: min(best_rank[c] for c in p["chunks"]))


def show(question: str, passages: list[dict], width: int) -> None:
    print(f"\n{BOLD}{question}{RESET}  {DIM}→ {len(passages)} passages{RESET}\n")
    for i, p in enumerate(passages, 1):
        crumbs = p["section"].split(" › ")
        print(f"{RED}{BOLD}[{i}] {crumbs[-1]}{RESET}  {DIM}{' › '.join(crumbs[:-1])}{RESET}")
        print(f"    {BOLD}{p['pages']}{RESET}  {DIM}score {p['score']:.2f}{RESET}")
        for para in p["text"].split("\n"):
            if not para.strip():
                continue
            if para.startswith("» "):
                print(f"    {BOLD}{para[2:]}{RESET}")
                continue
            for line in textwrap.wrap(para, width - 4) or [""]:
                print("    " + line)
        for f in p["figures"]:
            print(f"    {DIM}figure: {f}{RESET}")
        print()


# ---------------------------------------------------------------- commands
def make_retriever(args):
    cfg, chunk_cfg = load_best()
    if args.precise:
        for k, v in PRECISE.items():
            setattr(cfg, k, v)
    if args.embedders:
        cfg.embedders = args.embedders.split(",")
    if args.reranker:
        cfg.reranker = None if args.reranker == "none" else args.reranker
    if args.max:
        cfg.max_n = args.max
    if args.fixed:
        cfg.max_n, cfg.min_score, cfg.rel_to_top, cfg.min_n = args.fixed, 0.0, 0.0, args.fixed
    docs = chunk_book(book.current().dir, chunk_cfg)
    return build_retriever(docs, cfg), cfg, chunk_cfg


def cmd_ask(args) -> None:
    print(f"{DIM}{book.current().title}: loading index …{RESET}", file=sys.stderr)
    retriever, cfg, chunk_cfg = make_retriever(args)
    print(f"{DIM}{cfg.name}, chunks {chunk_cfg.size}/{chunk_cfg.overlap}{RESET}", file=sys.stderr)
    if cfg.reranker:
        print(f"{DIM}(reranker on: ~3–4 s of GPU per question){RESET}", file=sys.stderr)
    width = min(110, os.get_terminal_size().columns if sys.stdout.isatty() else 100)

    def answer(q: str) -> None:
        passages = merge_passages(retriever.invoke(q))
        if args.json:
            print(json.dumps({"question": q, "passages": passages}, ensure_ascii=False, indent=1))
        else:
            show(q, passages, width)

    if args.question:
        answer(" ".join(args.question))
        return
    print("Type a question (empty line = quit)")
    while True:
        try:
            q = input(f"{RED}?{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        answer(q)


def cmd_build(args) -> None:
    _, cfg, chunk_cfg = make_retriever(args)
    print(f"index ready: {cfg.name}, chunks {chunk_cfg.key}")


def cmd_eval(args) -> None:
    from evaluation import evaluate

    sizes = [int(s) for s in args.chunk_sizes.split(",")] if args.chunk_sizes else []
    emb = args.embedders.split(",") if args.embedders else available_embedders()
    rr = args.rerankers.split(",") if args.rerankers else []
    if not any(book.current().questions_dir.glob("*.jsonl")):
        sys.exit(f"No evaluation questions in {book.current().questions_dir} – see rag/README.md, 'Evaluation'.")
    best = evaluate(book.current().dir, emb, rr, sizes, max_latency_ms=args.max_latency_ms)
    print(json.dumps(best, indent=1))


def cmd_validate(args) -> None:
    from validate import main

    paths = sorted(str(p) for p in book.current().questions_dir.glob("*.jsonl"))
    if not paths:
        sys.exit(f"No evaluation questions in {book.current().questions_dir}")
    sys.exit(main(paths))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", help="book folder written by md_encoder (default: $BOOK_DIR, ./markdown or .)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ask", "build"):
        p = sub.add_parser(name)
        if name == "ask":
            p.add_argument("question", nargs="*")
            p.add_argument("--json", action="store_true", help="machine-readable output")
        p.add_argument("--precise", action="store_true",
                       help="add the Qwen3 reranker: better recall, but 3-4 s of GPU load per question")
        p.add_argument("--embedders", help="override the embedding model(s), comma-separated (see `eval`)")
        p.add_argument("--reranker", help=f"override the reranker ({', '.join(RERANKERS)}, none)")
        p.add_argument("--max", type=int, help="maximum number of chunks (default from best_config.json)")
        p.add_argument("--fixed", type=int, help="return exactly this many chunks (no adaptive cut-off)")
    p = sub.add_parser("eval")
    p.add_argument("--embedders", help=f"comma-separated, default: all available ({','.join(available_embedders())})")
    p.add_argument("--rerankers", default="", help=f"rerankers to compare, e.g. {','.join(RERANKERS)} (heavy; default none)")
    p.add_argument("--max-latency-ms", type=float, default=250,
                   help="only pipelines at most this slow per question can win (0 = no limit)")
    p.add_argument("--chunk-sizes", default="1000,1400,1800,2200", help="chunk sizes to compare ('' to skip)")
    sub.add_parser("validate")
    args = ap.parse_args()
    book.use(book.find(args.book))
    {"ask": cmd_ask, "build": cmd_build, "eval": cmd_eval, "validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    main()
