"""Evaluation harness.

Gold standard: <book>/rag/questions/*.jsonl. Each question lists the verbatim
"evidence" passages a complete answer needs. Evidence is located in the
normalised book text, so labels stay valid when the chunking changes. A
retrieved chunk "covers" an evidence passage if it contains at least half of it.

Metrics per question, averaged over all questions:
  - recall@k: share of the evidence passages covered by the top-k chunks.
    The key number for "retrieve all relevant parts".
  - full@k: 1 if ALL evidence passages are in the top k.
  - MRR: 1 / rank of the first relevant chunk.
  - recall@4k chars: recall within a fixed reading budget, so chunk sizes compare fairly.
  - auto (adaptive cut-off): precision, recall, F3 and the average number of chunks
    returned. F3 weighs recall three times as much as precision.
    Precision is a lower bound: the gold lists only what an answer needs, and
    other on-topic chunks count as misses.
"""
from __future__ import annotations

import itertools
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import book
from chunking import ChunkConfig, chunk_book, load_book, norm, page_label
from retrieval import (EMBEDDERS, PRECISE, PipelineConfig, available_embedders, build_retriever, fill_gaps,
                       release, rrf, select)

KS = (3, 5, 10, 20)
BUDGET = 4000


# ---------------------------------------------------------------- gold data
def load_questions(md_dir: Path) -> list[dict]:
    texts = {f: norm(t) for f, t in load_book(md_dir).items()}
    qs = []
    for path in sorted(book.current().questions_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            q = json.loads(line)
            for ev in q["evidence"]:
                nq = norm(ev["quote"])
                ev["loc"] = next(((f, t.find(nq), t.find(nq) + len(nq)) for f, t in texts.items() if nq in t), None)
            q["evidence"] = [ev for ev in q["evidence"] if ev["loc"]]
            if q["evidence"]:
                qs.append(q)
    return qs


def covered(doc_meta: dict, evidence: list[dict]) -> set[int]:
    span = doc_meta.get("span")
    if not span:
        return set()
    hits = set()
    for i, ev in enumerate(evidence):
        f, s, e = ev["loc"]
        if f == doc_meta["file"] and min(e, span[1]) - max(s, span[0]) >= 0.5 * (e - s):
            hits.add(i)
    return hits


# ---------------------------------------------------------------- metrics
def rank_metrics(q: dict, docs: list) -> dict:
    ev = q["evidence"]
    cov = [covered(d.metadata, ev) for d in docs]
    out = {}
    for k in KS:
        got = set().union(*cov[:k]) if docs else set()
        out[f"R@{k}"] = len(got) / len(ev)
        out[f"full@{k}"] = float(len(got) == len(ev))
    first = next((i for i, c in enumerate(cov) if c), None)
    out["MRR"] = 1 / (first + 1) if first is not None else 0.0
    got, used = set(), 0
    for d, c in zip(docs, cov):
        if used and used + len(d.metadata["body"]) > BUDGET:
            break
        used += len(d.metadata["body"])
        got |= c
    out["R@4k"] = len(got) / len(ev)
    out["pool"] = len(set().union(*cov)) / len(ev) if docs else 0.0  # recall of everything returned
    return out


def cutoff(docs: list, min_score: float, rel: float, max_n: int, min_n: int = 1) -> list:
    if not docs:
        return []
    floor = max(min_score, rel * docs[0].metadata["score"])
    return [d for i, d in enumerate(docs[:max_n]) if i < min_n or d.metadata["score"] >= floor]


def auto_metrics(q: dict, docs: list) -> dict:
    ev = q["evidence"]
    cov = [covered(d.metadata, ev) for d in docs]
    got = set().union(*cov) if docs else set()
    p = sum(bool(c) for c in cov) / len(docs) if docs else 0.0
    r = len(got) / len(ev)
    f2 = 5 * p * r / (4 * p + r) if p + r else 0.0
    f3 = 10 * p * r / (9 * p + r) if p + r else 0.0
    return {"P": p, "R": r, "F2": f2, "F3": f3, "n": len(docs)}


def mean_metrics(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {k: statistics.fmean(r[k] for r in rows) for k in keys}


# ---------------------------------------------------------------- runs
def accuracy(m: dict) -> float:
    """Single number used to pick the winner: mean of R@5, R@10 and R@4k."""
    return (m["R@5"] + m["R@10"] + m["R@4k"]) / 3


def reorder(docs: list, fuse_weight: float) -> list:
    """Re-derive the final order of a reranked pool for another fusion weight (no model calls)."""
    by_score = sorted(range(len(docs)), key=lambda i: -docs[i].metadata["score"])
    by_first = sorted(range(len(docs)), key=lambda i: docs[i].metadata["rank0"])
    order = rrf([by_score, by_first], [fuse_weight, 1.0]) if fuse_weight else by_score
    return [docs[i] for i in order]


def run(docs, questions, cfg: PipelineConfig, rerankers: dict, log=print, label: str | None = None) -> dict:
    # keep every candidate with its score; fusion weight and cut-off are evaluated offline
    cfg = replace(cfg, max_n=cfg.candidates, min_score=0.0, rel_to_top=0.0, max_drop=1e9, max_gap=0)
    t0 = time.perf_counter()
    retriever = build_retriever(docs, cfg, rerankers)
    build_s = time.perf_counter() - t0
    per_q, lat = [], []
    for q in questions:
        t = time.perf_counter()
        res = retriever.invoke(q["question"])
        lat.append(time.perf_counter() - t)
        per_q.append({"q": q, "docs": res, "m": rank_metrics(q, res)})
    summary = mean_metrics([r["m"] for r in per_q])
    summary["latency_ms"] = 1000 * statistics.median(lat)
    log(f"  {(label or cfg.name):58s} R@5 {summary['R@5']:.3f}  R@10 {summary['R@10']:.3f}  R@4k {summary['R@4k']:.3f}  "
        f"MRR {summary['MRR']:.3f}  {summary['latency_ms']:.0f} ms/q  (setup {build_s:.0f}s)")
    return {"cfg": cfg, "summary": summary, "per_q": per_q, "chunks": docs}


def with_weight(result: dict, w: float, pool: int | None = None) -> dict:
    """Same reranked run, other fusion weight and/or a smaller candidate pool (latency scaled accordingly)."""
    pool = pool or result["cfg"].candidates
    per_q = [{"q": r["q"], "docs": reorder([d for d in r["docs"] if d.metadata["rank0"] < pool], w)}
             for r in result["per_q"]]
    for r in per_q:
        r["m"] = rank_metrics(r["q"], r["docs"])
    lat = result["summary"]["latency_ms"] * pool / result["cfg"].candidates
    summary = {**mean_metrics([r["m"] for r in per_q]), "latency_ms": lat}
    return {"cfg": replace(result["cfg"], fuse_weight=w, candidates=pool), "summary": summary, "per_q": per_q,
            "chunks": result["chunks"]}


def cutoff(docs: list, min_score: float, rel: float, max_n: int, max_drop: float = 1e9, min_n: int = 1) -> list:
    scores = [d.metadata["score"] for d in docs]
    return [docs[i] for i in select(list(range(len(docs))), scores, max_n, min_n, min_score, rel, max_drop)]


def has_scores(result: dict) -> bool:
    return any("score" in d.metadata for r in result["per_q"] for d in r["docs"][:1])


def tune_cutoff(result: dict) -> dict:
    """Grid-search the adaptive cut-off. Objective: F3, i.e. recall weighted 3× over precision,
    because the goal is to return *all* relevant passages.

    Reranker scores are probabilities → absolute and relative floors. Embedding similarities are
    compressed into a narrow band → absolute floor and maximum drop below the best hit."""
    if result["cfg"].reranker:
        grid = [(ms, rel, 1e9) for ms in (0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5)
                for rel in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)]
    else:
        grid = [(ms, 0.0, drop) for ms in (0.0, 0.76, 0.78, 0.8, 0.82, 0.84)
                for drop in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 1e9)]
    best = None
    for (ms, rel, drop), gap, max_n in itertools.product(grid, (0, 2, 4, 6), (10, 12, 15)):
        m = mean_metrics([auto_metrics(r["q"], fill_gaps(cutoff(r["docs"], ms, rel, max_n, drop), result["chunks"], gap))
                          for r in result["per_q"]])
        if best is None or m["F3"] > best["F3"]:
            best = {**m, "min_score": ms, "rel_to_top": rel, "max_drop": drop, "max_n": max_n, "max_gap": gap}
    return best


def breakdown(result: dict, key: str) -> dict:
    groups = defaultdict(list)
    for r in result["per_q"]:
        groups[r["q"].get(key, "?")].append(r["m"])
    return {g: {"n": len(v), **mean_metrics(v)} for g, v in sorted(groups.items())}


def pick(results: dict, keys: list[str], tol: float = 0.01) -> str:
    """Most accurate; among those within `tol` that also rank the first hit about as well (MRR), the fastest.
    The MRR condition matters for small or easy question sets, where every pipeline reaches recall 1.0."""
    top = max(accuracy(results[k]["summary"]) for k in keys)
    near = [k for k in keys if accuracy(results[k]["summary"]) >= top - tol]
    best_mrr = max(results[k]["summary"]["MRR"] for k in near)
    near = [k for k in near if results[k]["summary"]["MRR"] >= best_mrr - 2 * tol]
    # fastest first; equal speed (e.g. same run, other fusion weight) → the more accurate one
    return min(near, key=lambda k: (round(results[k]["summary"]["latency_ms"], -1), -accuracy(results[k]["summary"])))


# ---------------------------------------------------------------- report
def _table(rows: list[tuple[str, dict]], cols: list[str], first: str = "pipeline") -> str:
    out = f"| {first} | " + " | ".join(cols) + " |\n|---|" + "---:|" * len(cols) + "\n"
    for name, m in rows:
        cells = []
        for c in cols:
            v = m.get(c)
            cells.append("–" if v is None else (f"{v:.0f}" if c in ("latency_ms", "chunks") else
                                               (v if isinstance(v, str) else f"{v:.3f}")))
        out += f"| {name} | " + " | ".join(cells) + " |\n"
    return out


def evaluate(md_dir: Path, embedders: list[str] | None, rerankers: list[str], chunk_sizes: list[int],
             max_latency_ms: float = 0, log=None) -> dict:
    """max_latency_ms > 0: only pipelines at most this slow per question can win (lightweight mode)."""
    log = log or (lambda *a: print(*a, flush=True))
    questions = load_questions(md_dir)
    n_ev = sum(len(q["evidence"]) for q in questions)
    log(f"{len(questions)} questions, {n_ev} evidence passages")
    if len(questions) < 30:
        log("note: with fewer than ~30 questions the comparison is noisy – the built-in defaults "
            "(tuned on a 94-question set) may be the safer choice; delete rag/best_config.json to go back to them")
    embedders = embedders or available_embedders()
    main_chunks = ChunkConfig()
    docs = chunk_book(md_dir, main_chunks)
    log(f"chunking {main_chunks.key}: {len(docs)} chunks")
    models: dict = {}  # loaded rerankers, shared between runs
    results: dict[str, dict] = {}

    def add(cfg: PipelineConfig, d=docs, label=None):
        res = run(d, questions, cfg, models, log=log, label=label)
        results[label or cfg.name] = res
        return res

    # 1) first stage
    log("\n[1] first stage: BM25, each embedding model, fusions")
    add(PipelineConfig(embedders=[], bm25=True, reranker=None))
    for e in embedders:
        add(PipelineConfig(embedders=[e], reranker=None))
        release()
    ranked = sorted(embedders, key=lambda e: -accuracy(results[e]["summary"]))
    fusions = []
    for n in (2, 3):
        if len(ranked) >= n:
            fusions.append(add(PipelineConfig(embedders=ranked[:n], reranker=None)))
    first_keys = [k for k, v in results.items() if not v["cfg"].reranker]
    best_first = results[pick(results, first_keys)]["cfg"]
    add(replace(best_first, bm25=True))
    first_keys.append(replace(best_first, bm25=True).name)
    best_first = results[pick(results, first_keys)]["cfg"]
    log(f"  → best first stage: {best_first.name}")
    release(set(best_first.embedders))
    models.clear()

    # 2) rerankers on the best first stage; fusion weight evaluated offline
    if rerankers:
        log("\n[2] rerankers (final order = weighted RRF of reranker and first stage)")
    for rk in rerankers:
        models.clear()
        release(set(best_first.embedders))
        base = add(replace(best_first, reranker=rk, fuse_weight=0.0), label=f"{best_first.name} → {rk} rerank (only)")
        for w in (1.0, 2.0, 3.0):
            res = with_weight(base, w)
            results[res["cfg"].name] = res
            log(f"  {res['cfg'].name:58s} R@5 {res['summary']['R@5']:.3f}  R@10 {res['summary']['R@10']:.3f}  "
                f"R@4k {res['summary']['R@4k']:.3f}  MRR {res['summary']['MRR']:.3f}")
    eligible = [k for k in results if not max_latency_ms or results[k]["summary"]["latency_ms"] <= max_latency_ms]
    winner = pick(results, eligible)
    if results[winner]["cfg"].reranker:  # smaller candidate pools = proportionally faster reranking
        src = results[f"{best_first.name} → {results[winner]['cfg'].reranker} rerank (only)"]
        for n in (15, 20, 30):
            res = with_weight(src, results[winner]["cfg"].fuse_weight, n)
            key = f"{res['cfg'].name} [{n} candidates]"
            results[key] = res
            log(f"  {key:58s} R@5 {res['summary']['R@5']:.3f}  R@10 {res['summary']['R@10']:.3f}  "
                f"R@4k {res['summary']['R@4k']:.3f}  MRR {res['summary']['MRR']:.3f}  ~{res['summary']['latency_ms']:.0f} ms/q")
        eligible = [k for k in results if not max_latency_ms or results[k]["summary"]["latency_ms"] <= max_latency_ms]
        winner = pick(results, eligible)
    win_cfg = results[winner]["cfg"]
    models.clear()
    release(set(win_cfg.embedders) | {f"rerank:{win_cfg.reranker}"})
    log(f"\nchosen pipeline: {winner}" + (f" (limit {max_latency_ms:g} ms/question)" if max_latency_ms else ""))

    # 3) chunk size sweep with the winning pipeline
    sweep, finals = {}, {main_chunks.key: results[winner]}
    if chunk_sizes:
        log("\n[3] chunk size sweep")
        for size in chunk_sizes:
            cc = ChunkConfig(size=size, overlap=size * 15 // 100)
            if cc.key not in finals:
                d = chunk_book(md_dir, cc)
                res = run(d, questions, replace(win_cfg, fuse_weight=0.0) if win_cfg.reranker else win_cfg, models, log=log,
                          label=f"{cc.key} ({len(d)} chunks)")
                finals[cc.key] = with_weight(res, win_cfg.fuse_weight) if win_cfg.reranker else res
            res = finals[cc.key]
            sweep[cc.key] = {"size": cc.size, "overlap": cc.overlap, **res["summary"],
                             **({f"auto_{k}": v for k, v in tune_cutoff(res).items()} if has_scores(res) else {})}
    best_key = max(sweep, key=lambda k: sweep[k]["R@4k"] + sweep[k].get("auto_F3", 0)) if sweep else main_chunks.key
    final = finals[best_key]
    best_chunk = ChunkConfig(size=int(best_key[1:].split("-")[0]), overlap=int(best_key.split("-o")[1]))
    tuned = tune_cutoff(final) if has_scores(final) else None

    chosen = asdict(win_cfg)
    if tuned:
        chosen.update(min_score=tuned["min_score"], rel_to_top=tuned["rel_to_top"], max_drop=tuned["max_drop"],
                      max_n=tuned["max_n"], max_gap=tuned["max_gap"])
    best = {"pipeline": chosen, "chunking": {"size": best_chunk.size, "overlap": best_chunk.overlap},
            "precise": PRECISE,
            "selected_by": f"evaluation on {len(questions)} questions",
            "scores": {k: round(v, 4) for k, v in final["summary"].items()},
            "auto_cutoff": {k: round(v, 4) for k, v in (tuned or {}).items()}}
    book.current().data.mkdir(parents=True, exist_ok=True)
    (book.current().data / "best_config.json").write_text(json.dumps(best, indent=1))
    log(f"chunking: {best_key}" + (f" | adaptive cut-off → recall {tuned['R']:.3f}, precision {tuned['P']:.3f}, "
                                   f"{tuned['n']:.1f} chunks per question" if tuned else ""))

    timings = json.loads((book.current().cache / "index_timings.json").read_text())
    report = write_report(questions, results, winner, final, tuned, sweep, timings, best, embedders, main_chunks)
    write_misses(final)
    log(f"\nreport → {report}")
    return best


def write_report(questions, results, winner, final, tuned, sweep, timings, best, embedders, main_chunks) -> Path:
    n_ev = sum(len(q["evidence"]) for q in questions)
    cols = ["R@3", "R@5", "R@10", "R@20", "full@10", "MRR", "R@4k", "pool", "latency_ms"]
    rows = sorted(results.items(), key=lambda kv: accuracy(kv[1]["summary"]))
    lines = [
        "# Retrieval evaluation",
        "",
        f"{len(questions)} questions ({sum(q.get('lang') == 'en' for q in questions)} in English), {n_ev} required "
        f"evidence passages. Pipelines compared on chunking {main_chunks.key}.",
        "",
        "- `R@k`: share of the required passages found in the top k chunks",
        "- `full@10`: all required passages in the top 10",
        "- `MRR`: 1 / rank of the first relevant chunk",
        "- `R@4k`: recall within the first 4,000 characters, so chunk sizes compare fairly",
        "- `pool`: recall of the whole candidate list, the ceiling for reranking",
        "- `latency_ms`: median time per question on this machine",
        "",
        "The winner is the most accurate pipeline by mean of R@5, R@10 and R@4k. Among pipelines within 0.01 of it, "
        "the fastest wins.",
        "",
        "## All pipelines (worst → best)",
        "",
        _table([(k, v["summary"]) for k, v in rows], cols),
        "",
        "## Embedding models (alone)",
        "",
        "| model | params | index time, s (uncached) | R@10 | MRR | latency_ms |\n|---|---|---:|---:|---:|---:|",
        *[f"| {e} | {EMBEDDERS[e]['params']} | {timings.get(e, {}).get('seconds', '–')} | "
          f"{results[e]['summary']['R@10']:.3f} | {results[e]['summary']['MRR']:.3f} | "
          f"{results[e]['summary']['latency_ms']:.0f} |" for e in embedders if e in results],
        "",
        f"## Chosen: `{winner}`",
        "",
        f"Chunking: size {best['chunking']['size']}, overlap {best['chunking']['overlap']}. "
        + " · ".join(f"{k} {final['summary'][k]:.3f}" for k in ("R@5", "R@10", "R@20", "full@10", "MRR", "R@4k")),
        "",
    ]
    if tuned:
        rule = (f"the reranker scores ≥ {tuned['min_score']} and ≥ {tuned['rel_to_top']} × the best score"
                if final["cfg"].reranker else
                f"whose similarity is ≥ {tuned['min_score']} and at most {tuned['max_drop']:g} below the best hit")
        lines += [
            f"Adaptive cut-off (used by `ask`): keep chunks {rule}, at most {tuned['max_n']}; then add the chunks "
            f"between two hits of the same section that are at most {tuned['max_gap']} chunks apart. Result: "
            f"**recall {tuned['R']:.3f}, precision {tuned['P']:.3f}, F3 {tuned['F3']:.3f}, "
            f"{tuned['n']:.1f} chunks per question on average**. Precision is a lower bound: the gold set lists "
            "only the passages an answer needs, so related on-topic chunks count as misses.",
            "",
        ]
    if not final["cfg"].reranker:
        lines += [
            "Rerankers are not part of the default because they load the GPU for a few seconds per question; "
            "`ask --precise` switches the Qwen3 reranker on. To measure them on this book, run "
            "`eval --rerankers qwen3-0.6b --max-latency-ms 0` (results in `report-rerankers.md`).",
            "",
        ]
    if sweep:
        scols = ["R@5", "R@10", "R@4k", "MRR", "auto_R", "auto_P", "auto_F3", "auto_n", "latency_ms"]
        lines += ["## Chunk size (chosen pipeline)", "", _table(list(sweep.items()), scols, "chunking"), ""]
    for key, title in (("type", "question type"), ("lang", "language")):
        b = breakdown(final, key)
        lines += [f"## By {title}", "", _table([(f"{g} (n={v['n']})", v) for g, v in b.items()],
                                               ["R@5", "R@10", "full@10", "MRR"], key), ""]
    path = book.current().data / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (book.current().data / "results.json").write_text(json.dumps(
        {"pipelines": {k: v["summary"] for k, v in results.items()}, "chunk_sweep": sweep, "chosen": best}, indent=1))
    return path


def write_misses(result: dict) -> None:
    out = ["# Questions with evidence missing from the top 10", ""]
    for r in result["per_q"]:
        if r["m"]["R@10"] == 1:
            continue
        q, docs = r["q"], r["docs"]
        got = set().union(*[covered(d.metadata, q["evidence"]) for d in docs[:10]])
        out.append(f"## {q['id']} ({q.get('type')}, {q.get('lang')}): {q['question']}")
        out.append(f"recall@10 {r['m']['R@10']:.2f}")
        for i, ev in enumerate(q["evidence"]):
            if i not in got:
                rank = next((j + 1 for j, d in enumerate(docs) if i in covered(d.metadata, q["evidence"])), None)
                out.append(f"- missing ({book.current().page_label([str(ev.get('page'))])}, rank {rank or '>' + str(len(docs))}): {ev['quote'][:120]}")
        out.append("- top 3: " + "; ".join(f"{page_label(d.metadata['pages'])} {d.metadata['section'].split(' › ')[-1]}"
                                         for d in docs[:3]))
        out.append("")
    (book.current().data / "misses.md").write_text("\n".join(out), encoding="utf-8")
