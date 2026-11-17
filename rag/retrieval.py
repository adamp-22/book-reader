"""Retrieval pipeline built from LangChain components.

    question ─┬─ FAISS vector store, embedding model A ─┐
              ├─ FAISS vector store, embedding model B ─┼─ EnsembleRetriever (reciprocal rank fusion) → candidate pool
              └─ BM25Retriever (stemming, optional) ──────┘
                                    │
          ContextualCompressionRetriever + cross-encoder reranker; final order = weighted RRF of
          the reranker's order and the first-stage order
                                    │
          adaptive cut-off: keep everything the reranker scores close enough to the best hit

Every stage can be switched off, so the evaluation compares stages, embedding models and rerankers.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.retrievers import BaseRetriever

import book

# Candidate embedding models: all multilingual (50+ languages).
# query/doc prompts follow each model card; {task} is filled with the book's instruction (book.query_task).
EMBEDDERS: dict[str, dict] = {
    "bge-m3": {"name": "BAAI/bge-m3", "params": "568M"},
    "qwen3-0.6b": {"name": "Qwen/Qwen3-Embedding-0.6B", "params": "596M",
                   "query_prompt": "Instruct: {task}\nQuery: "},
    "e5-large-instruct": {"name": "intfloat/multilingual-e5-large-instruct", "params": "560M",
                          "query_prompt": "Instruct: {task}\nQuery: "},
    "arctic-l-v2": {"name": "Snowflake/snowflake-arctic-embed-l-v2.0", "params": "568M", "query_prompt": "query: "},
    "e5-small": {"name": "intfloat/multilingual-e5-small", "params": "118M",
                 "query_prompt": "query: ", "doc_prompt": "passage: "},
    # Only used when OPENAI_API_KEY is set. This sends the book text to OpenAI.
    "openai-3-large": {"provider": "openai", "name": "text-embedding-3-large", "params": "API"},
    "openai-3-small": {"provider": "openai", "name": "text-embedding-3-small", "params": "API"},
}
RERANKERS: dict[str, dict] = {
    "bge-v2-m3": {"name": "BAAI/bge-reranker-v2-m3", "params": "568M"},
    "qwen3-0.6b": {"name": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls", "params": "596M", "template": "qwen3"},
}
_QWEN_PRE = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
             'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
_QWEN_SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def available_embedders() -> list[str]:
    return [k for k, v in EMBEDDERS.items() if v.get("provider") != "openai" or os.environ.get("OPENAI_API_KEY")]


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- BM25 (lexical)
_STOP = set("""
aber alle allem allen aller alles als also am an ander andere anderem anderen anderer anderes auch auf aus bei bin bis bist
da damit dann der den des dem die das dass daß du dein deine dem denn derer dessen dich dir dies diese diesem diesen dieser
dieses doch dort durch ein eine einem einen einer eines einig einige er es etwas euer für gegen gewesen hab habe haben hat
hatte hatten hier hin hinter ich ihr ihre im in indem ins ist jede jedem jeden jeder jedes jene jenem jenen jener jenes jetzt
kann kein keine können könnte machen man manche mein mit muss musste nach nicht nichts noch nun nur ob oder ohne sehr sein
seine sich sie sind so solche soll sollte sondern sonst über um und uns unter viel vom von vor während war waren warum was
weg weil weiter welche welchem welchen welcher welches wenn wer werde werden wie wieder will wir wird wirst wo wollen wollte
würde würden zu zum zur zwar zwischen z b bzw u a s welche gibt versteht unter
the a an and or of to in is are what which how why does do for on with as by be it this that from
""".split())


def make_tokenizer():
    import snowballstemmer

    language = {"de": "german", "en": "english", "fr": "french", "es": "spanish", "it": "italian",
                "nl": "dutch", "pt": "portuguese"}.get(book.current().language, "english")
    stem = snowballstemmer.stemmer(language).stemWord

    def tokenize(text: str) -> list[str]:
        words = re.findall(r"\w+", text.lower())
        return [stem(w) for w in words if w not in _STOP and len(w) > 1]

    return tokenize


def bm25_retriever(docs: list[Document], k: int):
    from langchain_community.retrievers import BM25Retriever

    return BM25Retriever.from_documents(docs, k=k, preprocess_func=make_tokenizer())


# ---------------------------------------------------------------- dense
_LOADED: dict[str, object] = {}  # each model is loaded once per process


def release(keep: set[str] = frozenset()) -> None:
    """Unload models not in `keep` (keys as in EMBEDDERS, rerankers as 'rerank:<key>') to free GPU memory."""
    import gc

    for k in [k for k in _LOADED if k not in keep]:
        del _LOADED[k]
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def embedding_model(key: str):
    if key in _LOADED:
        return _LOADED[key]
    _LOADED[key] = _load_embedding_model(key)
    return _LOADED[key]


def _load_embedding_model(key: str):
    spec = EMBEDDERS[key]
    if spec.get("provider") == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=spec["name"])
    from langchain_huggingface import HuggingFaceEmbeddings

    doc_kw = {"normalize_embeddings": True, "batch_size": 16}
    query_kw = {"normalize_embeddings": True}
    if spec.get("doc_prompt"):
        doc_kw["prompt"] = spec["doc_prompt"]
    if spec.get("query_prompt"):
        query_kw["prompt"] = spec["query_prompt"].format(task=book.current().query_task)
    emb = HuggingFaceEmbeddings(model_name=spec["name"], model_kwargs={"device": device()},
                                encode_kwargs=doc_kw, query_encode_kwargs=query_kw)
    emb._client.max_seq_length = 512  # chunks are ≤ ~350 tokens; keeps long-context models fast
    return emb


def cached_embeddings(key: str):
    """LangChain CacheBackedEmbeddings: chunk vectors are computed once per model and text."""
    from langchain_classic.embeddings import CacheBackedEmbeddings
    from langchain_classic.storage import LocalFileStore

    store = LocalFileStore(str(book.current().cache / "embeddings"))
    return CacheBackedEmbeddings.from_bytes_store(embedding_model(key), store, namespace=key, key_encoder="sha256")


def vector_store(docs: list[Document], key: str):
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    emb = cached_embeddings(key)
    t = time.perf_counter()
    store = FAISS.from_documents(docs, emb, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    _record_timing(key, len(docs), time.perf_counter() - t)
    return store


def _record_timing(key: str, n: int, seconds: float) -> None:
    """Remember the slowest (= uncached) indexing time per model for the efficiency table."""
    path = book.current().cache / "index_timings.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    if seconds > data.get(key, {}).get("seconds", 0):
        data[key] = {"seconds": round(seconds, 1), "chunks": n}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1))


# ---------------------------------------------------------------- reranking + adaptive cut-off
def rrf(orders: list[list[int]], weights: list[float], k: int = 60) -> list[int]:
    score: dict[int, float] = {}
    for order, w in zip(orders, weights):
        for rank, idx in enumerate(order):
            score[idx] = score.get(idx, 0.0) + w / (k + rank + 1)
    return sorted(score, key=lambda i: -score[i])


def select(order: list[int], scores: list[float], max_n: int, min_n: int = 1, min_score: float = 0.0,
           rel_to_top: float = 0.0, max_drop: float = 1e9) -> list[int]:
    """Adaptive cut-off: walk `order`, keep items scoring >= max(min_score, rel_to_top * best, best - max_drop)."""
    if not order:
        return []
    top = max(scores)
    floor = max(min_score, rel_to_top * top, top - max_drop)
    out = []
    for i in order:
        if len(out) >= max_n:
            break
        if len(out) < min_n or scores[i] >= floor:
            out.append(i)
    return out


class ScoredVectorRetriever(BaseRetriever):
    """Single-embedder vector search that keeps the similarity score and applies the adaptive cut-off.

    This is the lightweight default: one query embedding (~20 ms) and a FAISS lookup."""

    store: object
    k: int = 30
    max_n: int = 12
    min_n: int = 1
    min_score: float = 0.0
    max_drop: float = 1e9

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        hits = self.store.similarity_search_with_score(query, k=self.k)
        scores = [float(s) for _, s in hits]
        keep = select(list(range(len(hits))), scores, self.max_n, self.min_n, self.min_score, 0.0, self.max_drop)
        return [Document(page_content=hits[i][0].page_content,
                         metadata={**hits[i][0].metadata, "score": round(scores[i], 4), "rank0": i}) for i in keep]


def fill_gaps(selected: list[Document], chunks: list[Document], max_gap: int) -> list[Document]:
    """If two selected chunks of the same section are at most `max_gap` chunks apart, add the chunks
    between them (typically the middle of an enumeration). Added chunks get metadata filled=True."""
    if max_gap <= 0 or len(selected) < 2:
        return selected
    pos = {d.metadata["id"]: i for i, d in enumerate(chunks)}
    have = sorted(pos[d.metadata["id"]] for d in selected if d.metadata["id"] in pos)
    added = []
    for a, b in zip(have, have[1:]):
        if 1 < b - a <= max_gap + 1 and chunks[a].metadata["section"] == chunks[b].metadata["section"]:
            added += [Document(page_content=chunks[i].page_content, metadata={**chunks[i].metadata, "filled": True})
                      for i in range(a + 1, b)]
    return selected + added


class GapFillingRetriever(BaseRetriever):
    """Wraps another retriever and fills gaps between nearby hits of the same section (see fill_gaps)."""

    base: BaseRetriever
    chunks: list
    max_gap: int = 4

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return fill_gaps(self.base.invoke(query), self.chunks, self.max_gap)


class ScoredReranker(BaseDocumentCompressor):
    """Cross-encoder reranker.

    Scores the first `pool` candidates. The final order is a weighted RRF of the reranker's
    order and the incoming first-stage order (weight 0 = reranker order only). Keeps hits
    the reranker scores >= max(min_score, rel_to_top * best score), at most max_n.
    """

    model: object
    template: str | None = None
    pool: int = 40
    fuse_weight: float = 2.0
    max_n: int = 12
    min_n: int = 1
    min_score: float = 0.0
    rel_to_top: float = 0.0
    max_drop: float = 1e9

    model_config = {"arbitrary_types_allowed": True}

    def _pairs(self, query: str, docs: list[Document]) -> list[tuple[str, str]]:
        if self.template == "qwen3":
            q = f"{_QWEN_PRE}<Instruct>: {book.current().rerank_task}\n<Query>: {query}\n"
            return [(q, f"<Document>: {d.page_content}{_QWEN_SUF}") for d in docs]
        return [(query, d.page_content) for d in docs]

    def compress_documents(self, documents: Sequence[Document], query: str, callbacks: Callbacks | None = None):
        docs = list(documents)[: self.pool]
        if not docs:
            return []
        scores = [float(s) for s in self.model.score(self._pairs(query, docs))]
        by_score = sorted(range(len(docs)), key=lambda i: -scores[i])
        order = rrf([by_score, list(range(len(docs)))], [self.fuse_weight, 1.0]) if self.fuse_weight else by_score
        keep = select(order, scores, self.max_n, self.min_n, self.min_score, self.rel_to_top, self.max_drop)
        return [Document(page_content=docs[i].page_content,
                         metadata={**docs[i].metadata, "score": round(scores[i], 4), "rank0": i}) for i in keep]


def cross_encoder(key: str):
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    if f"rerank:{key}" not in _LOADED:
        _LOADED[f"rerank:{key}"] = HuggingFaceCrossEncoder(model_name=RERANKERS[key]["name"],
                                                           model_kwargs={"device": device(), "max_length": 1024})
    return _LOADED[f"rerank:{key}"]


# ---------------------------------------------------------------- assembly
@dataclass
class PipelineConfig:
    embedders: list[str] = field(default_factory=lambda: ["e5-large-instruct"])
    bm25: bool = False
    reranker: str | None = None  # None = lightweight (no cross-encoder)
    fuse_weight: float = 2.0  # weight of the reranker's order vs. the first-stage order (0 = reranker only)
    candidates: int = 30  # per first-stage retriever; the reranker sees the best `candidates` of the fused list
    bm25_weight: float = 1.0  # relative to each embedder (weight 1)
    max_n: int = 12  # most results returned
    min_n: int = 1
    min_score: float = 0.0  # adaptive cut-off, see select()
    rel_to_top: float = 0.0
    max_drop: float = 1e9
    max_gap: int = 0  # fill gaps of up to this many chunks between hits of the same section

    @property
    def name(self) -> str:
        parts = (["bm25"] if self.bm25 else []) + list(self.embedders)
        name = " + ".join(parts)
        if self.reranker:
            name += f" → {self.reranker} rerank" + (f" (rrf {self.fuse_weight:g})" if self.fuse_weight else "")
        return name


# Optional high-accuracy mode (`ask --precise`): Qwen3 reranker over the 30 best candidates, its order blended
# 2:1 with the embedding order. Measured on 94 questions (eval/report-rerankers.md): recall@10 0.853 vs 0.810
# without it, but ~3–4 s of full GPU load per question.
# Settings for a book without its own evaluation (no <book>/rag/best_config.json): the winner of
# the evaluation on the reference book (94 questions): recall 0.81 at ~9 chunks, ~20 ms per question.
DEFAULT = {"embedders": ["e5-large-instruct"], "reranker": None, "candidates": 30, "max_n": 10,
           "min_score": 0.8, "rel_to_top": 0.0, "max_drop": 0.04, "max_gap": 2}
PRECISE = {"reranker": "qwen3-0.6b", "fuse_weight": 2.0, "candidates": 30,
           "min_score": 0.0, "rel_to_top": 0.5, "max_drop": 1e9, "max_n": 15, "max_gap": 4}


def build_retriever(docs: list[Document], cfg: PipelineConfig, rerankers: dict | None = None) -> BaseRetriever:
    retriever = _build(docs, cfg, rerankers)
    return GapFillingRetriever(base=retriever, chunks=docs, max_gap=cfg.max_gap) if cfg.max_gap > 0 else retriever


def _build(docs: list[Document], cfg: PipelineConfig, rerankers: dict | None = None) -> BaseRetriever:
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever

    if len(cfg.embedders) == 1 and not cfg.bm25:
        store = vector_store(docs, cfg.embedders[0])
        if not cfg.reranker:
            return ScoredVectorRetriever(store=store, k=max(cfg.candidates, cfg.max_n), max_n=cfg.max_n, min_n=cfg.min_n,
                                         min_score=cfg.min_score, max_drop=cfg.max_drop)
        first = ScoredVectorRetriever(store=store, k=cfg.candidates, max_n=cfg.candidates)
    else:
        stages, weights = [], []
        if cfg.bm25:
            stages.append(bm25_retriever(docs, cfg.candidates))
            weights.append(cfg.bm25_weight)
        for e in cfg.embedders:
            stages.append(vector_store(docs, e).as_retriever(search_kwargs={"k": cfg.candidates}))
            weights.append(1.0)
        first = EnsembleRetriever(retrievers=stages, weights=weights) if len(stages) > 1 else stages[0]
        if not cfg.reranker:
            return first
    model = (rerankers or {}).get(cfg.reranker) or cross_encoder(cfg.reranker)
    if rerankers is not None:
        rerankers[cfg.reranker] = model
    reranker = ScoredReranker(model=model, template=RERANKERS[cfg.reranker].get("template"), pool=cfg.candidates,
                              fuse_weight=cfg.fuse_weight, max_n=cfg.max_n, min_n=cfg.min_n,
                              min_score=cfg.min_score, rel_to_top=cfg.rel_to_top, max_drop=cfg.max_drop)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=first)
