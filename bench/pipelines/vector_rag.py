"""Traditional vector RAG baseline: page text -> token chunks -> OpenAI embeddings ->
cosine top-k -> answer with the SAME chat model PageIndex uses.

The vector store is an in-memory float32 matrix (exact cosine search). That is the
*best case* for vector-search latency and recall; a hosted vector DB (pgvector, Pinecone,
Azure AI Search) adds network latency and ANN approximation on top. Swap `VectorStore`
if your team wants to benchmark a specific DB.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from bench.common import clean, count_tokens, read_pdf_pages

try:
    from bench.common import _ENC  # tiktoken encoder, if installed
except ImportError:
    _ENC = None

ANSWER_SYSTEM = (
    "You are a clinical documentation assistant. Answer ONLY from the provided excerpts. "
    "Cite every claim as [doc_id p.N]. If the excerpts do not contain the answer, say "
    "'Not found in the provided documents.' Be concise and exact with numbers, doses and units."
)


def chunk_pages(pages: list[str], chunk_tokens: int, overlap: int) -> list[dict]:
    """Sliding-window chunks over the whole document, remembering which pages each spans."""
    if _ENC is not None:
        toks, owner = [], []
        for i, p in enumerate(pages):
            t = _ENC.encode(clean(p) + "\n", disallowed_special=())
            toks += t
            owner += [i + 1] * len(t)
        out, step = [], max(1, chunk_tokens - overlap)
        for s in range(0, max(1, len(toks)), step):
            seg = toks[s: s + chunk_tokens]
            if not seg:
                break
            out.append({"text": _ENC.decode(seg), "pages": sorted(set(owner[s: s + len(seg)])),
                        "tokens": len(seg)})
            if s + chunk_tokens >= len(toks):
                break
        return out
    # character fallback (no tiktoken): ~4 chars per token
    text, owner = "", []
    for i, p in enumerate(pages):
        c = clean(p) + "\n"
        text += c
        owner += [i + 1] * len(c)
    size, step = chunk_tokens * 4, max(1, (chunk_tokens - overlap) * 4)
    out = []
    for s in range(0, max(1, len(text)), step):
        seg = text[s: s + size]
        out.append({"text": seg, "pages": sorted(set(owner[s: s + len(seg)])), "tokens": count_tokens(seg)})
        if s + size >= len(text):
            break
    return out


class VectorStore:
    def __init__(self, dim: int | None = None):
        self.vecs: np.ndarray | None = None
        self.meta: list[dict] = []

    def add(self, vecs: np.ndarray, meta: list[dict]) -> None:
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-9)
        self.vecs = vecs.astype(np.float32) if self.vecs is None else np.vstack([self.vecs, vecs.astype(np.float32)])
        self.meta += meta
        self._doc_index = None

    def search(self, q: np.ndarray, k: int, doc_id: str | None = None) -> list[tuple[float, dict]]:
        q = q / max(np.linalg.norm(q), 1e-9)
        if doc_id:
            if self._doc_index is None:
                self._doc_index = {}
                for i, m in enumerate(self.meta):
                    self._doc_index.setdefault(m["doc_id"], []).append(i)
            idx = np.array(self._doc_index.get(doc_id, []), dtype=int)
            if idx.size == 0:
                return []
            sims = self.vecs[idx] @ q
            top = np.argsort(-sims)[:k]
            return [(float(sims[t]), self.meta[idx[t]]) for t in top]
        sims = self.vecs @ q
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        top = top[np.argsort(-sims[top])]
        return [(float(sims[t]), self.meta[t]) for t in top]

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vecs)
        (d / "chunks.jsonl").write_text("\n".join(json.dumps(m) for m in self.meta))

    @classmethod
    def load(cls, d: Path) -> "VectorStore":
        s = cls()
        s.vecs = np.load(d / "vectors.npy")
        s.meta = [json.loads(l) for l in (d / "chunks.jsonl").read_text().splitlines()]
        s._doc_index = None
        return s

    def nbytes(self) -> int:
        return int(self.vecs.nbytes) + sum(len(m["text"]) for m in self.meta)


class VectorRAG:
    name = "vector_rag"

    def __init__(self, cfg: dict, client, store_dir: Path):
        self.cfg, self.client, self.dir = cfg, client, store_dir
        v = cfg["vector_rag"]
        self.chunk_tokens, self.overlap, self.k, self.batch = v["chunk_tokens"], v["chunk_overlap"], v["top_k"], v["embed_batch"]
        self.embed_model, self.chat_model = cfg["models"]["embedding"], cfg["models"]["chat"]
        self.store = VectorStore.load(store_dir) if (store_dir / "vectors.npy").exists() else VectorStore()

    # ------------------------------------------------------------ indexing
    def _embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch):
            r = self.client.embeddings.create(model=self.embed_model, input=texts[i: i + self.batch])
            out += [d.embedding for d in r.data]
        return np.array(out, dtype=np.float32)

    def index_document(self, doc: dict) -> dict:
        t0 = time.perf_counter()
        pages = read_pdf_pages(doc["path"])
        t1 = time.perf_counter()
        chunks = chunk_pages(pages, self.chunk_tokens, self.overlap)
        t2 = time.perf_counter()
        vecs = self._embed([c["text"] for c in chunks])
        t3 = time.perf_counter()
        self.store.add(vecs, [{"doc_id": doc["doc_id"], "pages": c["pages"], "text": c["text"]} for c in chunks])
        return {"chunks": len(chunks), "parse_s": t1 - t0, "chunk_s": t2 - t1, "embed_s": t3 - t2,
                "storage_bytes": int(vecs.nbytes) + sum(len(c["text"]) for c in chunks)}

    def finalize_index(self) -> None:
        self.store.save(self.dir)

    # ------------------------------------------------------------ querying
    def query(self, question: str, doc_id: str | None) -> dict:
        t0 = time.perf_counter()
        qv = self._embed([question])[0]
        t1 = time.perf_counter()
        hits = self.store.search(qv, self.k, doc_id)
        t2 = time.perf_counter()
        context = "\n\n".join(f"[{m['doc_id']} p.{m['pages'][0]}"
                              + (f"-{m['pages'][-1]}" if len(m["pages"]) > 1 else "")
                              + f"]\n{m['text']}" for _, m in hits)
        kwargs = {}
        if self.cfg["models"].get("reasoning_effort"):
            kwargs["reasoning_effort"] = self.cfg["models"]["reasoning_effort"]
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "system", "content": ANSWER_SYSTEM},
                      {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"}],
            **kwargs)
        t3 = time.perf_counter()
        retrieved = sorted({(m["doc_id"], p) for _, m in hits for p in m["pages"]})
        return {"answer": resp.choices[0].message.content or "",
                "retrieved": [{"doc_id": d, "page": p} for d, p in retrieved],
                "context_tokens": count_tokens(context),
                "stage_s": {"embed_query": t1 - t0, "search": t2 - t1, "generate": t3 - t2},
                "turns": 1}
