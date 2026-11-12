"""Vector RAG on Postgres + pgvector (what most teams deploy), as a drop-in for VectorRAG.

Same chunking, embedding model, prompt and answering model as `vector_rag.py`; only the
store changes:

* `vector_pg`         pgvector HNSW index, cosine distance (approximate nearest neighbours)
* `vector_pg_hybrid`  the same + Postgres full-text search (tsvector), merged with
                      reciprocal rank fusion (RRF). Keyword matches rescue exact terms
                      (drug names, "EXP-4", "CrCl < 30") that embeddings blur.

Document scope is a `WHERE doc_id = ...` filter. With HNSW, plain post-filtering can return
fewer than k rows when the document is a small part of the table; pgvector >= 0.8's
iterative index scans (`hnsw.iterative_scan`) fix that, and we enable them.

Connection: env PGVECTOR_DSN, else config `vector_rag.pgvector.dsn`.
Local server for this repo: see "pgvector" in README.md (port 5433).
"""
from __future__ import annotations

import os
import time

import numpy as np

from bench.common import count_tokens, read_pdf_pages
from bench.pipelines.vector_rag import ANSWER_SYSTEM, VectorRAG, chunk_pages


class PgVectorStore:
    """Chunk store with the same interface as VectorStore: add / search / save / doc_ids / nbytes."""

    def __init__(self, dsn: str, table: str = "chunks", ef_search: int = 100, m: int = 16,
                 ef_construction: int = 64, iterative_scan: str = "relaxed_order"):
        import psycopg
        from pgvector.psycopg import register_vector
        self.table, self.ef_search, self.iterative = table, ef_search, iterative_scan
        self.m, self.efc = m, ef_construction
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.dim = self._existing_dim()

    # ----------------------------------------------------------------- schema
    def _existing_dim(self) -> int | None:
        row = self.conn.execute(
            "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
            "WHERE a.attrelid = to_regclass(%s) AND a.attname = 'embedding'", (self.table,)).fetchone()
        return int(row[0].split("(")[1].rstrip(")")) if row else None

    def _ensure_table(self, dim: int) -> None:
        if self.dim == dim:
            return
        if self.dim is not None:
            raise ValueError(f"table {self.table} has dim {self.dim}, got {dim}")
        t = self.table
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {t} (
                id        bigserial PRIMARY KEY,
                doc_id    text NOT NULL,
                pages     int[] NOT NULL,
                text      text NOT NULL,
                tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
                embedding vector({dim}) NOT NULL)""")
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_doc_idx ON {t} (doc_id)")
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_tsv_idx ON {t} USING gin (tsv)")
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_hnsw_idx ON {t} USING hnsw (embedding vector_cosine_ops) "
                          f"WITH (m = {self.m}, ef_construction = {self.efc})")
        self.dim = dim

    # ----------------------------------------------------------------- write
    def add(self, vecs: np.ndarray, meta: list[dict]) -> None:
        if not len(meta):
            return
        self._ensure_table(int(vecs.shape[1]))
        with self.conn.cursor() as cur, cur.copy(
                f"COPY {self.table} (doc_id, pages, text, embedding) FROM STDIN WITH (FORMAT BINARY)") as cp:
            cp.set_types(["text", "int4[]", "text", "vector"])
            for v, m in zip(vecs.astype(np.float32), meta):
                cp.write_row((m["doc_id"], list(m["pages"]), m["text"], v))

    def delete_doc(self, doc_id: str) -> None:
        if self.dim is not None:
            self.conn.execute(f"DELETE FROM {self.table} WHERE doc_id = %s", (doc_id,))

    def save(self, d=None) -> None:  # rows are durable on insert; kept for interface parity
        if self.dim is not None:
            self.conn.execute(f"ANALYZE {self.table}")

    # ----------------------------------------------------------------- read
    def _settings(self, cur) -> None:
        cur.execute(f"SET LOCAL hnsw.ef_search = {int(self.ef_search)}")
        if self.iterative:
            cur.execute(f"SET LOCAL hnsw.iterative_scan = {self.iterative}")

    def search(self, q: np.ndarray, k: int, doc_id: str | None = None) -> list[tuple[float, dict]]:
        if self.dim is None:
            return []
        where, params = ("WHERE doc_id = %(doc)s", {"doc": doc_id}) if doc_id else ("", {})
        params.update(q=np.asarray(q, dtype=np.float32), k=k)
        sql = (f"SELECT doc_id, pages, text, 1 - (embedding <=> %(q)s) AS sim FROM {self.table} {where} "
               f"ORDER BY embedding <=> %(q)s LIMIT %(k)s")
        with self.conn.transaction(), self.conn.cursor() as cur:
            self._settings(cur)
            rows = cur.execute(sql, params).fetchall()
        return [(float(s), {"doc_id": d, "pages": list(p), "text": t}) for d, p, t, s in rows]

    def hybrid_search(self, q: np.ndarray, query_text: str, k: int, doc_id: str | None = None,
                      pool: int = 40, rrf_k: int = 60) -> list[tuple[float, dict]]:
        """Reciprocal rank fusion of pgvector cosine ranks and full-text ranks (OR of question terms)."""
        if self.dim is None:
            return []
        filt = "AND doc_id = %(doc)s" if doc_id else ""
        where = "WHERE doc_id = %(doc)s" if doc_id else ""
        sql = f"""
        WITH sem AS (
            SELECT id, row_number() OVER (ORDER BY embedding <=> %(q)s) AS r
            FROM (SELECT id, embedding FROM {self.table} {where}
                  ORDER BY embedding <=> %(q)s LIMIT %(pool)s) s),
        kw AS (
            SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) AS r
            FROM {self.table},
                 -- OR the question's lexemes (AND would demand every word in one chunk)
                 CAST(NULLIF(replace(CAST(plainto_tsquery('english', %(text)s) AS text), '&', '|'), '')
                      AS tsquery) query
            WHERE tsv @@ query {filt}
            ORDER BY ts_rank_cd(tsv, query) DESC LIMIT %(pool)s),
        fused AS (
            SELECT id, sum(1.0 / (%(rrf)s + r)) AS score FROM (
                SELECT * FROM sem UNION ALL SELECT * FROM kw) u GROUP BY id)
        SELECT c.doc_id, c.pages, c.text, f.score
        FROM fused f JOIN {self.table} c USING (id)
        ORDER BY f.score DESC LIMIT %(k)s"""
        params = {"q": np.asarray(q, dtype=np.float32), "text": query_text, "k": k, "pool": pool,
                  "rrf": rrf_k, "doc": doc_id}
        with self.conn.transaction(), self.conn.cursor() as cur:
            self._settings(cur)
            rows = cur.execute(sql, params).fetchall()
        return [(float(s), {"doc_id": d, "pages": list(p), "text": t}) for d, p, t, s in rows]

    # ----------------------------------------------------------------- stats
    def doc_ids(self) -> set[str]:
        if self.dim is None:
            return set()
        return {r[0] for r in self.conn.execute(f"SELECT DISTINCT doc_id FROM {self.table}")}

    def nbytes(self) -> int:
        if self.dim is None:
            return 0
        return int(self.conn.execute("SELECT pg_total_relation_size(%s)", (self.table,)).fetchone()[0])


def pg_dsn(cfg: dict) -> str:
    return os.getenv("PGVECTOR_DSN") or cfg["vector_rag"].get("pgvector", {}).get(
        "dsn", "postgresql://bench@127.0.0.1:5433/clinical_bench")


class PgVectorRAG(VectorRAG):
    """VectorRAG with a pgvector store. hybrid=True adds full-text search fused with RRF."""

    def __init__(self, cfg: dict, client, hybrid: bool = False):
        # VectorRAG.__init__ would load the in-memory store; set the fields we need instead.
        self.cfg, self.client, self.dir = cfg, client, None
        v = cfg["vector_rag"]
        self.chunk_tokens, self.overlap, self.k, self.batch = v["chunk_tokens"], v["chunk_overlap"], v["top_k"], v["embed_batch"]
        self.embed_model, self.chat_model = cfg["models"]["embedding"], cfg["models"]["chat"]
        p = v.get("pgvector", {})
        self.hybrid = hybrid
        self.name = "vector_pg_hybrid" if hybrid else "vector_pg"
        self.store = PgVectorStore(pg_dsn(cfg), table=p.get("table", "chunks"), ef_search=p.get("ef_search", 100),
                                   m=p.get("m", 16), ef_construction=p.get("ef_construction", 64),
                                   iterative_scan=p.get("iterative_scan", "relaxed_order"))

    def index_document(self, doc: dict) -> dict:
        if self.hybrid:   # hybrid reads the same table as vector_pg; nothing extra to index
            return {"skipped": True}
        if doc["doc_id"] in self.store.doc_ids():
            return {"skipped": True}
        t0 = time.perf_counter()
        pages = read_pdf_pages(doc["path"])
        chunks = chunk_pages(pages, self.chunk_tokens, self.overlap)
        t1 = time.perf_counter()
        vecs = self._embed([c["text"] for c in chunks])
        t2 = time.perf_counter()
        self.store.add(vecs, [{"doc_id": doc["doc_id"], "pages": c["pages"], "text": c["text"]} for c in chunks])
        t3 = time.perf_counter()
        return {"chunks": len(chunks), "parse_s": t1 - t0, "embed_s": t2 - t1, "insert_s": t3 - t2,
                "storage_bytes": None}

    def finalize_index(self) -> None:
        self.store.save()

    def query(self, question: str, doc_id: str | None) -> dict:
        t0 = time.perf_counter()
        qv = self._embed([question])[0]
        t1 = time.perf_counter()
        if self.hybrid:
            hits = self.store.hybrid_search(qv, question, self.k, doc_id)
        else:
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
                "context_tokens": count_tokens(context), "chunks_returned": len(hits),
                "stage_s": {"embed_query": t1 - t0, "search": t2 - t1, "generate": t3 - t2},
                "turns": 1}
