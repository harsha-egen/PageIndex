"""pgvector store checks against a live Postgres (no OpenAI calls). Skips if no server.

    python -m tests.test_pgvector          # needs PGVECTOR_DSN or the local server on :5433
"""
from __future__ import annotations

import os
import uuid

import numpy as np

DSN = os.getenv("PGVECTOR_DSN", "postgresql://bench@127.0.0.1:5433/clinical_bench")


def _store():
    try:
        from bench.pipelines.pgvector_rag import PgVectorStore
        return PgVectorStore(DSN, table=f"test_{uuid.uuid4().hex[:8]}")
    except Exception as e:  # noqa: BLE001
        print(f"SKIP pgvector tests (no server at {DSN}: {type(e).__name__})")
        return None


def test_store_roundtrip_filter_and_hybrid(s):
    rng = np.random.default_rng(0)
    V = rng.normal(size=(400, 48)).astype(np.float32)
    meta = [{"doc_id": f"D{i % 20}", "pages": [i % 9 + 1, i % 9 + 2], "text": "renal dosing creatinine clearance"
             if i == 77 else f"unrelated chunk {i}"} for i in range(400)]
    s.add(V, meta)
    s.save()
    assert s.doc_ids() == {f"D{i}" for i in range(20)}
    top = s.search(V[5], 3)
    assert top[0][1]["text"] == "unrelated chunk 5" and abs(top[0][0] - 1) < 1e-4
    assert top[0][1]["pages"] == [6, 7]
    # document scope: a 5% slice of the table must still return k rows (iterative HNSW scan)
    rows = s.search(V[5], 8, doc_id="D3")
    assert len(rows) == 8 and {r[1]["doc_id"] for r in rows} == {"D3"}
    # hybrid: an exact keyword match is promoted even when its vector is unrelated
    hy = s.hybrid_search(V[200], "creatinine clearance renal", 5)
    assert any(r[1]["text"].startswith("renal dosing") for r in hy)
    assert s.hybrid_search(V[200], "creatinine", 5, doc_id="D17")[0][1]["doc_id"] == "D17"


def test_recall_vs_exact(s):
    """HNSW top-8 should match exact cosine top-8 almost perfectly at this size."""
    rng = np.random.default_rng(1)
    V = rng.normal(size=(3000, 64)).astype(np.float32)
    s.add(V, [{"doc_id": f"D{i % 50}", "pages": [1], "text": str(i)} for i in range(3000)])
    s.save()
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    rec = []
    for qi in range(0, 3000, 150):
        q = rng.normal(size=64).astype(np.float32)
        exact = set(np.argsort(-(Vn @ (q / np.linalg.norm(q))))[:8].tolist())
        got = {int(m["text"]) for _, m in s.search(q, 8)}
        rec.append(len(exact & got) / 8)
    assert np.mean(rec) >= 0.9, np.mean(rec)


if __name__ == "__main__":
    for fn in (test_store_roundtrip_filter_and_hybrid, test_recall_vs_exact):
        s = _store()
        if s is None:
            break
        try:
            fn(s)
            print("PASS", fn.__name__)
        finally:
            s.conn.execute(f"DROP TABLE IF EXISTS {s.table}")
