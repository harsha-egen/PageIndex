"""Offline tests (no API keys, no network): python -m pytest -q tests  (or python tests/test_offline.py)"""
import json, random, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from bench.common import load_config, read_pdf_pages, read_pdf_toc
from bench.data.pdf_render import render_pdf
from bench.data.synth_notes import generate_notes
from bench.evaluate import page_metrics
from bench.meter import UsageMeter, parse_usage
from bench.pipelines.vector_rag import VectorStore, chunk_pages
from bench.qa_gen import gen_for_note, _verify


def test_pdf_render_has_bookmarks():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.pdf"
        secs = [{"title": f"{i} SECTION {i}", "text": ("Lorem ipsum dolor. " * 120 + "\n") * 3,
                 "children": [{"title": f"{i}.1 Sub", "text": "Dose 10 mg daily. " * 80}]} for i in range(1, 6)]
        render_pdf(p, "Test label", secs)
        pages, toc = read_pdf_pages(p), read_pdf_toc(p)
        assert len(pages) >= 4 and len(toc) == 10 and toc[0][1].startswith("1 SECTION")


def test_notes_and_note_questions():
    cfg = load_config()
    with tempfile.TemporaryDirectory() as d:
        rows = generate_notes(cfg, Path(d), 5)
        assert len(rows) == 5 and all(r["pages"] >= 1 for r in rows)
        text = read_pdf_pages(rows[0]["path"])[0]
        q = gen_for_note(rows[0], random.Random(1))
        assert rows[0]["mrn"] in q["question"]
        # gold answer must literally appear in the note
        assert q["gold_answer"].replace(" mmHg", "").split()[0] in text


def test_chunks_track_pages():
    pages = [f"page {i} " + "word " * 900 for i in range(1, 6)]
    ch = chunk_pages(pages, 512, 64)
    assert ch[0]["pages"][0] == 1 and ch[-1]["pages"][-1] == 5
    assert all(p >= 1 for c in ch for p in c["pages"])


def test_vector_store_doc_filter():
    s = VectorStore()
    rng = np.random.default_rng(0)
    v = rng.normal(size=(30, 8)).astype(np.float32)
    s.add(v, [{"doc_id": "A" if i < 15 else "B", "pages": [i], "text": ""} for i in range(30)])
    hits = s.search(v[20], 3, doc_id="B")
    assert hits[0][1]["pages"] == [20] and all(m["doc_id"] == "B" for _, m in hits)
    assert s.search(v[3], 1)[0][1]["pages"] == [3]


def test_page_metrics():
    m = page_metrics([{"doc_id": "D", "page": 4}, {"doc_id": "D", "page": 9}], "D", [4, 12])
    assert m["page_hit"] == 1 and m["page_recall"] == 0.5 and m["wrong_doc"] == 0
    assert page_metrics([{"doc_id": "X", "page": 4}], "D", [4])["wrong_doc"] == 1


def test_verify_evidence():
    assert _verify(["Patients with eGFR < 30 mL/min"], ["... patients with eGFR <30 mL/min are excluded"])
    assert not _verify(["invented quote"], ["real page text"])


def _fake_httpx():
    """Minimal stand-in for httpx (Client.send + Response) when httpx isn't installed."""
    import types
    m = types.ModuleType("httpx")

    class URL:
        def __init__(self, u): self.path = "/" + u.split("://", 1)[1].split("/", 1)[1]

    class Request:
        def __init__(self, method, url, json_body=None):
            self.url, self.content = URL(url), json.dumps(json_body or {}).encode()

    class Response:
        def __init__(self, status_code, json=None):
            self.status_code, self._j, self.read_called = status_code, json, 0
        def read(self): self.read_called += 1
        def json(self): return self._j

    class Client:
        def __init__(self, transport=None): self.t = transport
        def send(self, request, **kw): return self.t(request)
        def post(self, url, json=None): return self.send(Request("POST", url, json))
        def get(self, url): return self.send(Request("GET", url))

    class AsyncClient:
        async def send(self, request, **kw): ...

    m.Client, m.AsyncClient, m.Response, m.MockTransport = Client, AsyncClient, Response, lambda h: h
    sys.modules["httpx"] = m
    return m


def test_usage_parsing_and_meter_http():
    try:
        import httpx
    except ImportError:
        httpx = _fake_httpx()
    u = parse_usage({"prompt_tokens": 1000, "completion_tokens": 200,
                     "prompt_tokens_details": {"cached_tokens": 400},
                     "completion_tokens_details": {"reasoning_tokens": 150}})
    assert u == {"input": 1000, "cached": 400, "output": 200, "reasoning": 150}
    pricing = {"gpt-5.6-terra": {"input": 2.0, "cached": 0.2, "output": 12.0}}
    meter = UsageMeter(pricing).install()

    def handler(request):
        body = json.loads(request.content or b"{}")
        if "model" not in body:
            return httpx.Response(200, json={"usage": {"prompt_tokens": 999}})
        return httpx.Response(200, json={"model": body["model"] + "-2026-07-09",
                                         "usage": {"prompt_tokens": 1000, "completion_tokens": 200,
                                                   "prompt_tokens_details": {"cached_tokens": 400}},
                                         "choices": [{"message": {"content": "ok"}}]})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    with meter.scope("t") as s:
        for _ in range(3):
            r = client.post("https://api.openai.com/v1/chat/completions", json={"model": "gpt-5.6-terra"})
            assert r.json()["choices"][0]["message"]["content"] == "ok"   # body still readable
        client.get("https://example.com/other")  # not metered
    summ = s.summary(pricing)
    assert summ["llm_calls"] == 3 and summ["input_tokens"] == 3000 and summ["cached_tokens"] == 1200
    expected = 3 * (600 * 2.0 + 400 * 0.2 + 200 * 12.0) / 1e6
    assert abs(summ["cost_usd"] - expected) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f(); print("PASS", f.__name__)
    print(f"{len(fns)} tests passed")
