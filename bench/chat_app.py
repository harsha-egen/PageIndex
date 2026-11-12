"""Side-by-side chat demo: ask one question, watch PageIndex and vector RAG answer it live.

    python -m bench.chat_app              # http://127.0.0.1:8765
    python -m bench.chat_app --fake       # UI rehearsal with simulated pipelines (no API calls, $0)
    options: --port 8765  --no-browser  --vector vector_pg|vector_pg_hybrid

The vector column runs on Postgres + pgvector (`vector_pg`); --vector vector_pg_hybrid adds
Postgres full-text search fused with RRF.

Both pipelines run in parallel on every question. Each OpenAI call they make is intercepted
by the same UsageMeter the benchmark uses and streamed to the browser as it happens (model,
tokens, latency, cost, and the tools PageIndex's agent asked for). With Langfuse keys in
.env, each answer is also a Langfuse trace in environment "demo", linked from the UI.

The demo uses the indexes built by `python -m bench.run_benchmark index`, so documents you
index for the demo are reused (not re-paid) by the full benchmark run.
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bench.common import ROOT, load_config, manifest_path, read_jsonl
from bench.meter import Call, UsageMeter, _current, cost_of
from bench.tracing import Tracer

UI = Path(__file__).with_name("chat_ui.html")
DEMO_QUESTIONS = ROOT / "docs" / "demo_questions.yaml"
STORES = ROOT / "results" / "stores"
PIPES = ("pageindex", "vector_rag")   # UI columns; the "vector_rag" column runs self.vector
VECTOR_LABELS = {"vector_pg": "pgvector HNSW", "vector_pg_hybrid": "pgvector + full-text (RRF)"}


# --------------------------------------------------------------------------- demo core
class ChatDemo:
    def __init__(self, cfg: dict, fake: bool = False, vector: str | None = None):
        self.cfg, self.fake = cfg, fake
        self.vector = vector or cfg.get("benchmark", {}).get("vector_baseline", "vector_pg")
        if self.vector not in VECTOR_LABELS:
            self.vector = "vector_pg"
        self.meter = UsageMeter(cfg["pricing"]).install()
        if fake:
            self.tracer = Tracer({"langfuse": {"enabled": False}})
        else:
            self.tracer = Tracer(cfg, environment="demo", run_name=f"{cfg.get('run_name', 'clinical-bench')}-demo")
        self.meter.tracer = self.tracer
        self.locks = {p: threading.Lock() for p in PIPES}   # one question at a time per pipeline
        self.pipes: dict = {}
        self.errors: dict[str, str] = {}
        self._load_pipelines()

    def _load_pipelines(self) -> None:
        if self.fake:
            self.pipes = {"pageindex": FakePipeline("pageindex", self.cfg),
                          "vector_rag": FakePipeline("vector_rag", self.cfg)}
            return
        try:
            from openai import OpenAI
            from bench.pipelines.pgvector_rag import PgVectorRAG
            self.pipes["vector_rag"] = PgVectorRAG(self.cfg, OpenAI(), hybrid=self.vector == "vector_pg_hybrid")
        except Exception as e:  # noqa: BLE001
            self.errors["vector_rag"] = f"{type(e).__name__}: {e} (is the pgvector server running? see README 'pgvector')"
        try:
            from bench.pipelines.pageindex_rag import PageIndexRAG
            self.pipes["pageindex"] = PageIndexRAG(self.cfg, STORES / "pageindex")
        except Exception as e:  # noqa: BLE001
            self.errors["pageindex"] = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ status
    def indexed_ids(self) -> dict[str, set]:
        out = {p: set() for p in PIPES}
        v = self.pipes.get("vector_rag")
        if v is not None:
            st = v.store
            out["vector_rag"] = st.doc_ids() if hasattr(st, "doc_ids") else {m["doc_id"] for m in (st.meta or [])}
        pi = self.pipes.get("pageindex")
        if pi is not None:
            out["pageindex"] = set(pi.doc_map)
        return out

    def docs(self) -> list[dict]:
        ids = self.indexed_ids()
        both_or_any = ids["vector_rag"] | ids["pageindex"]
        manifest = dict(FAKE_MANIFEST) if self.fake else {}
        if not self.fake and manifest_path(self.cfg).exists():
            manifest = {r["doc_id"]: r for r in read_jsonl(manifest_path(self.cfg))}
        out = []
        for d in sorted(both_or_any):
            m = manifest.get(d, {})
            out.append({"doc_id": d, "title": m.get("title") or d, "doc_type": m.get("doc_type", "other"),
                        "pages": m.get("pages"),
                        "indexed": {p: d in ids[p] for p in PIPES}})
        return out

    def status(self) -> dict:
        lf = self.tracer
        base = None
        if lf.on:
            import os
            base = os.getenv("LANGFUSE_BASE_URL")
        docs = self.docs()
        return {"fake": self.fake,
                "demo_questions": self.demo_questions(docs),
                "vector": {"pipeline": self.vector, "label": VECTOR_LABELS.get(self.vector, self.vector)},
                "models": {"chat": self.cfg["models"]["chat"], "embedding": self.cfg["models"]["embedding"],
                           "pageindex_index": self.cfg["models"]["pageindex_index"],
                           "reasoning_effort": self.cfg["models"].get("reasoning_effort")},
                "top_k": self.cfg["vector_rag"]["top_k"],
                "langfuse": {"on": lf.on, "base_url": base, "environment": "demo"},
                "pipelines": {p: {"ready": p in self.pipes, "error": self.errors.get(p)} for p in PIPES},
                "docs": docs,
                "both_indexed": sum(1 for d in docs if all(d["indexed"].values())),
                "manifest": manifest_path(self.cfg).exists()}

    def demo_questions(self, docs: list[dict]) -> list[dict]:
        """Questions from docs/demo_questions.yaml whose document is indexed by both pipelines."""
        if self.fake or not DEMO_QUESTIONS.exists():
            return []
        import yaml
        ready = {d["doc_id"] for d in docs if all(d["indexed"].values())}
        try:
            qs = yaml.safe_load(DEMO_QUESTIONS.read_text()) or []
        except Exception:  # noqa: BLE001
            return []
        return [{"id": q["id"], "act": q.get("act"), "doc": q.get("doc"), "q": q["q"]}
                for q in qs if q.get("doc") is None or q.get("doc") in ready]

    # ------------------------------------------------------------------ asking
    def _call_dict(self, c: Call) -> dict:
        return {"kind": c.kind, "model": c.model, "input": c.input, "cached": c.cached, "output": c.output,
                "reasoning": c.reasoning, "latency_s": round(c.latency_s, 3), "ok": c.ok,
                "cost_usd": round(cost_of(c, self.meter.pricing), 6), "tools": c.tools}

    def _pipe_name(self, name: str) -> str:
        return self.vector if name == "vector_rag" else name

    def run_one(self, name: str, question: str, doc_id: str | None, emit) -> None:
        pipe = self.pipes.get(name)
        if pipe is None:
            emit({"type": "error", "pipeline": name, "message": self.errors.get(name, "pipeline not available")})
            return
        if doc_id and doc_id not in self.indexed_ids()[name]:
            emit({"type": "error", "pipeline": name,
                  "message": f"{doc_id} is not indexed for {name}. Run: python -m bench.run_benchmark index"})
            return
        scope = "document" if doc_id else "corpus"
        emit({"type": "start", "pipeline": name, "scope": scope})
        pricing = self.meter.pricing
        try:
            with self.locks[name]:
                pn = self._pipe_name(name)
                with self.tracer.trace(f"{pn}/{scope}", f"{pn}/{scope}", [self._pipe_name(name), scope, "demo", "chat"],
                                       {"doc_id": doc_id or "", "source": "chat_demo"}, input=question) as span, \
                        self.meter.scope(f"demo:{name}") as s:
                    s.on_call = lambda c: emit({"type": "call", "pipeline": name, "call": self._call_dict(c),
                                                "totals": s.summary(pricing)})
                    r = pipe.query(question, doc_id)
                    tid = self.tracer.current_trace_id()
                    if span is not None:
                        span.update(output={"answer": r["answer"], "retrieved": r["retrieved"][:20]})
            summ = s.summary(pricing)
            pu = r.get("protocol_usage") or {}
            if summ["input_tokens"] == 0 and pu:     # same fallback as the benchmark runner
                summ["input_tokens"] = pu.get("prompt_tokens", 0)
                summ["output_tokens"] = pu.get("completion_tokens", 0)
                summ["usage_source"] = "pageindex_protocol"
            for k, v in (("latency_s", summ["wall_s"]), ("llm_calls", summ["llm_calls"])):
                self.tracer.score(tid, k, v)
            url = None
            if self.tracer.on and tid:
                try:
                    url = self.tracer.lf.get_trace_url(trace_id=tid)
                except Exception:
                    url = None
            emit({"type": "done", "pipeline": name, "answer": r["answer"], "retrieved": r["retrieved"][:30],
                  "summary": summ, "stage_s": {k: round(v, 3) for k, v in (r.get("stage_s") or {}).items()},
                  "context_tokens": r.get("context_tokens"), "trace_url": url})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            emit({"type": "error", "pipeline": name, "message": f"{type(e).__name__}: {e}"})

    def ask(self, question: str, doc_id: str | None):
        """Yield events from both pipelines as they happen."""
        q: queue.Queue = queue.Queue()
        threads = [threading.Thread(target=self.run_one, args=(n, question, doc_id, q.put), daemon=True)
                   for n in PIPES]
        for t in threads:
            t.start()
        remaining = len(threads)
        while remaining:
            ev = q.get()
            if ev["type"] in ("done", "error"):
                remaining -= 1
            yield ev
        threading.Thread(target=self.tracer.flush, daemon=True).start()
        yield {"type": "end"}


# --------------------------------------------------------------------------- fake mode
FAKE_MANIFEST = {
    "DEMO-LABEL-1": {"title": "Simulated prescribing information (demo)", "doc_type": "fda_label", "pages": 32},
    "DEMO-TRIAL-1": {"title": "Simulated phase 3 protocol (demo)", "doc_type": "trial_protocol", "pages": 118},
}


class FakePipeline:
    """Simulated pipeline for --fake: same event shape, no network, $0."""

    def __init__(self, name: str, cfg: dict):
        self.name, self.cfg = name, cfg
        self.store = type("S", (), {"meta": [{"doc_id": "DEMO-LABEL-1"}, {"doc_id": "DEMO-TRIAL-1"}]})()
        self.doc_map = {"DEMO-LABEL-1": "x", "DEMO-TRIAL-1": "y"}

    def _call(self, kind, model, inp, out, latency, cached=0, reasoning=0, tools=()):
        time.sleep(latency)
        s = _current.get()
        if s is not None:
            s.add(Call(kind, model, inp, cached, out, reasoning, latency, True, list(tools)))

    def query(self, question: str, doc_id: str | None) -> dict:
        m = self.cfg["models"]
        doc = doc_id or "DEMO-LABEL-1"
        if self.name == "vector_rag":
            self._call("embeddings", m["embedding"], 18, 0, 0.3)
            self._call("chat", m["chat"], 4300, 260, 2.2, reasoning=120)
            return {"answer": f"(simulated) The recommended starting dose is 10 mg once daily [{doc} p.4]. "
                              f"Reduce to 5 mg in severe renal impairment [{doc} p.9].",
                    "retrieved": [{"doc_id": doc, "page": p} for p in (3, 4, 9, 12)],
                    "context_tokens": 4100, "stage_s": {"embed_query": 0.3, "search": 0.001, "generate": 2.2},
                    "turns": 1}
        steps = [("get_document_structure", "{}"), ("get_page_content", '{"pages": "4-5"}'),
                 ("get_page_content", '{"pages": "9"}'), (None, None)]
        inp = 1800
        for i, (tool, args) in enumerate(steps):
            self._call("chat", m["chat"], inp, 90 if tool else 240, 1.4, cached=max(0, inp - 1200) if i else 0,
                       reasoning=60, tools=[{"name": tool, "args": args}] if tool else [])
            inp += 1500
        return {"answer": "(simulated) **Starting dose:** 10 mg orally once daily (section 2.1).\n\n"
                          "- Severe renal impairment (CrCl < 30 mL/min): 5 mg once daily (section 2.3).",
                "retrieved": [{"doc_id": doc, "page": 4}, {"doc_id": doc, "page": 9}],
                "stage_s": {"agent_total": 5.6}, "turns": None}


# --------------------------------------------------------------------------- HTTP
def make_handler(demo: ChatDemo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter console
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                b = UI.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif self.path == "/api/status":
                self._json(demo.status())
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/api/ask":
                return self._json({"error": "not found"}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except ValueError:
                return self._json({"error": "bad json"}, 400)
            question = str(body.get("question") or "").strip()
            if not question:
                return self._json({"error": "empty question"}, 400)
            doc_id = body.get("doc_id") or None
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for ev in demo.ask(question, doc_id):
                    self.wfile.write((json.dumps(ev, default=str) + "\n").encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--fake", action="store_true", help="simulated pipelines, no API calls")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--vector", choices=list(VECTOR_LABELS), help="vector pipeline for the right-hand column")
    args = ap.parse_args()
    demo = ChatDemo(load_config(), fake=args.fake, vector=args.vector)
    st = demo.status()
    for p, v in st["pipelines"].items():
        label = f"{p} ({st['vector']['label']})" if p == "vector_rag" else p
        print(f"  {label:36s} {'ready' if v['ready'] else 'NOT READY: ' + str(v['error'])}")
    print(f"  documents indexed by both pipelines: {st['both_indexed']}")
    url = f"http://127.0.0.1:{args.port}"
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(demo))
    print(f"Chat demo on {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        demo.tracer.flush()


if __name__ == "__main__":
    main()
