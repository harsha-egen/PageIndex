"""Step 3 — run both pipelines on the same corpus and questions, capturing every metric.

    python -m bench.run_benchmark index                    # build both indexes
    python -m bench.run_benchmark query                    # answer every question, both scopes
    python -m bench.run_benchmark judge                    # grade answers (blind LLM judge)
    python -m bench.run_benchmark all                      # all three
    options: --pipelines vector_rag pageindex  --limit 20  --scopes document

Everything is resumable: finished rows in results/*.jsonl are skipped on re-run.
Outputs: results/index_metrics.jsonl, results/query_metrics.jsonl (+ .csv copies).
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import pandas as pd

from bench.common import ROOT, load_config, load_manifest, read_jsonl
from bench.evaluate import judge, page_metrics
from bench.meter import UsageMeter
from bench.tracing import Tracer

RES = ROOT / "results"


def _append(path: Path, row: dict) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _done(path: Path, keys: tuple[str, ...]) -> set:
    if not path.exists():
        return set()
    return {tuple(r.get(k) for k in keys) for r in read_jsonl(path) if not r.get("error")}


def make_pipelines(cfg: dict, names: list[str]):
    from openai import OpenAI
    out = {}
    if "vector_rag" in names:
        from bench.pipelines.vector_rag import VectorRAG
        out["vector_rag"] = VectorRAG(cfg, OpenAI(), RES / "stores" / "vector")
    if "pageindex" in names:
        from bench.pipelines.pageindex_rag import PageIndexRAG
        out["pageindex"] = PageIndexRAG(cfg, RES / "stores" / "pageindex")
    return out


# ---------------------------------------------------------------------- index
def run_index(cfg, meter, pipes, limit=None, tracer=None):
    tracer = tracer or Tracer({"langfuse": {"enabled": False}})
    docs = load_manifest(cfg)[:limit] if limit else load_manifest(cfg)
    out = RES / "index_metrics.jsonl"
    for name, pipe in pipes.items():
        done = _done(out, ("pipeline", "doc_id"))
        todo = [d for d in docs if (name, d["doc_id"]) not in done]
        print(f"[index] {name}: {len(todo)} docs to index ({len(docs) - len(todo)} already done)")
        for i, d in enumerate(todo, 1):
            row = {"pipeline": name, "doc_id": d["doc_id"], "doc_type": d["doc_type"], "pages": d["pages"]}
            try:
                with tracer.trace(f"{name}/index", f"{name}/index",
                                  [name, "index", d["doc_type"], "measured"],
                                  {"doc_id": d["doc_id"], "pages": d["pages"], "doc_type": d["doc_type"]},
                                  input={"doc_id": d["doc_id"], "pages": d["pages"]}) as span, \
                        meter.scope(f"{name}:index:{d['doc_id']}") as s:
                    extra = pipe.index_document(d)
                    row["trace_id"] = tracer.current_trace_id()
                    if span is not None:
                        span.update(output={k: v for k, v in extra.items() if k != "skipped"})
                if extra.get("skipped"):
                    continue
                row.update(s.summary(meter.pricing))
                row.update(extra)
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            _append(out, row)
            if i % 10 == 0 or i == len(todo):
                print(f"  {name}: {i}/{len(todo)}  last={d['doc_id']} {row.get('wall_s', '?')}s ${row.get('cost_usd', 0):.4f}")
        pipe.finalize_index()


# ---------------------------------------------------------------------- query
def run_query(cfg, meter, pipes, scopes, limit=None, tracer=None):
    tracer = tracer or Tracer({"langfuse": {"enabled": False}})
    qs = read_jsonl(RES / "questions.jsonl")
    qs = qs[:limit] if limit else qs
    out = RES / "query_metrics.jsonl"
    for scope in scopes:
        for name, pipe in pipes.items():
            done = _done(out, ("pipeline", "scope", "q_id"))
            todo = [q for q in qs if (name, scope, q["q_id"]) not in done]
            print(f"[query] {name} / {scope}: {len(todo)} questions")
            for i, q in enumerate(todo, 1):
                row = {k: q[k] for k in ("q_id", "doc_id", "doc_type", "q_type", "doc_pages")}
                row.update({"pipeline": name, "scope": scope})
                try:
                    with tracer.trace(f"{name}/{scope}", f"{name}/{scope}",
                                      [name, scope, q["doc_type"], q["q_type"], "query", "measured"],
                                      {"q_id": q["q_id"], "doc_id": q["doc_id"], "doc_pages": q["doc_pages"],
                                       "q_type": q["q_type"], "doc_type": q["doc_type"]},
                                      input=q["question"]) as span, \
                            meter.scope(f"{name}:{scope}:{q['q_id']}") as s:
                        r = pipe.query(q["question"], q["doc_id"] if scope == "document" else None)
                        row["trace_id"] = tracer.current_trace_id()
                        if span is not None:
                            span.update(output={"answer": r["answer"], "retrieved": r["retrieved"][:20]})
                    summ = s.summary(meter.pricing)
                    # cross-check: if the HTTP meter saw no tokens (e.g. a non-httpx transport),
                    # fall back to the usage PageIndex reports for the run
                    pu = r.get("protocol_usage") or {}
                    if summ["input_tokens"] == 0 and pu:
                        summ["input_tokens"] = pu.get("prompt_tokens", 0)
                        summ["output_tokens"] = pu.get("completion_tokens", 0)
                        summ["usage_source"] = "pageindex_protocol"
                    row.update(summ)
                    row["latency_s"] = summ["wall_s"]
                    row["turns"] = r.get("turns") or summ["llm_calls"]
                    row["answer"] = r["answer"]
                    row["context_tokens"] = r.get("context_tokens")
                    for k, v in r.get("stage_s", {}).items():
                        row[f"t_{k}"] = round(v, 4)
                    row.update(page_metrics(r["retrieved"], q["doc_id"], q["gold_pages"]))
                    row["retrieved"] = r["retrieved"][:30]
                    for k in ("page_hit", "wrong_doc", "latency_s", "llm_calls"):
                        tracer.score(row.get("trace_id"), k, row.get(k))
                except Exception as e:  # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {e}"
                    traceback.print_exc()
                _append(out, row)
                if i % 10 == 0 or i == len(todo):
                    print(f"  {name}/{scope}: {i}/{len(todo)}  {row.get('latency_s', '?')}s  ${row.get('cost_usd', 0):.4f}")
    if meter.unscoped.calls:
        print(f"WARNING: {len(meter.unscoped.calls)} LLM calls happened outside any scope "
              "(likely a background thread). Their cost is reported in results/unscoped.json.")
    if meter.unknown_models:
        print(f"WARNING: no price configured for models {meter.unknown_models}; their cost counts as $0.")


# ---------------------------------------------------------------------- judge
def run_judge(cfg, meter, tracer=None):
    from openai import OpenAI
    client, model = OpenAI(), cfg["models"]["judge"]
    qs = {q["q_id"]: q for q in read_jsonl(RES / "questions.jsonl")}
    path = RES / "query_metrics.jsonl"
    rows = read_jsonl(path)
    todo = [r for r in rows if r.get("judge_score") is None and not r.get("error")]
    print(f"[judge] grading {len(todo)} answers with {model}")
    with meter.scope("judge") as s:
        for i, r in enumerate(todo, 1):
            q = qs[r["q_id"]]
            r.update(judge(client, model, q["question"], q["gold_answer"], r["answer"]))
            if tracer is not None and r.get("judge_score") is not None:
                tracer.score(r.get("trace_id"), "accuracy", r["judge_score"], r.get("judge_reason"))
                tracer.score(r.get("trace_id"), f"accuracy/{r['q_type']}", r["judge_score"])
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")
    import json
    path.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n")
    print(f"  judge cost (not charged to pipelines): ${s.summary(meter.pricing)['cost_usd']:.2f}")


def export_csv():
    for name in ("index_metrics", "query_metrics"):
        p = RES / f"{name}.jsonl"
        if p.exists():
            df = pd.DataFrame(read_jsonl(p))
            df.drop(columns=[c for c in ("retrieved",) if c in df], errors="ignore").to_csv(RES / f"{name}.csv", index=False)
            print(f"  wrote {RES / f'{name}.csv'} ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["index", "query", "judge", "all"])
    ap.add_argument("--pipelines", nargs="+", default=["vector_rag", "pageindex"])
    ap.add_argument("--scopes", nargs="+")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    cfg = load_config()
    meter = UsageMeter(cfg["pricing"]).install()
    tracer = Tracer(cfg, environment=cfg.get("langfuse", {}).get("environment", "benchmark"))
    meter.tracer = tracer
    t = time.time()
    if args.step in ("index", "query", "all"):
        pipes = make_pipelines(cfg, args.pipelines)
    if args.step in ("index", "all"):
        run_index(cfg, meter, pipes, args.limit, tracer)
    if args.step in ("query", "all"):
        run_query(cfg, meter, pipes, args.scopes or cfg["benchmark"]["scopes"], args.limit, tracer)
    if args.step in ("judge", "all"):
        meter.tracer = None          # judge calls are graded separately, not pipeline cost
        run_judge(cfg, meter, tracer)
    tracer.flush()
    export_csv()
    print(f"done in {time.time() - t:.0f}s. Next: python -m bench.report")


if __name__ == "__main__":
    main()
