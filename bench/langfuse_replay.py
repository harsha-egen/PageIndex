"""Replay a results folder into Langfuse, so dashboards can be built and rehearsed
before (or without) a live run.

    python -m bench.langfuse_replay --dir results/modeled            # environment "modeled"
    python -m bench.langfuse_replay --dir results/modeled --limit 200 # quick smoke test
    python -m bench.langfuse_replay --dir results --env benchmark     # re-send a measured run

Each query row becomes a trace "<pipeline>/<scope>" with ONE aggregated generation that
carries the row's tokens and cost, plus the same scores a live run writes (accuracy,
accuracy/<q_type>, page_hit, latency_s, llm_calls). Replayed traces have near-zero
duration, so chart LATENCY from the "latency_s" score for replayed data; live runs have
real trace latency as well.

Langfuse Cloud bills per unit (traces + observations + scores). The modeled set is about
2,320 traces x 7 units = ~16k units (plus ~5.6k with --with-index). Use --limit to sample.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from bench.common import ROOT, load_config
from bench.meter import Call
from bench.tracing import Tracer


def _num(v, default=0):
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "modeled"))
    ap.add_argument("--env", default=None, help="Langfuse environment (default: modeled if run_type=modeled)")
    ap.add_argument("--limit", type=int, help="replay a stratified sample of N query rows")
    ap.add_argument("--with-index", action="store_true", help="also replay per-document indexing rows")
    a = ap.parse_args()
    cfg = load_config()
    d = Path(a.dir)
    q = pd.read_csv(d / "query_metrics.csv")
    modeled = "run_type" in q and (q.run_type == "modeled").any()
    env = a.env or ("modeled" if modeled else "benchmark")
    tracer = Tracer(cfg, environment=env, run_name=f"{cfg['run_name']}-{env}")
    if not tracer.on:
        raise SystemExit("Langfuse is not configured: set keys in .env (see .env.example)")
    if a.limit:
        q = (q.sample(frac=1, random_state=0)
              .groupby(["pipeline", "scope", "doc_type"]).head(max(1, a.limit // 12)))
    chat_model, emb_model = cfg["models"]["chat"], cfg["models"]["embedding"]
    price = cfg["pricing"]
    n = 0
    for r in q.to_dict("records"):
        name = f"{r['pipeline']}/{r['scope']}"
        tags = [r["pipeline"], r["scope"], r["doc_type"], r["q_type"], "query", "modeled" if modeled else "measured"]
        with tracer.trace(name, name, tags, {"q_id": r["q_id"], "doc_id": r["doc_id"],
                                             "doc_pages": r.get("doc_pages"), "replayed": True},
                          input=r.get("question") or r["q_id"]) as span:
            tid = tracer.current_trace_id()
            call = Call("chat", chat_model, int(_num(r["input_tokens"])), int(_num(r.get("cached_tokens"))),
                        int(_num(r["output_tokens"])), int(_num(r.get("reasoning_tokens"))), 0.0)
            with tracer.generation("chat", chat_model) as g:
                usage, cost = tracer.usage_and_cost(call, price[chat_model])
                g.update(usage_details=usage, cost_details=cost,
                         metadata={"llm_calls_aggregated": _num(r.get("llm_calls"), 1),
                                   "note": "aggregate of all LLM calls in this query (replay)"})
            if _num(r.get("embed_tokens")) > 0:
                with tracer.generation("embeddings", emb_model) as g:
                    e = int(_num(r["embed_tokens"]))
                    g.update(usage_details={"input": e}, cost_details={"input": e * price[emb_model]["input"] / 1e6})
            if span is not None:
                span.update(output=r.get("answer") or "(modeled row)")
        for k in ("page_hit", "latency_s", "llm_calls"):
            tracer.score(tid, k, r.get(k))
        if r.get("judge_score") is not None and not pd.isna(r.get("judge_score")):
            tracer.score(tid, "accuracy", r["judge_score"])
            tracer.score(tid, f"accuracy/{r['q_type']}", r["judge_score"])
        n += 1
        if n % 200 == 0:
            print(f"  {n}/{len(q)} query traces")
    if a.with_index:
        i = pd.read_csv(d / "index_metrics.csv")
        for r in i.to_dict("records"):
            name = f"{r['pipeline']}/index"
            with tracer.trace(name, name, [r["pipeline"], "index", r["doc_type"]],
                              {"doc_id": r["doc_id"], "pages": r["pages"], "wall_s": r["wall_s"], "replayed": True}):
                tid = tracer.current_trace_id()
                with tracer.generation("embeddings" if r["pipeline"] == "vector_rag" else "chat",
                                       emb_model if r["pipeline"] == "vector_rag" else cfg["models"]["pageindex_index"]) as g:
                    g.update(usage_details={"input": int(_num(r.get("input_tokens")) + _num(r.get("embed_tokens"))),
                                            "output": int(_num(r.get("output_tokens")))},
                             cost_details={"total": _num(r["cost_usd"])})
            tracer.score(tid, "index_time_s", r["wall_s"])
    tracer.flush()
    print(f"replayed {n} query traces into Langfuse (environment={env}). Open Dashboards in Langfuse.")


if __name__ == "__main__":
    main()
