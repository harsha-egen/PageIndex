"""Run the demo questions (docs/demo_questions.yaml) through several pipelines and grade them.

    python -m bench.demo_compare                                              # pgvector + PageIndex, ~$1
    python -m bench.demo_compare --pipelines pageindex --ids 12 13            # a subset
    python -m bench.demo_compare --judge                                      # add LLM-judge grades

Each answer is metered exactly like the benchmark (same UsageMeter), so latency, tokens and
cost are comparable across pipelines. Results append to results/demo_compare.jsonl (one row
per question x pipeline x run) and print as a table. --judge grades each answer against the
gold answer with the benchmark's blind judge (config models.judge; not charged to pipelines).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from bench.common import ROOT, load_config, read_jsonl
from bench.meter import UsageMeter
from bench.run_benchmark import make_pipelines

QFILE = ROOT / "docs" / "demo_questions.yaml"
OUT = ROOT / "results" / "demo_compare.jsonl"


def run(cfg, pipes, questions, meter, run_id) -> list[dict]:
    rows = []
    for q in questions:
        for name, pipe in pipes.items():
            row = {"run": run_id, "id": q["id"], "act": q["act"], "doc": q["doc"], "pipeline": name}
            try:
                with meter.scope(f"demo:{name}:{q['id']}") as s:
                    r = pipe.query(q["q"], q["doc"])
                summ = s.summary(meter.pricing)
                pu = r.get("protocol_usage") or {}
                if summ["input_tokens"] == 0 and pu:
                    summ["input_tokens"], summ["output_tokens"] = pu.get("prompt_tokens", 0), pu.get("completion_tokens", 0)
                row.update(latency_s=round(summ["wall_s"], 3), cost_usd=summ["cost_usd"], llm_calls=summ["llm_calls"],
                           input_tokens=summ["input_tokens"], output_tokens=summ["output_tokens"],
                           search_s=round((r.get("stage_s") or {}).get("search", float("nan")), 4),
                           answer=r["answer"], pages=[[x["doc_id"], x["page"]] for x in r["retrieved"]][:40])
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            print(f"  #{q['id']:>2} {name:17s} {row.get('latency_s', '-'):>6}s ${row.get('cost_usd', 0):.4f}"
                  + (f"  ERROR {row['error']}" if "error" in row else ""), flush=True)
    return rows


def run_via_app(url: str, questions, run_id) -> list[dict]:
    """Send each question through a running chat app (POST /api/ask), exactly as the browser does.

    Checks that every question works end to end in the demo: its document is indexed by both
    pipelines, both stream a 'done' event (no 'error'), and neither answers "Not found" unless
    the gold answer says the information is absent.
    """
    import urllib.request
    status = json.load(urllib.request.urlopen(url + "/api/status", timeout=30))
    vec = status.get("vector", {}).get("pipeline", "vector_rag")
    ready = {d["doc_id"] for d in status["docs"] if all(d["indexed"].values())}
    rows = []
    for q in questions:
        base = {"run": run_id, "id": q["id"], "act": q["act"], "doc": q["doc"], "via": "app"}
        if q["doc"] and q["doc"] not in ready:
            for p in ("pageindex", vec):
                rows.append({**base, "pipeline": p, "error": f"{q['doc']} not indexed by both pipelines"})
            continue
        req = urllib.request.Request(url + "/api/ask", data=json.dumps({"question": q["q"], "doc_id": q["doc"]}).encode(),
                                     headers={"Content-Type": "application/json"})
        got = {}
        for line in urllib.request.urlopen(req, timeout=600):
            if not line.strip():
                continue
            ev = json.loads(line)
            p = vec if ev.get("pipeline") == "vector_rag" else ev.get("pipeline")
            if ev["type"] == "done":
                s = ev["summary"]
                got[p] = {**base, "pipeline": p, "latency_s": round(s["wall_s"], 3), "cost_usd": s["cost_usd"],
                          "llm_calls": s["llm_calls"], "input_tokens": s["input_tokens"],
                          "output_tokens": s["output_tokens"], "answer": ev["answer"],
                          "pages": [[x["doc_id"], x["page"]] for x in ev["retrieved"]][:40]}
            elif ev["type"] == "error":
                got[p] = {**base, "pipeline": p, "error": ev["message"]}
        for p in ("pageindex", vec):
            r = got.get(p, {**base, "pipeline": p, "error": "no result streamed"})
            absent_ok = "not in the label" in q["gold"].lower()
            if "answer" in r and r["answer"].lower().lstrip().startswith("not found") and not absent_ok:
                r["app_flag"] = "answered 'Not found'"
            rows.append(r)
            print(f"  #{q['id']:>2} {p:17s} {r.get('latency_s', '-'):>6}s ${r.get('cost_usd', 0):.4f}"
                  + (f"  ERROR {r['error']}" if "error" in r else "") + (f"  FLAG {r['app_flag']}" if "app_flag" in r else ""),
                  flush=True)
    return rows


def judge_rows(cfg, rows, questions, meter) -> None:
    from openai import OpenAI
    from bench.evaluate import judge
    client, model = OpenAI(), cfg["models"]["judge"]
    qs = {q["id"]: q for q in questions}
    with meter.scope("judge") as s:
        for r in rows:
            if "answer" in r:
                r.update(judge(client, model, qs[r["id"]]["q"], qs[r["id"]]["gold"], r["answer"]))
    print(f"  judge cost (not charged to pipelines): ${s.summary(meter.pricing)['cost_usd']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipelines", nargs="+", default=["vector_pg", "pageindex"],
                    choices=["vector_pg", "vector_pg_hybrid", "pageindex"])
    ap.add_argument("--ids", nargs="+", type=int)
    ap.add_argument("--questions", default=str(QFILE), help="YAML question file (default: docs/demo_questions.yaml)")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--app", metavar="URL", help="send questions through a running chat app, e.g. http://127.0.0.1:8767")
    a = ap.parse_args()
    from bench.tracing import load_env
    load_env()
    cfg = load_config()
    questions = [q for q in yaml.safe_load(Path(a.questions).read_text()) if not a.ids or q["id"] in a.ids]
    meter = UsageMeter(cfg["pricing"]).install()
    run_id = time.strftime("%Y%m%dT%H%M%S")
    if a.app:
        print(f"[demo_compare] {len(questions)} questions via app {a.app}  run={run_id}")
        rows = run_via_app(a.app.rstrip("/"), questions, run_id)
    else:
        pipes = make_pipelines(cfg, a.pipelines)
        print(f"[demo_compare] {len(questions)} questions x {list(pipes)}  run={run_id}")
        rows = run(cfg, pipes, questions, meter, run_id)
    if a.judge:
        judge_rows(cfg, rows, questions, meter)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    tot = {}
    for r in rows:
        t = tot.setdefault(r["pipeline"], {"n": 0, "cost": 0.0, "lat": 0.0, "score": 0.0, "graded": 0})
        if "error" in r:
            continue
        t["n"] += 1; t["cost"] += r["cost_usd"]; t["lat"] += r["latency_s"]
        if r.get("judge_score") is not None:
            t["score"] += r["judge_score"]; t["graded"] += 1
    print("\npipeline           answers  mean latency  total cost  judge")
    for p, t in tot.items():
        acc = f"{t['score'] / t['graded']:.0%}" if t["graded"] else "-"
        print(f"{p:18s} {t['n']:>7}  {t['lat'] / max(t['n'], 1):>10.2f}s  ${t['cost']:>9.4f}  {acc:>5}")
    flags = [r for r in rows if "error" in r or "app_flag" in r]
    if a.app:
        print(f"app check: {len(flags)} problem(s)" + "".join(
            f"\n  #{r['id']} {r['pipeline']}: {r.get('error') or r.get('app_flag')}" for r in flags))
    print(f"rows appended to {OUT}")


if __name__ == "__main__":
    main()
