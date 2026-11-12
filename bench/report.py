"""Step 4 — turn results into a self-contained HTML dashboard (Plotly via CDN).

    python -m bench.report                      # measured results in results/
    python -m bench.report --dir results/modeled --out results/report_modeled.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bench.common import ROOT

PIPE = {"vector_rag": "Vector RAG", "pageindex": "PageIndex"}


def _pipe_labels(cfg_path: Path = ROOT / "config.yaml") -> dict:
    """The report compares two pipelines; config benchmark.vector_baseline picks the vector one."""
    try:
        import yaml
        base = yaml.safe_load(cfg_path.read_text())["benchmark"].get("vector_baseline", "vector_rag")
    except Exception:
        base = "vector_rag"
    return {base: "Vector RAG", "pageindex": "PageIndex"}
COL = {"Vector RAG": "#2f6fdb", "PageIndex": "#e0752d"}
DT = {"trial_protocol": "Trial protocols", "fda_label": "FDA labels", "clinical_note": "Patient notes"}


def load(d: Path):
    q = pd.read_csv(d / "query_metrics.csv")
    i = pd.read_csv(d / "index_metrics.csv")
    for df in (q, i):
        labels = _pipe_labels() if set(df["pipeline"]) - set(PIPE) or "vector_rag" not in set(df["pipeline"]) else PIPE
        df["Pipeline"] = df["pipeline"].map(labels)
        df.drop(df[df["Pipeline"].isna()].index, inplace=True)   # other vector variants: see compare script
        df["Doc type"] = df["doc_type"].map(DT)
        if "error" in df:
            df.drop(df[df["error"].notna()].index, inplace=True)
    q["uncached_input"] = q["input_tokens"] - q.get("cached_tokens", 0).fillna(0)
    return q, i


def figures(q: pd.DataFrame, i: pd.DataFrame) -> dict:
    F = {}
    pipes = ["Vector RAG", "PageIndex"]
    dts = [d for d in DT.values() if d in set(q["Doc type"])]
    for scope in ("document", "corpus"):
        s = q[q.scope == scope]
        if s.empty:
            continue
        # accuracy by question type
        g = s.groupby(["q_type", "Pipeline"]).judge_score.mean().unstack()
        F[f"acc_qtype_{scope}"] = [{"type": "bar", "name": p, "x": list(g.index), "y": list((g[p] * 100).round(1)),
                                    "marker": {"color": COL[p]}, "text": list((g[p] * 100).round(0).astype(int).astype(str) + "%"),
                                    "textposition": "outside"} for p in pipes if p in g]
        # latency box by doc type
        F[f"lat_{scope}"] = [{"type": "box", "name": p, "x": list(s[s.Pipeline == p]["Doc type"]),
                              "y": list(s[s.Pipeline == p].latency_s.round(2)), "marker": {"color": COL[p]},
                              "boxpoints": False} for p in pipes]
        # cost per query and per correct answer
        c = s.groupby(["Doc type", "Pipeline"]).agg(cost=("cost_usd", "mean"), acc=("judge_score", "mean")).reset_index()
        c["cpc"] = c.cost / c.acc.clip(lower=0.01)
        F[f"cost_{scope}"] = [{"type": "bar", "name": p, "x": list(c[c.Pipeline == p]["Doc type"]),
                               "y": list((c[c.Pipeline == p].cost * 1000).round(2)), "marker": {"color": COL[p]}}
                              for p in pipes]
        F[f"cpc_{scope}"] = [{"type": "bar", "name": p, "x": list(c[c.Pipeline == p]["Doc type"]),
                              "y": list((c[c.Pipeline == p].cpc * 1000).round(2)), "marker": {"color": COL[p]}}
                             for p in pipes]
        # token composition
        t = s.groupby(["Pipeline", "Doc type"])[["uncached_input", "cached_tokens", "output_tokens"]].mean().reset_index()
        t["label"] = t.Pipeline + " · " + t["Doc type"]
        t = t.sort_values(["Doc type", "Pipeline"])
        F[f"tok_{scope}"] = [{"type": "bar", "name": n, "x": list(t.label), "y": list(t[k].round(0)),
                              "marker": {"color": col}}
                             for k, n, col in (("uncached_input", "Input (uncached)", "#5b6b85"),
                                               ("cached_tokens", "Input (cached)", "#a9b6cc"),
                                               ("output_tokens", "Output + reasoning", "#d8a23a"))]
        # accuracy vs cost bubbles
        F[f"tradeoff_{scope}"] = [{"type": "scatter", "mode": "markers+text", "name": p,
                                   "x": list((c[c.Pipeline == p].cost * 1000).round(2)),
                                   "y": list((c[c.Pipeline == p].acc * 100).round(1)),
                                   "text": list(c[c.Pipeline == p]["Doc type"]), "textposition": "top center",
                                   "marker": {"size": 18, "color": COL[p]}} for p in pipes]
    # scope penalty (document -> corpus)
    sp = q.groupby(["Pipeline", "scope"]).agg(cost=("cost_usd", "mean"), lat=("latency_s", "median"),
                                              tok=("input_tokens", "mean")).reset_index()
    F["scope_tok"] = [{"type": "bar", "name": sc.title() + " scope", "x": list(sp[sp.scope == sc].Pipeline),
                       "y": list(sp[sp.scope == sc].tok.round(0)),
                       "marker": {"color": "#8796b0" if sc == "document" else "#3d4a63"}} for sc in ("document", "corpus")]
    # page hit / traceability
    ph = q.groupby(["scope", "Pipeline"]).page_hit.mean().unstack()
    F["pagehit"] = [{"type": "bar", "name": p, "x": [x.title() + " scope" for x in ph.index],
                     "y": list((ph[p] * 100).round(1)), "marker": {"color": COL[p]}} for p in pipes if p in ph]
    # indexing
    F["idx_cost"] = [{"type": "scatter", "mode": "markers", "name": p, "x": list(i[i.Pipeline == p].pages),
                      "y": list(i[i.Pipeline == p].cost_usd.round(6)), "marker": {"color": COL[p], "size": 6, "opacity": .6}}
                     for p in pipes]
    F["idx_time"] = [{"type": "scatter", "mode": "markers", "name": p, "x": list(i[i.Pipeline == p].pages),
                      "y": list(i[i.Pipeline == p].wall_s.round(2)), "marker": {"color": COL[p], "size": 6, "opacity": .6}}
                     for p in pipes]
    it = i.groupby(["Doc type", "Pipeline"]).agg(cost=("cost_usd", "sum"), hrs=("wall_s", lambda x: x.sum() / 3600)).reset_index()
    F["idx_total"] = [{"type": "bar", "name": p, "x": list(it[it.Pipeline == p]["Doc type"]),
                       "y": list(it[it.Pipeline == p].cost.round(2)), "marker": {"color": COL[p]}} for p in pipes]
    # monthly projection at volumes (document scope mix)
    vols = [100, 1000, 5000, 10000, 50000]
    d = q[q.scope == "document"].groupby("Pipeline").cost_usd.mean()
    F["projection"] = [{"type": "scatter", "mode": "lines+markers", "name": p, "x": vols,
                        "y": [round(v * 30 * d[p], 2) for v in vols], "line": {"color": COL[p], "width": 3}}
                       for p in pipes if p in d]
    return F


def kpis(q, i):
    out = {}
    for scope in ("document", "corpus"):
        s = q[q.scope == scope]
        out[scope] = {p: {"acc": round(g.judge_score.mean() * 100, 1), "p50": round(g.latency_s.median(), 2),
                          "p95": round(g.latency_s.quantile(.95), 2), "cost": round(g.cost_usd.mean() * 1000, 2),
                          "tok": int(g.input_tokens.mean() + g.output_tokens.mean()),
                          "turns": round(g.turns.mean(), 1), "hit": round(g.page_hit.mean() * 100, 1)}
                      for p, g in s.groupby("Pipeline")}
    out["index"] = {p: {"cost": round(g.cost_usd.sum(), 2), "hours": round(g.wall_s.sum() / 3600, 2),
                        "docs": int(len(g)), "pages": int(g.pages.sum())} for p, g in i.groupby("Pipeline")}
    return out


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>__TITLE__</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#18202e;--mute:#5d6778;--line:#e3e6ec;--v:#2f6fdb;--p:#e0752d}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:28px 32px 8px}h1{margin:0;font-size:24px}h2{font-size:17px;margin:28px 0 4px}p.s{color:var(--mute);margin:2px 0 0}
.banner{margin:14px 32px 0;padding:10px 14px;border-radius:8px;background:#fff4e5;border:1px solid #f3c98b;color:#7a4a00}
.wrap{padding:0 32px 40px}.tabs{display:flex;gap:6px;margin:18px 0 0}.tabs button{border:1px solid var(--line);background:#fff;padding:7px 14px;border-radius:999px;cursor:pointer;font:inherit}
.tabs button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:14px}
.k{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.k b{display:block;font-size:12px;color:var(--mute);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.k .row{display:flex;justify-content:space-between;margin-top:6px;font-variant-numeric:tabular-nums}.v{color:var(--v);font-weight:700}.p{color:var(--p);font-weight:700}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}.c{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
.c h3{margin:4px 8px 0;font-size:14px}.c p{margin:0 8px;color:var(--mute);font-size:12.5px}.plot{height:330px}
@media (max-width:520px){.grid{grid-template-columns:1fr}header,.wrap{padding-left:14px;padding-right:14px}.banner{margin:12px 14px 0}}
</style></head><body>
<header><h1>PageIndex vs Vector RAG — clinical corpus benchmark</h1>
<p class="s">__SUB__</p></header>__BANNER__
<div class="wrap">
<div class="tabs"><button class="on" data-s="document">Document scope</button><button data-s="corpus">Corpus scope</button></div>
<p class="s" id="scopenote"></p>
<div class="kpis" id="kpis"></div>
<h2>Answer quality</h2><div class="grid">
<div class="c"><h3>Accuracy by question type</h3><p>LLM-judge score (1 / 0.5 / 0), blind to pipeline</p><div class="plot" id="acc"></div></div>
<div class="c"><h3>Accuracy vs cost per query</h3><p>Up and to the left is better</p><div class="plot" id="trade"></div></div>
<div class="c"><h3>Traceability: gold page retrieved or cited</h3><p>Did the answer come from the right page?</p><div class="plot" id="hit"></div></div>
</div>
<h2>Latency, tokens and cost per query</h2><div class="grid">
<div class="c"><h3>End-to-end latency (s)</h3><p>Box = p25–p75, whiskers to p5/p95</p><div class="plot" id="lat"></div></div>
<div class="c"><h3>Tokens per query</h3><p>PageIndex re-sends tree + pages each agent turn (partly cached)</p><div class="plot" id="tok"></div></div>
<div class="c"><h3>Cost per query (USD × 1,000)</h3><p>Same answering model on both sides</p><div class="plot" id="cost"></div></div>
<div class="c"><h3>Cost per correct answer (USD × 1,000)</h3><p>Cost ÷ accuracy</p><div class="plot" id="cpc"></div></div>
<div class="c"><h3>Scope penalty: input tokens per query</h3><p>Corpus scope = system must also find the document</p><div class="plot" id="scope"></div></div>
<div class="c"><h3>Monthly answering cost at volume</h3><p>Queries per day × 30, document-scope mix</p><div class="plot" id="proj"></div></div>
</div>
<h2>Indexing (one-off)</h2><div class="grid">
<div class="c"><h3>Indexing cost per document vs pages</h3><p>Log–log</p><div class="plot" id="icost"></div></div>
<div class="c"><h3>Indexing time per document vs pages (s)</h3><p>Log–log; PageIndex has a fixed per-document overhead</p><div class="plot" id="itime"></div></div>
<div class="c"><h3>Total indexing cost by corpus (USD)</h3><p id="itotal"></p><div class="plot" id="iall"></div></div>
</div></div>
<script>
const F=__FIGS__, K=__KPIS__;
const L=(o)=>Object.assign({margin:{l:52,r:12,t:10,b:58},barmode:'group',legend:{orientation:'h',y:-0.22},
 paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{size:12}},o||{});
const cfg={displayModeBar:false,responsive:true};
function kp(scope){const k=K[scope],v=k['Vector RAG'],p=k['PageIndex'];if(!v||!p)return;
 const rows=[['Accuracy','acc','%'],['Median latency','p50',' s'],['p95 latency','p95',' s'],['Cost / query','cost',' m$'],['Tokens / query','tok',''],['LLM calls / query','turns',''],['Gold page hit','hit','%']];
 document.getElementById('kpis').innerHTML=rows.map(([t,f,u])=>`<div class="k"><b>${t}</b><div class="row"><span>Vector</span><span class="v">${v[f].toLocaleString()}${u}</span></div><div class="row"><span>PageIndex</span><span class="p">${p[f].toLocaleString()}${u}</span></div></div>`).join('');
 document.getElementById('scopenote').textContent=scope==='document'?'Each question is routed to its document (doc_id / metadata filter). Typical for "ask this protocol" features.':'No document given: the system must find the right document among all '+K.index['PageIndex'].docs.toLocaleString()+' first. Typical for "search the library" features.';}
function draw(scope){kp(scope);
 Plotly.react('acc',F['acc_qtype_'+scope],L({yaxis:{title:'% correct',range:[0,105]}}),cfg);
 Plotly.react('trade',F['tradeoff_'+scope],L({xaxis:{title:'cost per query (USD × 1,000)'},yaxis:{title:'% correct'}}),cfg);
 Plotly.react('lat',F['lat_'+scope],L({boxmode:'group',yaxis:{title:'seconds'}}),cfg);
 Plotly.react('tok',F['tok_'+scope],L({barmode:'stack',yaxis:{title:'tokens'},xaxis:{tickangle:-20}}),cfg);
 Plotly.react('cost',F['cost_'+scope],L({yaxis:{title:'USD × 1,000'}}),cfg);
 Plotly.react('cpc',F['cpc_'+scope],L({yaxis:{title:'USD × 1,000'}}),cfg);}
Plotly.newPlot('hit',F.pagehit,L({yaxis:{title:'%',range:[0,105]}}),cfg);
Plotly.newPlot('scope',F.scope_tok,L({yaxis:{title:'input tokens'}}),cfg);
Plotly.newPlot('proj',F.projection,L({xaxis:{title:'queries / day',type:'log'},yaxis:{title:'USD / month'}}),cfg);
Plotly.newPlot('icost',F.idx_cost,L({xaxis:{type:'log',title:'pages'},yaxis:{type:'log',title:'USD'}}),cfg);
Plotly.newPlot('itime',F.idx_time,L({xaxis:{type:'log',title:'pages'},yaxis:{type:'log',title:'seconds'}}),cfg);
Plotly.newPlot('iall',F.idx_total,L({yaxis:{title:'USD'}}),cfg);
const I=K.index;document.getElementById('itotal').textContent=`${I['PageIndex'].docs.toLocaleString()} docs / ${I['PageIndex'].pages.toLocaleString()} pages — PageIndex $${I['PageIndex'].cost} in ${I['PageIndex'].hours} h sequential; Vector $${I['Vector RAG'].cost} in ${I['Vector RAG'].hours} h`;
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('on'));b.classList.add('on');draw(b.dataset.s)});
draw('document');
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results"))
    ap.add_argument("--out")
    a = ap.parse_args()
    d = Path(a.dir)
    q, i = load(d)
    modeled = "run_type" in q and (q.run_type == "modeled").any()
    out = Path(a.out) if a.out else d / "report.html"
    sub = (f"{q.q_id.nunique()} questions × {q.scope.nunique()} scopes · {i.doc_id.nunique():,} documents · "
           f"{int(i[i.pipeline == 'pageindex'].pages.sum()):,} pages")
    banner = ('<div class="banner"><b>Modeled preview — not measured.</b> Numbers come from bench/simulate.py '
              'assumptions. Run the benchmark to replace them with measured results.</div>') if modeled else ""
    html = (HTML.replace("__TITLE__", "PageIndex vs Vector RAG").replace("__SUB__", sub).replace("__BANNER__", banner)
            .replace("__FIGS__", json.dumps(figures(q, i))).replace("__KPIS__", json.dumps(kpis(q, i))))
    out.write_text(html)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
