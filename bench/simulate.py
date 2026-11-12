"""MODELED preview results — for rehearsing the demo before the measured run.

    python -m bench.simulate     -> results/modeled/{index,query}_metrics.csv

These numbers are NOT measurements. They are produced by the token/latency model below,
built from each pipeline's mechanics:
  * vector RAG = 1 embedding call + 1 LLM call with top_k chunks
  * PageIndex  = an agent loop: each turn re-sends the system prompt, the tree, and the pages
                 read so far (prefix-cached), plus document discovery when no doc_id is given
and from PageIndex's own published indexing figures (~$0.0011/page with gpt-5.6-luna,
~13 s for 9 pages to ~4.5 min for 1,098 pages). Accuracy rates are ASSUMPTIONS chosen to
reflect the published pattern (structure-heavy questions favour tree search) — the measured
run is what decides them for your data. Every assumption is a named constant here and is
written to results/modeled/assumptions.json so the dashboard can show it.
"""
from __future__ import annotations

import json
import math
import random

import numpy as np
import pandas as pd

from bench.common import ROOT, load_config

A = {  # ------------------------------------------------------------ assumptions
    "tokens_per_page": 650,
    "llm_ttft_s": 0.55, "llm_tok_per_s": 110,            # gpt-5.6-terra class, medium effort
    "reasoning_tokens_per_call": (120, 420),
    "answer_tokens": (90, 260),
    "vector_system_tokens": 300, "vector_question_tokens": 40,
    "vector_embed_call_s": (0.12, 0.30), "vector_search_s_per_10k_chunks": 0.003,
    "pi_system_tokens": 1200, "pi_tree_tokens_per_page": 12, "pi_pages_read_per_turn": 2,
    "pi_tool_call_tokens": 80, "pi_cache_hit_share": 0.55,
    "pi_doc_listing_tokens_per_doc": 40, "pi_discovery_turns": (1, 2),
    "pi_index_usd_per_page": 0.0011, "pi_index_fixed_s": 8.0, "pi_index_s_per_page": 0.24,
    "vec_parse_s_per_page": 0.015, "vec_embed_s_per_batch": 0.35, "embedding_dim": 1536,
    # P(judge score = 1) by (doc_type, q_type) -> (vector, pageindex), document scope
    "p_correct": {
        "trial_protocol/lookup": (0.80, 0.88), "trial_protocol/multi_hop": (0.52, 0.81),
        "trial_protocol/section_ref": (0.60, 0.90),
        "fda_label/lookup": (0.86, 0.88), "fda_label/cross_ref": (0.58, 0.84),
        "clinical_note/patient": (0.95, 0.96),
    },
    # multiplier on P(correct) when the system must also find the document (corpus scope)
    "corpus_scope_factor": {"trial_protocol": (0.85, 0.93), "fda_label": (0.88, 0.95),
                            "clinical_note": (0.72, 0.80)},
}


def _turns_pi(pages: int, rng) -> int:
    return max(2, round(1.6 + math.log2(max(pages, 1) / 6) * 0.8 + rng.uniform(-0.6, 0.9)))


def _llm_time(out_tokens: int, rng) -> float:
    return A["llm_ttft_s"] * rng.uniform(0.8, 1.4) + out_tokens / A["llm_tok_per_s"]


def build_corpus(cfg, rng):
    sp = cfg["scale_params"]
    docs = []
    for i in range(sp["trials"]):
        docs.append({"doc_id": f"NCT{rng.randint(10**7, 10**8 - 1):08d}", "doc_type": "trial_protocol",
                     "pages": int(np.clip(rng.lognormvariate(math.log(130), 0.45), 42, 340))})
    for i in range(sp["labels"]):
        docs.append({"doc_id": f"LBL-{rng.randrange(16**8):08x}", "doc_type": "fda_label",
                     "pages": int(np.clip(rng.lognormvariate(math.log(30), 0.4), 12, 80))})
    for i in range(sp["notes"]):
        docs.append({"doc_id": f"NOTE-{100000 + i * 7}", "doc_type": "clinical_note",
                     "pages": rng.choice([1, 1, 1, 2])})
    return docs


def sim_index(docs, cfg, rng):
    pr, rows = cfg["pricing"], []
    emb_price = pr[cfg["models"]["embedding"]]["input"]
    idx_p = pr[cfg["models"]["pageindex_index"]]
    for d in docs:
        p = d["pages"]
        density = rng.uniform(0.7, 1.3)
        # vector
        toks = int(p * A["tokens_per_page"] * density * 1.14)          # 1.14 = overlap
        chunks = max(1, toks // 512)
        batches = math.ceil(chunks / cfg["vector_rag"]["embed_batch"])
        t = p * A["vec_parse_s_per_page"] + batches * A["vec_embed_s_per_batch"] * rng.uniform(0.8, 1.5)
        rows.append({"pipeline": "vector_rag", "doc_id": d["doc_id"], "doc_type": d["doc_type"], "pages": p,
                     "wall_s": round(t, 3), "cost_usd": toks * emb_price / 1e6, "embed_tokens": toks,
                     "input_tokens": 0, "output_tokens": 0, "llm_calls": 0, "embed_calls": batches,
                     "chunks": chunks, "storage_bytes": chunks * (A["embedding_dim"] * 4 + 2100)})
        # pageindex (published: ~$0.0011/page, cost spread = text density)
        cost = p * A["pi_index_usd_per_page"] * density + 0.0004
        out_share = 0.18
        in_tok = int(cost * (1 - out_share) / idx_p["input"] * 1e6)
        out_tok = int(cost * out_share / idx_p["output"] * 1e6)
        t = A["pi_index_fixed_s"] * rng.uniform(0.7, 1.3) + p * A["pi_index_s_per_page"] * density
        rows.append({"pipeline": "pageindex", "doc_id": d["doc_id"], "doc_type": d["doc_type"], "pages": p,
                     "wall_s": round(t, 3), "cost_usd": cost, "embed_tokens": 0, "input_tokens": in_tok,
                     "output_tokens": out_tok, "llm_calls": max(2, p // 6), "embed_calls": 0,
                     "tree_nodes": max(3, int(p * rng.uniform(0.35, 0.7))),
                     "storage_bytes": int(p * (A["tokens_per_page"] * 4.2 + 900))})
    return pd.DataFrame(rows)


def build_questions(docs, cfg, rng):
    sp, qs = cfg["scale_params"], []
    plans = {"trial_protocol": ["lookup", "multi_hop", "section_ref", "multi_hop"][: sp["q_per_trial"]],
             "fda_label": ["lookup", "cross_ref"][: sp["q_per_label"]]}
    for d in docs:
        for j, qt in enumerate(plans.get(d["doc_type"], [])):
            qs.append({"q_id": f"{d['doc_id']}-q{j + 1}", "doc_id": d["doc_id"], "doc_type": d["doc_type"],
                       "q_type": qt, "doc_pages": d["pages"]})
    notes = [d for d in docs if d["doc_type"] == "clinical_note"]
    for d in rng.sample(notes, min(sp["q_notes"], len(notes))):
        qs.append({"q_id": f"{d['doc_id']}-q1", "doc_id": d["doc_id"], "doc_type": "clinical_note",
                   "q_type": "patient", "doc_pages": d["pages"]})
    return qs


def _score(p, rng):
    u = rng.random()
    return 1.0 if u < p else (0.5 if u < p + (1 - p) * 0.3 else 0.0)


def sim_queries(qs, docs, cfg, rng):
    chat = cfg["pricing"][cfg["models"]["chat"]]
    emb_price = cfg["pricing"][cfg["models"]["embedding"]]["input"]
    k, n_docs = cfg["vector_rag"]["top_k"], len(docs)
    total_chunks = sum(max(1, int(d["pages"] * A["tokens_per_page"] * 1.14) // 512) for d in docs)
    rows = []
    for scope in ("document", "corpus"):
        for q in qs:
            key = f"{q['doc_type']}/{q['q_type']}"
            pv, pp = A["p_correct"][key]
            if scope == "corpus":
                fv, fp = A["corpus_scope_factor"][q["doc_type"]]
                pv, pp = pv * fv, pp * fp
            # ---------------- vector RAG
            reason = rng.randint(*A["reasoning_tokens_per_call"])
            ans = rng.randint(*A["answer_tokens"])
            inp = A["vector_system_tokens"] + A["vector_question_tokens"] + k * (512 + 15)
            if q["doc_type"] == "clinical_note" and scope == "document":
                inp = A["vector_system_tokens"] + A["vector_question_tokens"] + q["doc_pages"] * 560
            t_emb = rng.uniform(*A["vector_embed_call_s"])
            t_search = (A["vector_search_s_per_10k_chunks"] * total_chunks / 1e4) if scope == "corpus" else 0.0008
            t_gen = _llm_time(reason + ans, rng)
            cost = (inp * chat["input"] + (reason + ans) * chat["output"] + 40 * emb_price) / 1e6
            s = _score(pv, rng)
            hit = int(rng.random() < (0.93 if s == 1 else 0.55 if s == 0.5 else 0.25))
            rows.append({"pipeline": "vector_rag", "scope": scope, **q, "latency_s": t_emb + t_search + t_gen,
                         "t_retrieval": t_emb + t_search, "t_generate": t_gen, "turns": 1, "llm_calls": 1,
                         "input_tokens": inp, "cached_tokens": 0, "output_tokens": reason + ans,
                         "reasoning_tokens": reason, "embed_tokens": 40, "cost_usd": cost,
                         "judge_score": s, "page_hit": hit, "cited": 1})
            # ---------------- PageIndex
            turns = _turns_pi(q["doc_pages"], rng)
            tree = A["pi_tree_tokens_per_page"] * q["doc_pages"]
            base = A["pi_system_tokens"] + tree
            disc = 0
            if scope == "corpus":
                disc = rng.randint(*A["pi_discovery_turns"])
            inp = cached = out = reason_t = 0
            t_total = 0.0
            for t in range(disc):
                listing = A["pi_doc_listing_tokens_per_doc"] * n_docs
                this_in = A["pi_system_tokens"] + listing
                r_t = rng.randint(80, 260)
                inp += this_in; cached += int(this_in * (0.0 if t == 0 else 0.9))
                out += r_t + A["pi_tool_call_tokens"]; reason_t += r_t
                t_total += _llm_time(r_t + A["pi_tool_call_tokens"], rng) + this_in / 60000
            for t in range(turns):
                read = t * A["pi_pages_read_per_turn"] * min(A["tokens_per_page"], 560 * q["doc_pages"])
                this_in = base + read + t * A["pi_tool_call_tokens"]
                r_t = rng.randint(*A["reasoning_tokens_per_call"]) // 2
                o_t = (ans if t == turns - 1 else A["pi_tool_call_tokens"])
                inp += this_in
                cached += int(this_in * (A["pi_cache_hit_share"] if t > 0 else 0))
                out += r_t + o_t; reason_t += r_t
                t_total += _llm_time(r_t + o_t, rng) + this_in / 80000 + 0.04  # + tool execution
            cost = ((inp - cached) * chat["input"] + cached * chat["cached"] + out * chat["output"]) / 1e6
            s = _score(pp, rng)
            hit = int(rng.random() < (0.96 if s == 1 else 0.6 if s == 0.5 else 0.3))
            rows.append({"pipeline": "pageindex", "scope": scope, **q, "latency_s": t_total,
                         "t_retrieval": t_total * 0.8, "t_generate": t_total * 0.2, "turns": turns + disc,
                         "llm_calls": turns + disc, "input_tokens": inp, "cached_tokens": cached,
                         "output_tokens": out, "reasoning_tokens": reason_t, "embed_tokens": 0,
                         "cost_usd": cost, "judge_score": s, "page_hit": hit, "cited": 1})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    rng = random.Random(cfg["seed"])
    docs = build_corpus(cfg, rng)
    idx = sim_index(docs, cfg, rng)
    qs = build_questions(docs, cfg, rng)
    qry = sim_queries(qs, docs, cfg, rng)
    out = ROOT / "results" / "modeled"
    out.mkdir(parents=True, exist_ok=True)
    idx["run_type"] = qry["run_type"] = "modeled"
    for df, name in ((idx, "index_metrics"), (qry, "query_metrics")):
        num = df.select_dtypes("number").columns
        df[num] = df[num].round(6)
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "assumptions.json").write_text(json.dumps(A, indent=1))
    print(f"modeled: {len(docs)} docs / {idx[idx.pipeline == 'pageindex'].pages.sum()} pages, "
          f"{len(qs)} questions x 2 pipelines x 2 scopes = {len(qry)} query rows -> {out}")


if __name__ == "__main__":
    main()
