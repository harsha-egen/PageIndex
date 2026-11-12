# PageIndex vs Vector RAG — clinical benchmark

A reproducible, apples-to-apples benchmark of **PageIndex** (vectorless, reasoning-based RAG)
against a **traditional vector RAG** pipeline on a large public clinical corpus. It captures
latency, tokens, cost, accuracy and traceability for both, then renders an interactive dashboard.

## What is compared

| | Vector RAG (baseline) | PageIndex |
|---|---|---|
| Index | page text → 512-token chunks (64 overlap) → `text-embedding-3-small` | per-document tree index (`gpt-5.6-luna`) |
| Retrieve | cosine top-8 (exact, in-memory: best case for vectors) | LLM agent searches the tree (multi-turn) |
| Answer | **same chat model** (`gpt-5.6-terra`, medium reasoning) | **same chat model** |
| Citations | pages of retrieved chunks | `<cite page=…>` from PageIndex |

Both are measured by the **same instrument**: `bench/meter.py` intercepts every OpenAI HTTP call
(the OpenAI SDK and LiteLLM — which PageIndex uses — both go through httpx), so tokens, cached
tokens, reasoning tokens, call count, latency and cost are counted identically on both sides.

## Corpus (scale `large`, configurable)

| Source | Docs | Typical length | Why it is in the test |
|---|---|---|---|
| ClinicalTrials.gov protocols | 40 | 40–300 pages | deep numbered structure, cross-section questions |
| openFDA prescribing information | 150 | 15–60 pages | fixed PLR structure, "(see 5.3)" cross-references |
| Synthetic patient notes | 1,200 | 1–2 pages | short, flat documents; many near-identical |

All data is public or fully synthetic. No PHI is used. Set `data.synthea_dir` to use real Synthea
FHIR output instead of the built-in note generator.

## Questions (~580 at `large`)

`lookup`, `multi_hop` (two sections), `section_ref` (clinician phrasing), `cross_ref` (label
cross-references), `patient` (exact fact in a note). LLM-written questions must quote evidence that
is **verified verbatim on the gold page**, or they are discarded. Each question runs in two scopes:

* **document scope** — the system is told which document (typical "ask this protocol" UX)
* **corpus scope** — the system must also find the document among all 1,390 (library search)

**Before the final run, have a clinician review ~30 questions** in `results/questions.jsonl`.

## Metrics captured

Per query: end-to-end latency (p50/p95), retrieval vs generation time, LLM calls (agent turns),
input / cached / output / reasoning tokens, embedding tokens, cost (USD), LLM-judge score
(blind to pipeline, judge cost not charged to either side), gold-page hit and recall, wrong-document
rate. Per document indexed: wall time, cost, tokens, LLM calls, storage bytes, tree nodes / chunks.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...            # optional: OPENFDA_API_KEY for higher openFDA limits

python tests/test_offline.py            # offline sanity checks (no key needed)
python -m tests.test_langfuse           # Langfuse integration with a stand-in SDK
python -m bench.build_corpus            # ~1,390 docs, ~12-13k pages (scale in config.yaml)
python -m bench.qa_gen                  # questions + gold pages
python -m bench.run_benchmark all       # index -> query (both scopes) -> judge. Resumable.
python -m bench.report                  # -> results/report.html
```

Or `./run.sh` for all of it. Try `--scale small` / `--limit 20` first to check keys and costs.

**Budget (modeled, `large`, list prices as of Nov 2026):** indexing ≈ $15 (PageIndex) + $0.20
(embeddings); queries ≈ $25 in document scope and ≈ $105 in corpus scope (PageIndex dominates the
latter); judge ≈ $15 with `gpt-5.6-sol`. Total ≈ $160. Wall time ≈ 4 h for PageIndex indexing
(sequential) + ≈ 6 h for all queries. Use `--scopes document` to halve query time and cost.

**Prices:** every model price is in `config.yaml`. OpenAI changed GPT-5.6 prices several times in
2026 (Sol has a promo rate until 2026-11-21), so check the pricing page before your run.

## Langfuse tracing and dashboard

Put keys in a local `.env` (copy `.env.example`; `.env` is git-ignored). With keys present,
every run sends one trace per query and per indexed document, one generation per OpenAI
call (including PageIndex's internal agent turns) with exact token buckets and USD cost,
and scores `accuracy`, `accuracy/<q_type>`, `page_hit`, `latency_s`, `llm_calls`.
Without keys, tracing is off and results still go to `results/`.

```bash
python -m bench.langfuse_replay --dir results/modeled --limit 200   # smoke test, ~1.4k units
python -m bench.langfuse_replay --dir results/modeled               # full modeled preview
```

Dashboard widget spec: `docs/LANGFUSE_DASHBOARD.md`.

## Side-by-side chat demo

Ask a question in the browser and watch both pipelines answer it at the same time, with
every OpenAI call (model, tokens, cached tokens, latency, cost, and the tools PageIndex's
agent called) appearing live as it happens.

```bash
python -m bench.chat_app --fake                         # try the UI with simulated answers ($0)

python -m bench.build_corpus --scale small --only labels  # 25 FDA labels (free, public API)
python -m bench.run_benchmark index --limit 5             # index 5 docs in both pipelines
python -m bench.chat_app                                  # http://127.0.0.1:8765
```

* Pick one document (document scope) or "Whole library" (corpus scope).
* Same answering model, prompt and reasoning effort as the benchmark; same meter.
* With Langfuse keys in `.env`, each answer is a trace in environment `demo`, linked from the UI.
* Documents indexed here are reused by the full benchmark run (indexing is resumable).
* The two pipelines run in parallel here, so latencies are a little noisier than in the
  benchmark (which runs one at a time). Use the benchmark for numbers you report.

## Rehearse without running anything

```bash
python -m bench.simulate                # MODELED numbers -> results/modeled/
python -m bench.report --dir results/modeled --out results/report_modeled.html
```

The modeled run uses explicit assumptions (token mechanics of each pipeline, PageIndex's published
indexing figures, and **assumed** accuracy rates) saved to `results/modeled/assumptions.json`. The
report shows a "Modeled preview — not measured" banner. Never present modeled accuracy as a finding.

## Fairness notes and known limits

* Same answering model, reasoning effort and instructions on both sides.
* The vector baseline uses exact search in memory: real vector DBs add latency, and a reranker or
  hybrid BM25 would raise its accuracy. Add them as extra pipelines if your team uses them.
* PageIndex local mode handles text-based PDFs only. Scanned protocols need PageIndex Cloud (OCR);
  the corpus builder drops scanned PDFs so both pipelines see the same text.
* Corpus scope for PageIndex uses its built-in local library search. PageIndex Cloud's File System
  (folders + file-level tree) is designed for large libraries and may behave differently; benchmark
  it separately with `pageindex.mode: cloud`.
* Keep `benchmark.concurrency: 1` for clean latency. Latency depends on OpenAI load; run at a
  quiet time and repeat a subset to see variance.
* Streaming calls carry no usage block, so both pipelines run non-streaming here.

## Layout

```
config.yaml                 models, prices, corpus scale, chunking, scopes
bench/meter.py              HTTP-level usage meter (tokens, cost, latency) + Langfuse generations
bench/tracing.py            Langfuse traces, tags, scores
bench/langfuse_replay.py    push a results folder (e.g. the modeled preview) into Langfuse
docs/LANGFUSE_DASHBOARD.md  widget-by-widget dashboard spec
bench/data/                 ClinicalTrials.gov, openFDA, synthetic notes, PDF renderer
bench/pipelines/            vector_rag.py, pageindex_rag.py
bench/qa_gen.py             verified question set with gold pages
bench/evaluate.py           LLM judge + page-hit metrics
bench/run_benchmark.py      index / query / judge (resumable)
bench/report.py             self-contained HTML dashboard
bench/simulate.py           modeled preview with explicit assumptions
```
