# Langfuse dashboard: "PageIndex vs Vector RAG"

Build once (about 10 minutes) in Langfuse → **Dashboards → Widgets → New widget**, then
**Dashboards → New dashboard** and add the widgets. Afterwards use the widget ⋯ menu
**Download as JSON** to version the dashboard in this repo and import it into other projects.

## How the data is shaped (what to break down by)

| Langfuse field | Value written by the harness | Use it for |
| --- | --- | --- |
| Trace name | `vector_rag/document`, `pageindex/corpus`, `pageindex/index`, … | filtering one pipeline + scope |
| User | same as trace name (`<pipeline>/<scope>`) | **the comparison breakdown in every widget** |
| Session | run name (`clinical-bench`, `clinical-bench-modeled`) | one session per run |
| Tags | pipeline, scope, doc_type, q_type, `query` or `index`, `measured`/`modeled` | filters |
| Environment | `benchmark` (live runs) or `modeled` (replay) | dashboard environment selector |
| Generation / embedding | one per OpenAI call, model + usage buckets + USD cost | cost, tokens, model mix |
| Scores (numeric) | `accuracy`, `accuracy/<q_type>`, `page_hit`, `wrong_doc`, `latency_s`, `llm_calls` | quality and per-query stats |

## Widgets

| # | Widget | View | Metric (aggregation) | Filters | Breakdown | Chart |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Accuracy by pipeline and scope | Scores (numeric) | Value (avg) | Score name = `accuracy` | User | Bar |
| 2 | Gold page hit rate | Scores (numeric) | Value (avg) | Score name = `page_hit` | User | Bar |
| 3 | Accuracy by question type | Scores (numeric) | Value (avg) | Score name starts with `accuracy/`; User = one pipeline | Score name | Bar (duplicate per pipeline) |
| 4 | Median latency (live runs) | Traces | Latency (p50) | Tags contain `query` | User | Bar |
| 5 | p95 latency (live runs) | Traces | Latency (p95) | Tags contain `query` | User | Bar |
| 6 | Latency incl. replay | Scores (numeric) | Value (p50 and p95) | Score name = `latency_s` | User | Bar |
| 7 | Cost per query | Traces | Total cost (avg) | Tags contain `query` | User | Bar |
| 8 | Tokens per query | Traces | Total tokens (avg) | Tags contain `query` | User | Bar |
| 9 | LLM calls per query | Scores (numeric) | Value (avg) | Score name = `llm_calls` | User | Bar |
| 10 | Total spend by pipeline | Traces | Total cost (sum) | none | User | Bar |
| 11 | Indexing cost by pipeline | Traces | Total cost (sum) | Tags contain `index` | User | Bar |
| 12 | Spend by model | Observations | Total cost (sum) | Type = generation or embedding | Model | Pie |
| 13 | Cost over time | Traces | Total cost (sum) | none | User | Line (time series) |

Layout suggestion: row 1 = widgets 1, 2, 7, 4 (the four headline numbers); row 2 = 3, 5, 8, 9;
row 3 = 10, 11, 12, 13. Set the dashboard Environment selector to `benchmark` for real
results and `modeled` to rehearse.

Notes
* Replayed (modeled) traces have near-zero duration: use widget 6 for their latency.
* Langfuse also ships curated **Cost** and **Latency** dashboards that work on this data as is.
* Every trace is clickable: open a `pageindex/*` trace to show the agent's turns (each
  generation) and its token growth; open a `vector_rag/*` trace to show the single call.
