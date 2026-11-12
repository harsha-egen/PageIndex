# Demo script: when PageIndex helps, and when vector RAG (pgvector) is enough

Run with `python -m bench.chat_app` (vector column: Postgres + pgvector) on 7 indexed documents:

| Doc ID | Document | Pages |
|---|---|---|
| LBL-000155a8 | Naproxen label | 17 |
| LBL-0002ad27 | Mekinist (trametinib) label | 26 |
| LBL-0003458f | Glimepiride label | 11 |
| LBL-0004a0bb | Varenicline label | 19 |
| LBL-00097e78 | Tamsulosin label | 13 |
| NCT01970865 | Lorlatinib (PF-06463922) phase 1/2 protocol, ALK/ROS1 NSCLC. *No PDF outline* | 208 |
| NCT01206062 | SPRINT blood pressure trial protocol | 117 |

**The two pipelines.** Same answering model (`gpt-5.6-terra`, medium reasoning) and same
answer instructions:

- **PageIndex:** an LLM agent reads each document's section tree and opens the pages it chooses.
- **Vector RAG on pgvector:** PostgreSQL 17 + pgvector 0.8. Each document is split into 512-token
  chunks and embedded with `text-embedding-3-small`. An HNSW cosine index returns the top 8
  chunks; document scope is a SQL `WHERE doc_id = …` filter.

**Measured** = the runs on 2026-11-11. Each answer was graded 0 / 0.5 / 1 against the correct
answer by a blind LLM judge (`gpt-5.6-sol`). Costs are at OpenAI list price (see the note
under the takeaway). Acts 1–5 show the first pgvector run; Act 6 shows the end-to-end run
through the demo app. **Across runs** counts every run of that question (PageIndex up to 4 runs,
pgvector up to 3). The question set is `docs/demo_questions.yaml`; raw rows are in
`results/demo_compare.jsonl`.

| Questions | PageIndex | pgvector |
|---|---|---|
| Acts 1–5 (17 questions, typical use) | 16.5 / 17 | 16 / 17 |
| **Act 6 (6 questions, where vector retrieval breaks)** | **6 / 6** ¹ | **1.5 / 6** |
| Mean time per answer (all 23) | 7.7s | 2.8s |
| Cost for all 23 | $1.81 | $0.25 |

¹ The judge gave PageIndex ½ on #20 against my original reference answer, which used naproxen's
acute-pain limit. The label states 1,500 mg/day as the maximum recommended daily dose, which
is what PageIndex answered, so #20 is counted as correct. The reference answer is now fixed.

**Check before a demo:** `python -m bench.demo_compare --app http://127.0.0.1:8765 --judge` sends
all 23 questions through the running app, exactly as the browser does. The last run: 23 of 23
answered, no errors. It costs about $2.

**In the app:** use the **Demo script** dropdown, or paste a question from this page. Either way the
app selects the question's document for you.

**Scope** = what to pick in the document dropdown. "Library" = *Whole library*.

---

## Act 1: simple lookups. pgvector is enough

Both are right. pgvector answers 2–3× faster at a fifth to a third of the cost. PageIndex's agent
spends 3 LLM calls to find what one index lookup finds.

| # | Scope | Question | Correct answer | Measured: PageIndex vs pgvector |
|---|---|---|---|---|
| 1 | Varenicline | What temperature should varenicline tablets be stored at? | 20–25 °C (68–77 °F); excursions 15–30 °C. p.19 | Both ✔. 6.4s / $0.041 vs **3.0s / $0.008** |
| 2 | Tamsulosin | Can tamsulosin be taken together with ketoconazole? | No: 0.4 mg should not be used with strong CYP3A4 inhibitors such as ketoconazole. | Both ✔, both added the PK numbers (Cmax ×2.2, AUC ×2.8). 5.0s / $0.033 vs **1.9s / $0.010** |

**Say:** "For FAQ-style lookups, PageIndex is paying for reasoning it doesn't need."

## Act 2: structure-heavy questions. Same answer, different traceability

Open **Live API calls** on the PageIndex card while it runs: you can watch the agent open
the document, pick sections and read specific pages. Then compare the page chips: PageIndex shows
**Pages cited** (what its answer's [1], [2] point to) and **Pages read** (what it opened). pgvector shows
**Pages retrieved** (all pages in its 8 passages), with the pages its answer cites highlighted.

| # | Scope | Question | Correct answer | Measured: PageIndex vs pgvector |
|---|---|---|---|---|
| 3 | Mekinist | If a patient's LVEF drops 10% or more from baseline and below the lower limit of normal, what should be done, and exactly what dose do they resume at if it recovers? | Withhold up to 4 weeks. If LVEF recovers, resume at the first dose reduction (adult 2 mg → **1.5 mg** daily, Table 3, p.3). If not, discontinue. | Both ✔. PageIndex named the resumed dose (**1.5 mg**). pgvector said "determined by the prior dose and dose-reduction table", then listed the whole table. **PageIndex used 2 pages, pgvector 8.** 8.7s / $0.095 vs 2.6s / $0.010 |
| 4 | Mekinist | List every warning and precaution in section 5 of this label, one line each. | 13 subsections, 5.1 New Primary Malignancies … 5.13 Embryo-Fetal Toxicity (pp. 4–8). | Both ✔, all 13. PageIndex cites pages 5–8 per item. pgvector cites "p.4-5" for items that are on pages 5–7. 7.9s / $0.075 vs 2.8s / $0.011 |
| 5 | Naproxen | Why is naproxen contraindicated around CABG surgery? Follow the cross-reference and explain what the referenced warning section says. | Section 4 → see 5.1: COX-2 NSAID trials in the 10–14 days after CABG showed more MI and stroke. p.4 | Both ✔. PageIndex used 2 pages; pgvector 7. 7.2s / $0.049 vs 2.9s / $0.010 |

**Say:** "When the answer depends on the document's structure (a table referenced from
another section, a whole numbered section), both get there. PageIndex's citations point to
the exact pages it read, which is what an auditor or reviewer wants."

## Act 3: across the whole library. Where they split

| # | Scope | Question | Correct answer | Measured: PageIndex vs pgvector | Across runs |
|---|---|---|---|---|---|
| 6 | Library | Which of these medicines need a lower starting dose in severe renal impairment, and what is that dose? | Varenicline 0.5 mg once daily (CrCl < 30). Glimepiride 1 mg once daily (renal impairment). | **PageIndex ✔ (found both). pgvector ½** (glimepiride only). 15.4s / $0.136 vs 2.8s / $0.010 | **PageIndex ✔ in 4 of 4 runs, pgvector ½ in 3 of 3.** A reliable PageIndex win |
| 7 | Library | Which drug should be taken at least 4 hours before colesevelam? | Glimepiride (sections 2.1, 7.4). | Both ✔. 7.4s / $0.029 vs 2.0s / $0.009 | **pgvector ✔ in 3 of 3. PageIndex ✔ in only 1 of 4** (three times "Not found"). A reliable pgvector win |
| 8 | Library | Which of these drugs carries a boxed warning, and what is it about? | Only naproxen: cardiovascular thrombotic events; GI bleeding, ulceration and perforation. | Both ✔. 8.8s / $0.059 vs **1.9s / $0.009** | Stable |
| 17 | Library | Which study requires stopping strong CYP3A4 inducers at least 12 days before the first dose of study drug? | The lorlatinib study (B7461001), exclusion 17, p.101. | Both ✔. 6.5s / $0.120 vs **1.7s / $0.009** | Stable |

**Say:** "When one answer has to be **collected from several documents** (#6), the agent keeps
looking until it has covered the question. Vector search returns the 8 closest chunks and
stops, and here they all came from one drug. When the question is **one distinctive fact
somewhere in the library** (#7), vector search finds it every time, while the agent first has to
guess which documents to open and sometimes guesses wrong."

## Act 4: honesty checks

| # | Scope | Question | Correct answer | Measured: PageIndex vs pgvector | Across runs |
|---|---|---|---|---|---|
| 9 | Tamsulosin | My patient is having eye surgery next month. Is there anything they should tell the surgeon about this medicine? | Tell the ophthalmologist: intraoperative floppy iris syndrome (IFIS) during cataract or glaucoma surgery (5.5). | Both ✔, though the question never says "cataract" or "IFIS". 6.5s vs 2.4s | Stable |
| 10 | Naproxen | What is the recommended dose of naproxen for migraine? | Not in the label: migraine is not an indication. | Both ✔. pgvector: a clean "Not found in the provided documents". PageIndex: "migraine is not listed as an indication", then offered the general pain dose. **Helpful or overreaching?** Good discussion point | Stable |
| 11 | Naproxen | What is the recommended starting dose, and how is it adjusted for renal impairment? | Indication-specific doses; not recommended if CrCl < 30 mL/min (8.7, p.13). | Both ✔ | **PageIndex varies:** it missed the CrCl < 30 rule in the first live demo (it stopped after pp. 1–6), was fully right in 3 runs, and gave a vaguer starting dose (½) in the app run. pgvector ✔ in 3 of 3. Run it twice live to show that agents vary |

## Act 5: long protocols (117–208 pages)

PageIndex built a 203-section tree for the lorlatinib protocol even though the PDF has no
outline (indexing: 53s, $0.10). pgvector chunked and embedded it in 4s for $0.003.

| # | Scope | Question | Correct answer | Measured: PageIndex vs pgvector | Across runs |
|---|---|---|---|---|---|
| 12 | Lorlatinib | What dose do Phase 2 patients receive, and what are the rules for strong CYP3A inhibitors and strong CYP3A inducers, including any washout before the first dose? | 100 mg once daily, 21-day cycles. Strong inhibitors not recommended; if unavoidable 100 → 75 mg (75 → 50 mg if already reduced). Strong inducers contraindicated. Washouts: inhibitors 10 days, inducers 12 days (pp. 101, 108, 119). | **Both ½.** PageIndex had the 100 mg dose and the washouts but not the inhibitor dose reduction. pgvector had the washouts but not the 100 mg dose or the dose reduction. 10.1s / $0.183 vs 2.5s / $0.011 | PageIndex was complete in 1 of 4 runs. pgvector ½ in 3 of 3. **Show it as "neither is reliable on a 3-chapter question"** |
| 13 | Lorlatinib | List all of the Phase 2 expansion cohorts (EXP-1 to EXP-6) and which patients go into each. | EXP-1 treatment-naïve ALK+ … EXP-6 ROS1+ (p.22, again in eligibility pp. 85–97). | Both ✔, all 6 including 3A/3B, with enrolment targets. 8.5s / $0.124 vs 4.3s / $0.015 | Stable |
| 14 | SPRINT | List all of the exclusion criteria, numbered. | 20 criteria, pp. 18–20. | **Both listed all 20.** 11.5s / $0.076 vs 6.9s / $0.019 | Stable |
| 15 | SPRINT | What are the systolic blood pressure targets in the two treatment arms? | < 120 vs < 140 mm Hg (p.25). | Both ✔. pgvector added "designed to achieve 135–139". 5.3s vs 1.7s | Stable |
| 16 | SPRINT | How is adherence to antihypertensive medication monitored, and when? | Adherence scale at baseline, 12 and 48 months, plus a one-item screen at every medication visit (p.37). | Both ✔. PageIndex cited the single right page (p.37); pgvector 9 pages. 6.8s vs 2.2s | Stable |

**Say:** "Even on 200-page protocols, a question with distinctive terms is found by vector
search, partly because protocols repeat key facts in the synopsis. Document length alone
doesn't decide it. Questions that need rules from several chapters (#12) are hard for both."

## Act 6: where pgvector fails and PageIndex answers

These questions target known blind spots of top-8 vector retrieval. **Both runs gave
identical scores** (a screening run and the end-to-end app run), so they are safe to show live.
Look at **Pages retrieved** on the pgvector card: it shows *why* it failed (for example, on #18 every
retrieved page is from the wrong document).

| # | Scope | Question | Correct answer | Measured: PageIndex vs pgvector | Why vector retrieval fails |
|---|---|---|---|---|---|
| 18 | Library | Which of the drug labels tells you to withhold the drug when left ventricular ejection fraction drops, and what exactly should be done? | **Mekinist** label: LVEF drop ≥ 10% from baseline and below the lower limit of normal → withhold up to 4 weeks; resume at a lower dose if it recovers, otherwise discontinue. | **PageIndex ✔** (Mekinist, 1 page). **pgvector ✘**: answered from the *lorlatinib protocol's* LVEF rules. 8.6s / $0.075 vs 3.7s / $0.012 | **Crowding.** All 8 closest passages are in the 208-page protocol, which has pages of LVEF rules. The 26-page label never makes the top 8 |
| 19 | Mekinist | What does section 8.6 of this label say? | 8.6 Hepatic Impairment: no dose adjustment for mild impairment; dose not established for moderate or severe. | **PageIndex ✔** (2 pages, both section 8.6). **pgvector ✘ "Not found"** (7 Mekinist pages, none of them 8.6). 7.4s / $0.067 vs 1.7s / $0.009 | **Section numbers mean nothing to an embedding.** "Section 8.6" isn't similar to the hepatic-impairment text. PageIndex navigates the numbered tree |
| 20 | Library | For each of the five drug labels in the library, what is the maximum recommended daily dose? | Naproxen 1,500 mg/day (the label's "maximum recommended human daily dose"); Mekinist 2 mg; glimepiride 8 mg; varenicline 1 mg twice daily; tamsulosin 0.8 mg. | **PageIndex ✔ all 5** (one page from each label). **pgvector ✘ "Not found"**. 10.6s / $0.125 vs 2.4s / $0.009 | **Coverage.** 8 passages can't cover 5 documents. pgvector's came from varenicline (1 page) and the lorlatinib protocol (11 pages). PageIndex opened one page in each of the 5 labels |
| 21 | Library | Which documents in the library are clinical trial protocols, and what is each one studying? | NCT01970865 lorlatinib in ALK+/ROS1+ NSCLC; NCT01206062 SPRINT, intensive (< 120) vs standard (< 140 mm Hg) SBP targets. | **PageIndex ✔**. **pgvector ½** (missed what one study compares). 8.4s / $0.162 vs 3.2s / $0.011 | **A question about the library, not its text.** PageIndex picks whole documents first; vector search sees only passages, here 10 of 13 pages from one protocol |
| 22 | Lorlatinib | List every protocol amendment with its date, from Amendment 1 to the latest. | Amendments 1–8, 17 Sep 2013 … 23 Jan 2019. | **PageIndex ✔ all 8.** **pgvector ½**: missed Amendment 7. 6.1s / $0.118 vs 4.2s / $0.013 | **A list spread over 10 pages** (pp. 2–11). Top 8 chunks drop one |
| 23 | Mekinist | What drug interactions does this label describe? | Section 7 refers to the dabrafenib label; 12.3: no change in trametinib AUC with dabrafenib, no clinically relevant effect on CYP3A4 substrates, CYP2C8 inhibitor in vitro, P-gp/BSEP substrate. | **PageIndex ✔**. **pgvector ½**: had 12.3 but missed section 7. 9.4s / $0.071 vs 3.5s / $0.013 | **The answer lives in two sections.** The short section-7 note doesn't look like "drug interactions" text |

**Say:** "These are the failure modes to design for: a big document drowning out a small one (#18),
references to the document's own structure (#19), questions that need every document (#20, #21),
and answers longer than 8 chunks (#22). PageIndex reads the document's structure, so it handles
them. Vector search only sees the passages nearest the question."

Two questions I tested did *not* make the cut: following the boxed warning's cross-references in
naproxen (both ½), and "which documents restrict strong CYP3A4 inhibitors" (pgvector ✔, PageIndex ½).

---

## The takeaway slide

| Use PageIndex when… | pgvector is enough when… |
|---|---|
| One answer must be **collected across several documents** (#6, #20, #21) | The question names a **distinctive fact** (dose, storage, interaction, criterion): 16 of 17 typical questions scored the same |
| A **large document can drown out a small one** in the same library (#18) | |
| The question refers to the **document's own structure**: a section number, "all amendments", a document type (#19, #21, #22) | |
| You need **pinpoint page citations** for audit or review (PageIndex cites the 1–4 pages it read; pgvector cites the 6–15 pages it retrieved) | You search a **library for one buried fact** (#7: pgvector 3 of 3, PageIndex 1 of 4) |
| You want the system to **explain absence**, e.g. "migraine isn't an indication" (#10) | **Latency** matters: 2.7s vs 8.0s per answer |
|  | **Cost** matters at volume: about 7× cheaper at list price ($0.011 vs $0.075 per answer) |
|  | You need **repeatable** answers: one retrieval and one call. PageIndex's #7, #11 and #12 changed between runs; pgvector's answers were stable |

**Bottom line:** on typical questions (Acts 1–5) accuracy is a near-tie, 16.5 vs 16 of 17, and
PageIndex costs about 7× more and takes 3× longer. On the questions in Act 6, which need structure,
coverage or a big library, PageIndex wins clearly: 6 vs 1.5 of 6, the same in both runs. Choose by
which kind of question your users ask. A common pattern is pgvector for everyday lookups, with an
agent like PageIndex for "find all / per section / across documents" questions.

**Cost note.** List price assumes no prompt caching. In practice PageIndex's agent re-sends
the same prefix every turn, so OpenAI caching roughly halves its bill (billed: $0.71 for the
17). Repeated identical questions also hit the cache for pgvector. Either way the ratio stays
around 4–7×.

**Caveats to say out loud**

- 23 hand-picked questions are anecdotes. Act 6 was chosen to expose vector weaknesses, the
  same way #7 shows a PageIndex weakness. For accuracy numbers to report, run the full
  benchmark (`bench.run_benchmark`), which uses an LLM judge and gold pages.
- The labels are short (11–26 pages). pgvector's 8 chunks cover roughly 10–30% of each label.
  On the long protocols pgvector still tied on 5 of 6 questions.
- pgvector here has no reranker and no keyword search. A tuned hybrid search or a reranker
  could improve it (see below).
- Known issue: the corpus builder cuts FDA-label section titles to their first word (for
  example "5.6 Renal") and reads some table rows as headings. This may make PageIndex's
  navigation of the labels slightly worse than on the real documents. The trial protocols
  are the original PDFs and aren't affected.

## Appendix: pgvector notes for the technical audience

- **Setup:** PostgreSQL 17.11, pgvector 0.8.7, HNSW (m=16, ef_construction=64, ef_search=100).
  The 7 documents are 673 chunks: 15 MB with indexes, loaded in 13s for $0.007. A search
  takes about 2–6 ms (library scope is the slowest), which is negligible next to the
  1.5–3s answering call.
- **Filtered-search trap:** on a large table, an HNSW search restricted to one document
  can return fewer than 8 chunks. The index finds the nearest candidates first and filters
  them afterwards. We enable `hnsw.iterative_scan = relaxed_order` (pgvector ≥ 0.8) to prevent it.
  At 673 chunks Postgres filters first, so it doesn't occur here; check it on the full corpus.
- **Hybrid search is not a free win.** `vector_pg_hybrid` adds Postgres full-text search
  merged with reciprocal rank fusion. Untuned, it scored **11 / 17**: keyword hits on generic
  question words ("list", "dose", "patients") pushed out the chunks that held the answer
  (about 45% of retrieved chunks changed). It did find the 100 mg dose on #12. It needs
  weighting, rare-term-only keyword queries or a reranker before it's worth using.
