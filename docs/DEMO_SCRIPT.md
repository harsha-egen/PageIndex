# Demo script: when PageIndex helps, and when vector RAG is enough

Run with `python -m bench.chat_app` on the 5 indexed FDA labels:

| Doc ID | Drug | Pages |
|---|---|---|
| LBL-000155a8 | Naproxen | 17 |
| LBL-0002ad27 | Mekinist (trametinib) | 26 |
| LBL-0003458f | Glimepiride | 11 |
| LBL-0004a0bb | Varenicline | 19 |
| LBL-00097e78 | Tamsulosin | 13 |
| NCT01970865 | Lorlatinib (PF-06463922) phase 1/2 protocol, ALK/ROS1 NSCLC. *No PDF outline* | 208 |
| NCT01206062 | SPRINT blood pressure trial protocol | 117 |

Every question below was dry-run on 2026-11-11 through both pipelines (same answering
model `gpt-5.6-terra`, medium reasoning). The "Measured" column is that run. Answers can
change from run to run (see step 11), so treat them as what to expect, not a guarantee.
Running the whole script costs about **$1.00** (Acts 1–4 about $0.50, Act 5 about $0.50).

**Scope** = what to pick in the document dropdown. "Library" = *Whole library*.

---

## Act 1: simple lookups. Vector RAG is enough

The point: when the answer is one distinctive fact, both get it right, and vector RAG is
3–4× faster and 2–3× cheaper. PageIndex's agent spends 3–4 LLM calls to find what one
embedding lookup finds.

| # | Scope | Question | Correct answer | Measured |
|---|---|---|---|---|
| 1 | Varenicline | What temperature should varenicline tablets be stored at? | 20–25 °C (68–77 °F); excursions 15–30 °C. Page 19. | Both correct. PageIndex 4.6s / $0.023 / 3 calls. Vector 1.5s / $0.009 |
| 2 | Tamsulosin | Can tamsulosin be taken together with ketoconazole? | No: 0.4 mg should not be used with strong CYP3A4 inhibitors such as ketoconazole. | Both correct. PageIndex 6.4s / $0.020. Vector 1.7s / $0.010, and added the PK numbers (Cmax ×2.2, AUC ×2.8) |

**Say:** "For FAQ-style lookups, PageIndex is paying for reasoning it doesn't need."

## Act 2: structure-heavy questions. Where PageIndex earns its cost

Open **Live API calls** on the PageIndex card while it runs: you can watch the agent open
the document, pick sections and read specific pages.

| # | Scope | Question | Correct answer | Measured |
|---|---|---|---|---|
| 3 | Mekinist | If a patient's LVEF drops 10% or more from baseline and below the lower limit of normal, what should be done, and exactly what dose do they resume at if it recovers? | Withhold up to 4 weeks. If LVEF recovers, resume at the first dose reduction (adult 2 mg → **1.5 mg** daily, Table 3, p.3). If not, discontinue. | **PageIndex** answered "1.5 mg" directly by following section 2.4 to the dose-reduction table (cited p.3–4). Vector said "no single exact dose is specified", then listed the table. 7.2s / $0.038 vs 2.5s / $0.010 |
| 4 | Mekinist | List every warning and precaution in section 5 of this label, one line each. | 13 subsections, 5.1 New Primary Malignancies … 5.13 Embryo-Fetal Toxicity (pp. 4–8). | Both listed all 13. **Look at the citations**: PageIndex cites pages 5–8 per item. Vector cites "p.4-5" for items that are on pages 5–7 |
| 5 | Naproxen | Why is naproxen contraindicated around CABG surgery? Follow the cross-reference and explain what the referenced warning section says. | Section 4 → see 5.1: COX-2 NSAID trials after CABG showed more MI and stroke. | Both correct. PageIndex cited **1 page** (p.4). Vector retrieved 7 pages. 6.8s / $0.027 vs 3.1s / $0.011 |

**Say:** "When the answer depends on the document's structure (a table referenced from
another section, a whole numbered section), PageIndex navigates it and its citations point
to the exact page. That's what matters for traceability."

## Act 3: across the whole library. Mixed results

| # | Scope | Question | Correct answer | Measured |
|---|---|---|---|---|
| 6 | Library | Which of these medicines need a lower starting dose in severe renal impairment, and what is that dose? | Varenicline 0.5 mg once daily (CrCl < 30). Glimepiride 1 mg once daily (any renal impairment). | **PageIndex found both**. Vector found only glimepiride ("no other medicine…"). 10.8s / $0.057 vs 3.7s / $0.011 |
| 7 | Library | Which drug should be taken at least 4 hours before colesevelam? | Glimepiride (sections 2.1, 7.4). | **PageIndex failed**: "Not found in the provided documents." Vector answered correctly in 1.5s |
| 8 | Library | Which of these drugs carries a boxed warning, and what is it about? | Only naproxen: cardiovascular thrombotic events and GI bleeding/ulceration/perforation. | Both correct. PageIndex 12.6s / $0.045, **9× slower and 5× more expensive** |

**Say:** "Questions that collect an answer from several documents (6) are where an agent
helps. Finding one fact buried somewhere in the library (7) is what vector search is built
for. PageIndex's agent first has to choose which documents to open, and a colesevelam note
inside a glimepiride label gives it little to go on."

## Act 4: honesty checks

| # | Scope | Question | Correct answer | Measured |
|---|---|---|---|---|
| 9 | Tamsulosin | My patient is having eye surgery next month. Is there anything they should tell the surgeon about this medicine? | Tell the ophthalmologist: intraoperative floppy iris syndrome (IFIS) during cataract or glaucoma surgery (5.5). | Both correct, even though the question never says "cataract" or "IFIS" |
| 10 | Naproxen | What is the recommended dose of naproxen for migraine? | Not in the label: migraine is not an indication. | Vector: a clean "Not found". PageIndex: "no migraine-specific dose", then offered the acute-pain dose. Helpful or overreaching? Good discussion point |
| 11 | Naproxen | What is the recommended starting dose, and how is it adjusted for renal impairment? | Indication-specific doses. Not recommended if CrCl < 30 mL/min (8.7, p.13). | **Run it twice.** In the first live run, PageIndex stopped after reading pp. 1–6 and missed the CrCl < 30 rule, which vector RAG found. In the dry run, PageIndex found it. Agents vary from run to run |

## Act 5: long protocols (117–208 pages). PageIndex's home turf?

PageIndex built a 203-section tree for the lorlatinib protocol even though the PDF has no
outline (indexing: 53s, $0.10). Vector RAG embedded it in 4s for $0.003.

| # | Scope | Question | Correct answer | Measured |
|---|---|---|---|---|
| 12 | Lorlatinib | What dose do Phase 2 patients receive, and what are the rules for strong CYP3A inhibitors and strong CYP3A inducers, including any washout before the first dose? | 100 mg once daily, 21-day cycles. Strong inhibitors not recommended; if unavoidable 100 → 75 mg (75 → 50 mg if already reduced). Strong inducers contraindicated. Exclusions: inhibitors within 10 days, inducers within 12 days before the first dose (pp. 101, 108, 119). | **PageIndex: complete**, combining 3 sections. **Vector missed the 100 mg dose** ("the Phase 1–determined RP2D") **and the inhibitor dose reduction**. 9.8s / $0.130 vs 3.1s / $0.011 (12× the cost) |
| 13 | Lorlatinib | List all of the Phase 2 expansion cohorts (EXP-1 to EXP-6) and which patients go into each. | EXP-1 treatment-naïve ALK+ … EXP-6 ROS1+ (p.22, again in eligibility pp. 85–97). | Both complete, all 6 including 3A/3B. Vector also gave enrolment targets. 12.5s / $0.079 vs 5.2s / $0.014 |
| 14 | SPRINT | List all of the exclusion criteria, numbered. | 20 criteria, pp. 18–20. | **Both listed all 20.** 11.2s / $0.048 vs 6.5s / $0.019 |
| 15 | SPRINT | What are the systolic blood pressure targets in the two treatment arms? | < 120 vs < 140 mm Hg (p.25). | Both correct. Vector added "designed to achieve 135–139". 6.5s / $0.032 vs 1.6s / $0.010 |
| 16 | SPRINT | How is adherence to antihypertensive medication monitored, and when? | Adherence scale at baseline, 12 and 48 months, plus a one-item screen at every medication visit (p.37). | Both correct. PageIndex cited the single right page (p.37). 6.1s / $0.035 vs 2.2s / $0.010 |
| 17 | Library | Which study requires stopping strong CYP3A4 inducers at least 12 days before the first dose of study drug? | The lorlatinib study (B7461001), exclusion 17, p.101. | Both correct this time (compare #7, where PageIndex missed). 7.3s / $0.067 vs 1.5s / $0.009 |

**Say:** "Even on 200-page protocols, a well-phrased question with distinctive terms is found
by vector search, partly because protocols repeat key facts in the synopsis. The gap opens
when one answer needs **several rules from different chapters** (#12): vector RAG returns the
8 closest chunks and stops; PageIndex keeps reading until the question is covered. That
thoroughness costs about 10× more per question."

---

## The takeaway slide

| Use PageIndex when… | Use vector RAG when… |
|---|---|
| The answer depends on document structure: a section, a table referenced from another section, "see 5.1" | The answer is one distinctive fact (storage, a named interaction, a contraindication) |
| You need exact page citations for audit or review | Latency matters: 1–4s vs 5–13s here |
| The question gathers facts from several documents (Act 3, #6) | Cost matters at volume: 2–5× cheaper per question here |
| One answer needs several rules from different chapters of a long document (Act 5, #12) | You search a large library for one fact (Act 3, #7) |
|  | You need repeatable answers: one retrieval and one call is more stable than an agent loop |

**Caveats to say out loud**

- The labels are short (11–26 pages), which favours vector RAG: its 8 chunks of 512 tokens
  cover roughly 10–30% of each label. On the long protocols (Act 5) vector RAG still tied on
  5 of 6 questions, so length alone is not the deciding factor; questions that span chapters are.
- 17 questions are anecdotes. The accuracy comparison is the full benchmark (`bench.run_benchmark`), with an LLM judge and gold pages.
- The vector baseline here uses exact in-memory search with no reranker: a best case for speed.
- Known issue: the corpus builder cuts section titles to their first word (for example "5.6 Renal")
  and reads some table rows as headings. This may make PageIndex's tree navigation slightly
  worse than it would be on the real labels.
