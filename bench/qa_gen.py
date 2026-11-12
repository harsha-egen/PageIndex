"""Step 2 — build the question set with gold answers AND gold pages.

Question types (each tagged so results can be sliced on the dashboard):
  lookup       one fact on one page of a long document         (both should do OK)
  multi_hop    needs two pages from different sections          (PageIndex's home turf)
  section_ref  phrased the way clinicians ask ("per the dose-modification section ...")
  cross_ref    FDA label statements that point to another section ("see 5.3")
  patient      exact fact from a short patient note             (vector RAG's home turf)

Every LLM-written question must quote evidence that is verified to exist on the gold
page(s); unverifiable questions are discarded. Note questions are templated from the
generator's ground-truth facts, so they need no LLM at all.

Clinicians should review a sample of questions before the final run (see README).
"""
from __future__ import annotations

import json
import random
import re

from bench.common import ROOT, load_config, load_manifest, read_pdf_pages, read_pdf_toc, write_jsonl

GEN_SYSTEM = """You write evaluation questions for a clinical document QA benchmark.
Return JSON: {"question": str, "answer": str, "evidence": [str, ...]}.
Rules:
- The question must be answerable ONLY from the supplied page text(s), with one exact answer
  (a dose, criterion, threshold, timepoint, count, or named item).
- Name the document naturally in the question using the identifier given (so it is unambiguous
  in a library of hundreds of documents), but do NOT copy section headings verbatim.
- Paraphrase: avoid reusing long phrases from the page, as a real clinician would.
- evidence: 1 short verbatim quote (<= 25 words) per page used, copied exactly.
- For two pages, the question must need BOTH pages (e.g. criterion on one, its consequence on the other)."""


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def _verify(evidence: list[str], pages_text: list[str]) -> bool:
    if not evidence or len(evidence) < len(pages_text):
        return False
    return all(_norm(q)[:120] in _norm(t) for q, t in zip(evidence, pages_text))


def _good_pages(pages: list[str]) -> list[int]:
    n = len(pages)
    return [i + 1 for i, p in enumerate(pages) if len(p) > 900 and 3 < i + 1 < n - 1]


def _identifier(doc: dict) -> str:
    if doc["doc_type"] == "trial_protocol":
        return f"the protocol for trial {doc['doc_id']} ({doc['title'][:80]})"
    if doc["doc_type"] == "fda_label":
        return f"the {doc['brand']} ({doc['drug']}) prescribing information"
    return f"patient MRN {doc.get('mrn')}"


def _ask(client, model: str, ident: str, texts: list[str]) -> dict | None:
    body = "\n\n".join(f"--- PAGE {i + 1} ---\n{t[:6000]}" for i, t in enumerate(texts))
    r = client.chat.completions.create(
        model=model, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": GEN_SYSTEM},
                  {"role": "user", "content": f"Document identifier: {ident}\n\n{body}"}])
    try:
        return json.loads(r.choices[0].message.content)
    except Exception:
        return None


def _section_pages(toc, n_pages) -> list[tuple[str, int, int]]:
    """(title, start, end) for top-level-ish sections from the PDF outline."""
    secs = [(t, p) for lvl, t, p in toc if lvl <= 2 and p > 0]
    out = []
    for i, (t, p) in enumerate(secs):
        end = secs[i + 1][1] - 1 if i + 1 < len(secs) else n_pages
        out.append((t, p, max(p, end)))
    return out


def gen_for_long_doc(client, model, doc, n_q, rng) -> list[dict]:
    pages = read_pdf_pages(doc["path"])
    cands = _good_pages(pages)
    if len(cands) < 4:
        return []
    secs = _section_pages(read_pdf_toc(doc["path"]), len(pages))
    plan = (["lookup", "multi_hop", "section_ref", "multi_hop"] if doc["doc_type"] == "trial_protocol"
            else ["lookup", "cross_ref", "multi_hop"]) * 3
    plan = plan[:n_q]
    out, tries = [], 0
    ident = _identifier(doc)
    for qtype in plan:
        for _ in range(3):  # retries per question
            tries += 1
            if qtype == "multi_hop" and len(secs) >= 2:
                a, b = rng.sample([s for s in secs if s[2] - s[1] < 40] or secs, 2)
                pa = rng.choice([p for p in range(a[1], a[2] + 1) if p in cands] or cands)
                pb = rng.choice([p for p in range(b[1], b[2] + 1) if p in cands and p != pa] or [c for c in cands if c != pa])
                gold = sorted({pa, pb})
            elif qtype == "cross_ref":
                refs = [i + 1 for i, p in enumerate(pages) if re.search(r"\((?:see )?\d{1,2}\.\d{1,2}\)", p) and i + 1 in cands]
                gold = [rng.choice(refs or cands)]
            else:
                gold = [rng.choice(cands)]
            texts = [pages[p - 1] for p in gold]
            js = _ask(client, model, ident, texts)
            if js and js.get("question") and _verify(js.get("evidence", []), texts):
                out.append({"q_id": f"{doc['doc_id']}-q{len(out) + 1}", "doc_id": doc["doc_id"],
                            "doc_type": doc["doc_type"], "q_type": qtype, "question": js["question"],
                            "gold_answer": js["answer"], "gold_pages": gold, "evidence": js["evidence"],
                            "doc_pages": doc["pages"]})
                break
    return out


def gen_for_note(doc, rng) -> dict:
    f, mrn = doc["facts"], doc["mrn"]
    choices = []
    if f.get("allergy"):
        choices.append((f"What drug allergy is documented for patient MRN {mrn}?", f["allergy"]))
    if f.get("labs"):
        lab = rng.choice(f["labs"])
        name, val = lab.split(":", 1)
        choices.append((f"What was the {name.strip()} result recorded for patient MRN {mrn}?", val.strip()))
    if f.get("meds"):
        med = rng.choice(f["meds"])
        choices.append((f"What dose and frequency of {med.split()[0]} is patient MRN {mrn} taking?", med))
    if f.get("bp"):
        choices.append((f"What blood pressure was recorded at the visit for patient MRN {mrn}?", f["bp"] + " mmHg"))
    q, a = rng.choice(choices)
    return {"q_id": f"{doc['doc_id']}-q1", "doc_id": doc["doc_id"], "doc_type": "clinical_note",
            "q_type": "patient", "question": q, "gold_answer": a, "gold_pages": [1],
            "evidence": [a], "doc_pages": doc["pages"]}


def main() -> None:
    from bench.tracing import load_env
    load_env()
    from openai import OpenAI
    cfg = load_config()
    sp, rng = cfg["scale_params"], random.Random(cfg["seed"])
    client, model = OpenAI(), cfg["models"]["qa_generator"]
    docs = load_manifest(cfg)
    qs: list[dict] = []
    for d in [d for d in docs if d["doc_type"] == "trial_protocol"]:
        got = gen_for_long_doc(client, model, d, sp["q_per_trial"], rng)
        qs += got
        print(f"  {d['doc_id']}: {len(got)} questions")
    for d in [d for d in docs if d["doc_type"] == "fda_label"]:
        qs += gen_for_long_doc(client, model, d, sp["q_per_label"], rng)
    notes = [d for d in docs if d["doc_type"] == "clinical_note"]
    qs += [gen_for_note(d, rng) for d in rng.sample(notes, min(sp["q_notes"], len(notes)))]
    out = ROOT / "results" / "questions.jsonl"
    write_jsonl(out, qs)
    from collections import Counter
    print(f"\n{len(qs)} questions -> {out}")
    print(Counter((q["doc_type"], q["q_type"]) for q in qs))


if __name__ == "__main__":
    main()
