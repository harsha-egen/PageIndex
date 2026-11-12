"""Grading. Two independent signals per answer:

1. judge_score  (0 / 0.5 / 1) — an LLM judge compares the answer to the gold answer.
   The judge is blind to which pipeline produced the answer. Its cost is tracked but
   NOT charged to either pipeline.
2. page_hit / page_recall — did the pipeline actually retrieve (vector) or cite
   (PageIndex) the gold page(s)? This is the traceability metric clinicians care about.
"""
from __future__ import annotations

import json

JUDGE_SYSTEM = """You grade answers to clinical document questions against a gold answer.
Score 1 if the answer states the gold fact correctly (numbers, units, dose, frequency, criterion must match;
extra correct detail is fine). Score 0.5 if partly correct or correct but hedged with a wrong alternative.
Score 0 if wrong, missing, or 'not found'. Exception: if the gold answer says the information is NOT in the
document, score 1 when the candidate says it is not found / not provided, and 0 if it invents an answer.
Return JSON {"score": 0|0.5|1, "reason": str}."""


def page_metrics(retrieved: list[dict], doc_id: str, gold_pages: list[int]) -> dict:
    got = {r["page"] for r in retrieved if r["doc_id"] == doc_id}
    hits = [p for p in gold_pages if p in got]
    wrong_doc = bool(retrieved) and not any(r["doc_id"] == doc_id for r in retrieved)
    return {"page_hit": int(bool(hits)), "page_recall": len(hits) / len(gold_pages),
            "retrieved_pages_n": len(retrieved), "wrong_doc": int(wrong_doc), "cited": int(bool(retrieved))}


def judge(client, model: str, question: str, gold: str, answer: str) -> dict:
    r = client.chat.completions.create(
        model=model, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": f"Question: {question}\nGold answer: {gold}\nCandidate answer: {answer}"}])
    try:
        js = json.loads(r.choices[0].message.content)
        return {"judge_score": float(js["score"]), "judge_reason": js.get("reason", "")}
    except Exception:
        return {"judge_score": None, "judge_reason": "unparseable judge output"}
